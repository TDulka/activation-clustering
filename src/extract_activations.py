# %% Imports and Logging Setup
# Core ML/DL libraries for model handling and tensor operations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pyarrow as pa
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Tuple
import torch.utils.data as data
from tqdm import tqdm

# Configure logging with timestamps for tracking extraction progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% Configuration
@dataclass
class ExtractionConfig:
    """Configuration for the activation extraction process.
    
    Controls model loading, batch processing, and output parameters.
    Default values are set for common use cases but can be overridden.
    """
    model_name: str = "google/gemma-2-2b"  # Model to extract activations from
    target_layer: int = 12               # Which transformer layer to extract from
    batch_size: int = 32                 # Number of sequences to process at once
    max_length: int = 512               # Maximum sequence length (longer sequences will be truncated)
    num_samples: int = 1000            # Number of samples to extract
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Computation device
    dtype: torch.dtype = torch.float32   # Precision for activation values
    output_dir: Path = Path("/workspace/data/activations")  # Where to save extracted activations

# %% Dataset handling
class PileDataset(data.Dataset):
    """Handles streaming and preprocessing of the Pile dataset.
    
    Implements PyTorch Dataset interface for efficient batching and loading.
    Preprocesses text into tokens and buffers them in memory for faster access.
    """
    def __init__(self, tokenizer, max_length: int, num_samples: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        # Initialize streaming dataset
        self.dataset = load_dataset(
            "monology/pile-uncopyrighted", 
            split="train", 
            streaming=True
        )
        
        # Pre-tokenize and buffer data for efficiency
        self.data = []
        for item in self.dataset.take(self.num_samples):
            tokens = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            self.data.append(tokens.input_ids[0])

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a single tokenized sample"""
        return self.data[idx]

# %% Model handling
class ActivationExtractor:
    """Handles model loading and activation extraction.
    
    Manages the ML model and provides methods to extract activations
    from specific layers during forward passes.
    """
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    @torch.no_grad()  # Disable gradient computation for memory efficiency
    def extract_batch(self, 
                     input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract activations from a batch of input sequences.
        
        Args:
            input_ids: Tensor of token IDs to process
            
        Returns:
            Tuple of (activations, token_ids) where activations are the neural
            activations from the target layer and token_ids are the original
            input token IDs.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")

        # Process inputs through model and extract target layer activations
        input_ids = input_ids.to(self.config.device)
        
        outputs = self.model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Access hidden states more safely
        if hasattr(outputs, 'hidden_states'):
            activations = outputs.hidden_states[self.config.target_layer]
        else:
            raise ValueError("Model did not return hidden states. Check model configuration.")
        
        # Return the original token IDs instead of converting to strings
        return activations, input_ids

    def setup(self):
        """Load model and tokenizer with specified configuration"""
        logger.info(f"Loading {self.config.model_name}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=self.config.dtype,
            output_hidden_states=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        return self

# %% Storage handling
class ActivationStorage:
    """Manages persistent storage of extracted activations.
    
    Handles conversion of activations to Arrow format and saves them
    to disk in chunks for efficient storage and later analysis.
    """
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.setup_storage()
        
    def setup_storage(self):
        """Create necessary directories for storing activations"""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
    
    def save_chunk(self, 
                  activations: torch.Tensor, 
                  token_ids: torch.Tensor, 
                  chunk_id: int) -> Path:
        """Save a batch of activations and their corresponding token IDs.
        
        Args:
            activations: Neural activations from the model (float32)
            token_ids: Token IDs from the tokenizer (int32)
            chunk_id: Unique identifier for this chunk
            
        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / "chunks" / f"chunk_{chunk_id:04d}.npz"
        np.savez_compressed(
            output_path,
            activations=activations.cpu().numpy(),
            token_ids=token_ids.cpu().numpy().astype(np.int32)
        )
        return output_path

# %% Main extraction pipeline
def run_extraction(config: ExtractionConfig):
    """Main pipeline for extracting and saving neural activations.
    
    This function orchestrates the entire extraction process by:
    1. Setting up model and dataset infrastructure
    2. Processing data in batches for memory efficiency
    3. Extracting neural activations from specified layers
    4. Saving results to disk in a structured format
    5. Logging progress and performance statistics
    
    Args:
        config: ExtractionConfig object containing all runtime parameters
        
    Note:
        - Handles memory management through batching
        - Uses Arrow format for efficient storage
        - Provides progress logging for long-running extractions
    """
    # Initialize core components
    extractor = ActivationExtractor(config).setup()
    storage = ActivationStorage(config)
    
    # Create dataset with appropriate tokenization
    dataset = PileDataset(extractor.tokenizer, config.max_length, config.num_samples)
    
    # Setup batched processing pipeline
    dataloader = data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0  # Single worker for deterministic processing
    )
    
    # Process each batch and extract activations
    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Processing batches", total=len(dataloader)):
      
        # Extract neural activations and corresponding tokens
        activations, tokens = extractor.extract_batch(batch)
        
        output_path = storage.save_chunk(activations, tokens, batch_idx)
        logger.info(f"Saved chunk to {output_path}")
        
        # Log batch statistics for monitoring
        logger.info(
            f"Batch stats - Shape: {activations.shape}, "
            f"Mean: {activations.mean():.4f}, "
            f"Std: {activations.std():.4f}"
        )

# %% Interactive running
if __name__ == "__main__":
    """Entry point for extraction script.
    """
    config = ExtractionConfig(
        max_length=128,
        num_samples=30000
    )
    
    # Run extraction pipeline
    run_extraction(config)
