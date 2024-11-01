# %% Imports and Logging Setup
# Core ML/DL libraries for model handling and tensor operations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
from dataclasses import dataclass
from typing import Optional
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
    """Configuration for the activation extraction process."""

    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_name: Optional[str] = None
    target_layer: int = 12
    batch_size: int = 32
    max_length: int = 512
    num_samples: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

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

    def preview_samples(self, n=10):
        """Print n samples from the dataset with their decoded text"""
        for i in range(min(n, len(self.data))):
            # Decode the token IDs back to text
            decoded_text = self.tokenizer.decode(self.data[i])
            print(f"\nSample {i+1}:")
            print("-" * 50)
            print(decoded_text)
            print("-" * 50)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a single tokenized sample"""
        return self.data[idx]


class ActivationExtractor:
    """Handles model loading and activation extraction."""

    def __init__(
        self, config: ExtractionConfig, dataset: Optional[data.Dataset] = None
    ):
        self.config = config
        self.model = config.model
        self.tokenizer = config.tokenizer
        self.dataset = dataset

        if self.model is None or self.tokenizer is None:
            if config.model_name is None:
                raise ValueError(
                    "Either model/tokenizer instances or model_name must be provided"
                )

            logger.info(f"Loading {config.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map="auto",
                torch_dtype=config.dtype,
                output_hidden_states=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    @torch.no_grad()
    def extract_activations(self) -> torch.Tensor:
        """Extract activations from the dataset and return as a single tensor."""
        # Create dataset and dataloader
        if self.dataset is None:
            self.dataset = PileDataset(
                self.tokenizer, self.config.max_length, self.config.num_samples
            )

        dataloader = data.DataLoader(
            self.dataset, batch_size=self.config.batch_size, num_workers=0
        )

        # Extract activations batch by batch
        activations_list = []

        for batch in tqdm(dataloader, desc="Extracting activations"):
            # Move batch to device
            batch = batch.to(self.config.device)

            # Get model outputs
            outputs = self.model(batch, output_hidden_states=True, return_dict=True)

            # Extract activations from target layer
            batch_activations = outputs.hidden_states[self.config.target_layer]

            # Move to CPU to save GPU memory
            activations_list.append(batch_activations.cpu())

            # Log batch statistics
            logger.info(
                f"Batch stats - Shape: {batch_activations.shape}, "
                f"Mean: {batch_activations.mean():.4f}, "
                f"Std: {batch_activations.std():.4f}"
            )

        # Concatenate all batches
        all_activations = torch.cat(activations_list, dim=0)
        logger.info(f"Final activations shape: {all_activations.shape}")

        return all_activations


def extract_activations(config: ExtractionConfig) -> torch.Tensor:
    """Convenience function to extract activations in one go."""
    extractor = ActivationExtractor(config)
    return extractor.extract_activations()


# %% Interactive running
if __name__ == "__main__":
    """Entry point for extraction script.
    """
    config = ExtractionConfig(
        model_name="google/gemma-2-2b", max_length=128, num_samples=1000, batch_size=32
    )

    activations = extract_activations(config)
    print(f"Extracted activations tensor: {activations.shape}")
