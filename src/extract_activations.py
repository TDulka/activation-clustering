# %% Imports and Logging Setup
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
from dataclasses import dataclass
from typing import Optional, List
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from pathlib import Path

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
    batch_size: int = 32
    max_length: int = 128
    num_samples: int = 300_000
    chunk_size: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16  # Changed to float16 for storage efficiency
    output_dir: str = "./processed_activations"
    exclude_tokens: List[str] = None

    def __post_init__(self):
        self.exclude_tokens = self.exclude_tokens or ["bos", "pad"]
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

# %% Dataset handling
class StreamingPileDataset(data.IterableDataset):
    """Streams data from the Pile dataset with token filtering."""

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

    def __iter__(self):
        count = 0
        for item in self.dataset:
            if count >= self.num_samples:
                break

            tokens = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            yield {
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0],
            }
            count += 1

class IncrementalWhitener:
    """Implements online computation of whitening statistics."""

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.reset_statistics()

    def reset_statistics(self):
        self.mean = torch.zeros(self.hidden_dim, dtype=torch.float32)
        self.S = torch.zeros(self.hidden_dim, self.hidden_dim, dtype=torch.float32)
        self.n = 0

    def update(self, batch_activations: torch.Tensor, attention_mask: torch.Tensor):
        """Update statistics using Welford's online algorithm."""
        # Flatten valid activations using attention mask
        valid_activations = batch_activations[attention_mask.bool()]

        batch_size = valid_activations.shape[0]
        if batch_size == 0:
            return

        # Update mean and covariance
        delta = valid_activations - self.mean
        self.n += batch_size
        self.mean += delta.sum(0) / self.n
        delta2 = valid_activations - self.mean
        self.S += (delta * delta2.T).sum(0)

    def finalize(self):
        """Compute final whitening parameters."""
        cov = self.S / (self.n - 1)
        U, S, _ = torch.svd(cov)
        self.transform = U @ torch.diag(S.pow(-0.5))

    def save(self, path: str):
        """Save whitening parameters."""
        torch.save(
            {"mean": self.mean, "transform": self.transform, "n_samples": self.n}, path
        )

    def load(self, path: str):
        """Load whitening parameters."""
        params = torch.load(path)
        self.mean = params["mean"]
        self.transform = params["transform"]
        self.n = params["n_samples"]

class ActivationExtractor:
    """Handles model loading and activation extraction with whitening."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._setup_model()
        self.whitener = None

    def _setup_model(self):
        """Initialize model and tokenizer."""
        if self.config.model is None or self.config.tokenizer is None:
            if self.config.model_name is None:
                raise ValueError(
                    "Either model/tokenizer instances or model_name must be provided"
                )

            logger.info(f"Loading {self.config.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=self.config.dtype,
                output_hidden_states=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            self.model = self.config.model
            self.tokenizer = self.config.tokenizer

    @torch.no_grad()
    def compute_whitening_statistics(self):
        """First pass: Compute whitening statistics."""
        dataset = StreamingPileDataset(
            self.tokenizer, self.config.max_length, self.config.num_samples
        )
        dataloader = data.DataLoader(dataset, batch_size=self.config.batch_size)

        self.whitener = IncrementalWhitener(self.model.config.hidden_size)

        for batch in tqdm(dataloader, desc="Computing whitening statistics"):
            # Extract activations
            outputs = self.model(
                batch["input_ids"].to(self.config.device), output_hidden_states=True
            )
            activations = outputs.hidden_states[-1]

            # Update statistics
            self.whitener.update(activations.cpu(), batch["attention_mask"])

        self.whitener.finalize()
        self.whitener.save(f"{self.config.output_dir}/whitening_params.pt")

    @torch.no_grad()
    def process_and_store_whitened(self):
        """Second pass: Apply whitening and store results."""
        if self.whitener is None:
            raise ValueError("Must compute whitening statistics first!")

        dataset = StreamingPileDataset(
            self.tokenizer, self.config.max_length, self.config.num_samples
        )
        dataloader = data.DataLoader(dataset, batch_size=self.config.batch_size)

        chunk_activations = []
        chunk_masks = []
        chunk_count = 0

        for batch in tqdm(dataloader, desc="Processing activations"):
            outputs = self.model(
                batch["input_ids"].to(self.config.device), output_hidden_states=True
            )
            activations = outputs.hidden_states[-1].cpu()

            # Apply whitening
            whitened = (activations - self.whitener.mean) @ self.whitener.transform

            chunk_activations.append(whitened.to(torch.float16))
            chunk_masks.append(batch["attention_mask"])

            if (
                len(chunk_activations) * self.config.batch_size
                >= self.config.chunk_size
            ):
                self._save_chunk(chunk_activations, chunk_masks, chunk_count)
                chunk_activations = []
                chunk_masks = []
                chunk_count += 1

        # Save any remaining data
        if chunk_activations:
            self._save_chunk(chunk_activations, chunk_masks, chunk_count)

    def _save_chunk(self, activations, masks, chunk_id):
        """Save a chunk of processed activations."""
        activations = torch.cat(activations, dim=0)
        masks = torch.cat(masks, dim=0)

        np.savez_compressed(
            f"{self.config.output_dir}/chunk_{chunk_id}.npz",
            activations=activations.numpy(),
            attention_mask=masks.numpy(),
        )

# %% Main execution
if __name__ == "__main__":
    config = ExtractionConfig(
        model_name="google/gemma-2b",
        num_samples=300_000,
        output_dir="./processed_activations",
    )

    extractor = ActivationExtractor(config)

    # First pass: compute whitening statistics
    logger.info("Computing whitening statistics...")
    extractor.compute_whitening_statistics()

    # Second pass: process and store whitened activations
    logger.info("Processing and storing whitened activations...")
    extractor.process_and_store_whitened()
