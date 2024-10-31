from pathlib import Path
from typing import List, Optional
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionalWhitening(nn.Module):
    def __init__(self, seq_len: int, n_features: int, epsilon: float = 1e-6):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.epsilon = epsilon
        
        # Per-position parameters
        self.register_buffer('weight', None)  # [seq_len, n_features, n_features]
        self.register_buffer('mean', None)    # [seq_len, n_features]
        self.register_buffer('reverse_weight', None)  # [seq_len, n_features, n_features]

    def fit(self, chunk_paths: list[Path]):
        n_samples = 0
        mean_sum = torch.zeros(self.seq_len, self.n_features, device='cuda')
        cov_sum = torch.zeros(self.seq_len, self.n_features, self.n_features, device='cuda')
        
        # First pass: compute means
        for path in tqdm(chunk_paths, desc="Computing means"):
            chunk = torch.from_numpy(np.load(path)['activations']).cuda()
            batch_size = chunk.shape[0]
            n_samples += batch_size
            mean_sum += chunk.sum(dim=0)
        
        position_means = mean_sum / n_samples
        
        # Second pass: compute covariance
        for path in tqdm(chunk_paths, desc="Computing covariance"):
            chunk = torch.from_numpy(np.load(path)['activations']).cuda()
            for pos in range(self.seq_len):
                pos_data = chunk[:, pos, :]  # [batch_size, n_features]
                delta = pos_data - position_means[pos].unsqueeze(0)
                cov_sum[pos] += delta.T @ delta
        
        position_covs = cov_sum / (n_samples - 1)
        
        # Compute transforms for each position
        weights = []
        reverse_weights = []
        
        for pos in range(self.seq_len):
            eigenvalues, eigenvectors = torch.linalg.eigh(position_covs[pos])
            
            # Forward transform
            weights.append(
                eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues + self.epsilon)) @ eigenvectors.T
            )
            
            # Reverse transform
            reverse_weights.append(
                eigenvectors @ torch.diag(torch.sqrt(eigenvalues + self.epsilon)) @ eigenvectors.T
            )
        
        self.weight = torch.stack(weights)
        self.mean = position_means
        self.reverse_weight = torch.stack(reverse_weights)
        
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, n_features]
        return torch.stack([
            (x[:, pos] - self.mean[pos]) @ self.weight[pos]
            for pos in range(self.seq_len)
        ], dim=1)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, n_features]
        return torch.stack([
            (x[:, pos] @ self.reverse_weight[pos]) + self.mean[pos]
            for pos in range(self.seq_len)
        ], dim=1)

def get_chunk_paths(data_dir: Path) -> List[Path]:
    """Get sorted list of all activation chunk paths."""
    return sorted(data_dir.glob("chunk_*.npz"))

def save_whitened_chunks(
    whitening: PositionalWhitening,
    input_paths: List[Path],
    output_dir: Path,
    batch_size: Optional[int] = None
):
    """
    Apply whitening transform to chunks and save results.
    
    Args:
        whitening: Fitted PositionalWhitening instance
        input_paths: List of paths to input chunks
        output_dir: Directory to save whitened chunks
        batch_size: Optional batch size for processing larger chunks
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    whitening.eval()  # Ensure in eval mode
    
    with torch.no_grad():
        for i, path in enumerate(tqdm(input_paths, desc="Whitening chunks")):
            # Load chunk and tokens
            data = np.load(path)
            chunk = torch.from_numpy(data['activations']).cuda()
            tokens = data['tokens']
            
            # Process in batches if specified
            if batch_size is not None:
                whitened_chunks = []
                for j in range(0, len(chunk), batch_size):
                    batch = chunk[j:j + batch_size]
                    whitened_chunks.append(whitening(batch).cpu().numpy())
                whitened = np.concatenate(whitened_chunks, axis=0)
            else:
                whitened = whitening(chunk).cpu().numpy()
            
            # Save whitened chunk
            output_path = output_dir / f"whitened_chunk_{i:04d}.npz"
            np.savez_compressed(
                output_path,
                activations=whitened,
                tokens=tokens
            )

def save_transform_params(whitening: PositionalWhitening, output_dir: Path):
    """Save whitening parameters for future use."""
    param_path = output_dir / "whitening_params.pt"
    torch.save({
        'mean': whitening.mean.cpu(),
        'weight': whitening.weight.cpu(),
        'reverse_weight': whitening.reverse_weight.cpu(),
        'seq_len': whitening.seq_len,
        'n_features': whitening.n_features,
        'epsilon': whitening.epsilon
    }, param_path)
    logger.info(f"Saved whitening parameters to {param_path}")

def load_transform_params(param_path: Path) -> PositionalWhitening:
    """Load saved whitening parameters."""
    params = torch.load(param_path)
    whitening = PositionalWhitening(
        seq_len=params['seq_len'],
        n_features=params['n_features'],
        epsilon=params['epsilon']
    )
    whitening.mean = params['mean'].cuda()
    whitening.weight = params['weight'].cuda()
    whitening.reverse_weight = params['reverse_weight'].cuda()
    return whitening

def validate_whitening(whitening: PositionalWhitening, sample_path: Path):
    """Validate whitening transform with basic statistical checks."""
    logger.info("Validating whitening transform...")
    
    # Load a sample chunk
    chunk = torch.from_numpy(np.load(sample_path)['activations']).cuda()
    whitened = whitening(chunk)
    
    # Check mean and covariance for each position
    for pos in range(whitening.seq_len):
        pos_data = whitened[:, pos, :]
        
        # Check mean (should be close to 0)
        pos_mean = pos_data.mean(dim=0)
        mean_norm = torch.norm(pos_mean).item()
        logger.info(f"Position {pos} mean norm: {mean_norm:.6f}")
        
        # Check covariance (should be close to identity)
        pos_cov = pos_data.T @ pos_data / (len(pos_data) - 1)
        eye_diff_norm = torch.norm(pos_cov - torch.eye(pos_cov.shape[0], device=pos_cov.device)).item()
        logger.info(f"Position {pos} covariance deviation from identity: {eye_diff_norm:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Whiten neural network activations")
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory containing activation chunks")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for whitened chunks")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length of the activations")
    parser.add_argument("--n_features", type=int, default=2304, help="Number of features in activations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Numerical stability constant")
    parser.add_argument("--load_params", type=Path, help="Load existing whitening parameters")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input chunks
    input_paths = get_chunk_paths(args.input_dir)
    logger.info(f"Found {len(input_paths)} input chunks")
    
    if args.load_params:
        # Load existing transform
        whitening = load_transform_params(args.load_params)
        logger.info(f"Loaded whitening parameters from {args.load_params}")
    else:
        # Fit new transform
        whitening = PositionalWhitening(args.seq_len, args.n_features, args.epsilon)
        whitening.fit(input_paths)
        logger.info("Fitted new whitening transform")
        
        # Save parameters
        save_transform_params(whitening, args.output_dir)
    
    # Validate transform
    validate_whitening(whitening, input_paths[0])
    
    # Apply transform to all chunks
    save_whitened_chunks(whitening, input_paths, args.output_dir, args.batch_size)
    logger.info("Completed whitening all chunks")

if __name__ == "__main__":
    main()