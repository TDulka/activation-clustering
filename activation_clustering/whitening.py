from pathlib import Path
from typing import Optional
import torch
from torch import nn
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ActivationWhitening(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

        # Parameters as regular instance variables
        self.weight = None  # [n_features, n_features]
        self.mean = None  # [n_features]
        self.reverse_weight = None  # [n_features, n_features]

    def fit(self, activations: torch.Tensor):
        """
        Fit whitening transform to activations.

        Args:
            activations: Tensor of shape [batch_size, seq_len, n_features]
                        Will be reshaped to [batch_size * seq_len, n_features]
        """
        # Reshape to [batch_size * seq_len, n_features]
        flat_activations = activations.reshape(-1, activations.shape[-1])

        # Compute mean
        self.mean = flat_activations.mean(dim=0)

        # Compute covariance
        centered = flat_activations - self.mean.unsqueeze(0)
        cov = (centered.T @ centered) / (len(centered) - 1)

        # Compute transforms using eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Forward transform (whitening)
        self.weight = (
            eigenvectors
            @ torch.diag(1.0 / torch.sqrt(eigenvalues + self.epsilon))
            @ eigenvectors.T
        )

        # Reverse transform (unwhitening)
        self.reverse_weight = (
            eigenvectors
            @ torch.diag(torch.sqrt(eigenvalues + self.epsilon))
            @ eigenvectors.T
        )

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply whitening transform.

        Args:
            x: Tensor of shape [batch_size, seq_len, n_features]
        Returns:
            Whitened tensor of same shape
        """
        original_shape = x.shape
        # Reshape to [batch_size * seq_len, n_features]
        x_flat = x.reshape(-1, original_shape[-1])
        # Apply transform
        whitened_flat = (x_flat - self.mean) @ self.weight
        # Restore original shape
        return whitened_flat.reshape(original_shape)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply reverse whitening transform.

        Args:
            x: Tensor of shape [batch_size, seq_len, n_features]
        Returns:
            Unwhitened tensor of same shape
        """
        original_shape = x.shape
        # Reshape to [batch_size * seq_len, n_features]
        x_flat = x.reshape(-1, original_shape[-1])
        # Apply reverse transform
        unwhitened_flat = (x_flat @ self.reverse_weight) + self.mean
        # Restore original shape
        return unwhitened_flat.reshape(original_shape)


def validate_whitening(
    whitening: ActivationWhitening, sample_activations: torch.Tensor
):
    """Validate whitening transform with basic statistical checks."""
    logger.info("Validating whitening transform...")

    whitened = whitening(sample_activations)
    flat_whitened = whitened.reshape(-1, whitened.shape[-1])

    # Check mean (should be close to 0)
    mean = flat_whitened.mean(dim=0)
    mean_norm = torch.norm(mean).item()
    logger.info(f"Mean norm: {mean_norm:.6f}")

    # Check covariance (should be close to identity)
    cov = flat_whitened.T @ flat_whitened / (len(flat_whitened) - 1)
    eye_diff_norm = torch.norm(cov - torch.eye(cov.shape[0], device=cov.device)).item()
    logger.info(f"Covariance deviation from identity: {eye_diff_norm:.6f}")
