import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class SequentialSoftClustering:
    def __init__(
        self,
        n_clusters: int,
        d_model: int,
        temperature: float = 1.0,
        min_membership: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize clustering for sequential, whitened activations.

        Args:
            n_clusters: Number of clusters to learn
            d_model: Dimension of the activation vectors
            temperature: Temperature for softmax (lower = harder assignments)
            min_membership: Minimum membership threshold for sparse assignments
            device: Device to use for computations
        """
        self.n_clusters = n_clusters
        self.d_model = d_model
        self.temperature = temperature
        self.min_membership = min_membership
        self.device = device

        # Initialize cluster centers on unit hypersphere
        # Since activations are whitened, initializing on unit sphere is appropriate
        self.centers = torch.randn(n_clusters, d_model, device=device)
        self.centers = F.normalize(self.centers, dim=1)

    def fit(
        self,
        activations: torch.Tensor,  # Shape: [batch, seq_len, d_model]
        n_iterations: int = 100,
        max_points_per_iter: int = 100000,
        verbose: bool = True,
    ) -> None:
        """Fit clustering model to sequential activations"""
        batch_size, seq_len, _ = activations.shape
        total_points = batch_size * seq_len

        for iteration in range(n_iterations):
            # Randomly sample positions if we have too many points
            if total_points > max_points_per_iter:
                # Flatten batch and sequence dimensions for sampling
                flat_activations = activations.reshape(-1, self.d_model)
                indices = torch.randperm(total_points, device=self.device)[
                    :max_points_per_iter
                ]
                batch_activations = flat_activations[indices]
            else:
                batch_activations = activations.reshape(-1, self.d_model)

            # Compute memberships
            memberships = self._compute_memberships(batch_activations)

            # Update centers
            self._update_centers(batch_activations, memberships)

            if verbose and (iteration + 1) % 10 == 0:
                with torch.no_grad():
                    # Compute average maximum membership as a measure of clustering quality
                    avg_max_membership = torch.mean(torch.max(memberships, dim=1)[0])
                    print(
                        f"Iteration {iteration + 1}/{n_iterations}, "
                        f"Avg Max Membership: {avg_max_membership:.4f}"
                    )

    def transform(
        self,
        activations: torch.Tensor,  # Shape: [batch, seq_len, d_model]
        return_sparse: bool = True,
        normalize_positions: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform activations to cluster memberships.

        Args:
            activations: Input activations
            return_sparse: Whether to return sparse memberships
            normalize_positions: Whether to normalize by position

        Returns:
            Dictionary containing:
                - memberships: Shape [batch, seq_len, n_clusters] or sparse version
                - indices: Indices of non-zero memberships if sparse
        """
        batch_size, seq_len, _ = activations.shape

        # Reshape to 2D for membership computation
        flat_activations = activations.reshape(-1, self.d_model)

        # Compute memberships
        memberships = self._compute_memberships(flat_activations)

        # Reshape back to sequential form
        memberships = memberships.reshape(batch_size, seq_len, self.n_clusters)

        if normalize_positions:
            # Normalize memberships position-wise
            memberships = memberships / (memberships.sum(dim=-1, keepdim=True) + 1e-10)

        if return_sparse:
            # Create sparse representation
            mask = memberships > self.min_membership
            sparse_memberships = memberships * mask

            # Get indices of non-zero memberships
            indices = torch.nonzero(mask)  # Shape: [n_nonzero, 3]

            return {"memberships": sparse_memberships, "indices": indices}

        return {"memberships": memberships}

    def _compute_memberships(self, X: torch.Tensor) -> torch.Tensor:
        """Compute soft cluster memberships using dot product similarity"""
        # Since inputs are whitened, dot product is appropriate
        similarities = torch.mm(X, self.centers.T)  # Shape: [n_points, n_clusters]

        # Apply temperature scaling and softmax
        memberships = torch.softmax(similarities / self.temperature, dim=1)
        return memberships

    def _update_centers(self, X: torch.Tensor, memberships: torch.Tensor) -> None:
        """Update cluster centers using weighted averages"""
        # Compute weighted sum of points
        weighted_sum = torch.mm(memberships.T, X)  # Shape: [n_clusters, d_model]

        # Normalize to unit sphere
        self.centers = F.normalize(weighted_sum, dim=1)


def analyze_sequential_clusters(
    model: SequentialSoftClustering,
    activations: torch.Tensor,
    tokens: Optional[torch.Tensor] = None,
    n_examples: int = 5,
) -> Dict:
    """
    Analyze cluster characteristics for sequential activations.

    Args:
        model: Trained clustering model
        activations: Input activations [batch, seq_len, d_model]
        tokens: Optional token ids [batch, seq_len]
        n_examples: Number of examples to store per cluster

    Returns:
        Dictionary containing analysis results
    """
    results = model.transform(activations)
    memberships = results["memberships"]

    # Compute cluster statistics
    cluster_stats = {
        "total_usage": memberships.sum(dim=(0, 1)).cpu().numpy(),
        "avg_assignments_per_pos": (memberships > model.min_membership)
        .sum(-1)
        .float()
        .mean()
        .item(),
        "position_usage": memberships.mean(dim=0)
        .cpu()
        .numpy(),  # Average usage by position
        "examples": {},
    }

    # Find exemplar activations for each cluster
    batch_size, seq_len, _ = activations.shape
    flat_memberships = memberships.reshape(-1, model.n_clusters)
    flat_positions = torch.arange(seq_len, device=activations.device).repeat(batch_size)

    for k in range(model.n_clusters):
        # Get top examples for cluster k
        cluster_scores = flat_memberships[:, k]
        top_indices = torch.topk(cluster_scores, n_examples)[1]

        examples = {
            "scores": cluster_scores[top_indices].cpu().numpy(),
            "positions": flat_positions[top_indices].cpu().numpy(),
        }

        if tokens is not None:
            flat_tokens = tokens.reshape(-1)
            examples["tokens"] = flat_tokens[top_indices.cpu()].cpu().numpy()

        cluster_stats["examples"][k] = examples

    return cluster_stats
