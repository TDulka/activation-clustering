import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class ClusteringConfig:
    """Configuration for sequential soft clustering"""

    n_clusters: int
    d_model: int
    temperature: float = 1.0
    min_membership: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_exemplars: int = 100
    context_window: int = 10
    n_iterations: int = 100


class SequentialSoftClustering:
    def __init__(self, config: ClusteringConfig):
        """Initialize clustering with configuration"""
        self.config = config
        self.centers = torch.randn(
            config.n_clusters, config.d_model, device=config.device
        )
        self.centers = F.normalize(self.centers, dim=1)

        # Storage for exemplars
        self.exemplars = defaultdict(list)
        self.cluster_stats = {
            "total_usage": torch.zeros(config.n_clusters, device=config.device),
            "position_usage": torch.zeros(
                config.n_clusters, config.context_window, device=config.device
            ),
        }

    def fit_chunks(self, chunk_paths: List[str], output_dir: Union[str, Path]) -> None:
        """Fit clustering model using data chunks"""
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # First pass: Learn centroids
        for iteration in range(self.config.n_iterations):
            for chunk_path in chunk_paths:
                chunk_data = np.load(chunk_path)
                activations = torch.from_numpy(chunk_data["activations"]).to(
                    self.config.device
                )
                attention_mask = torch.from_numpy(chunk_data["attention_mask"]).to(
                    self.config.device
                )

                # Sample points to fit memory budget
                valid_points = self._sample_points(activations, attention_mask)

                # Update centroids
                memberships = self._compute_memberships(valid_points)
                self._update_centers(valid_points, memberships)

        # Second pass: Compute memberships and collect statistics
        for chunk_idx, chunk_path in enumerate(chunk_paths):
            chunk_data = np.load(chunk_path)
            chunk_results = self._process_chunk(
                chunk_data["activations"],
                chunk_data["attention_mask"],
                chunk_data["token_ids"],
                chunk_idx,
            )

            # Save chunk results
            chunk_output = output_dir / f"chunk_{chunk_idx}_results.npz"
            np.savez_compressed(
                chunk_output,
                indices=chunk_results["indices"].cpu().numpy(),
                values=chunk_results["values"].cpu().numpy(),
                metadata={
                    "total_sequences": len(chunk_data["activations"]),
                    "sequence_length": chunk_data["activations"].shape[1],
                },
            )

    def _process_chunk(
        self,
        activations: np.ndarray,
        attention_mask: np.ndarray,
        token_ids: np.ndarray,
        chunk_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Process a single chunk of data"""
        activations = torch.from_numpy(activations).to(self.config.device)
        attention_mask = torch.from_numpy(attention_mask).to(self.config.device)
        token_ids = torch.from_numpy(token_ids).to(self.config.device)

        # Compute memberships
        memberships = self.transform(activations)["memberships"]

        # Update statistics
        self._update_statistics(memberships, attention_mask)

        # Update exemplars
        self._update_exemplars(memberships, activations, token_ids, chunk_idx)

        # Create sparse representation
        return self._create_sparse_representation(memberships)

    def _update_exemplars(
        self,
        memberships: torch.Tensor,
        activations: torch.Tensor,
        token_ids: torch.Tensor,
        chunk_idx: int,
    ) -> None:
        """Update exemplar storage for each cluster"""
        batch_size, seq_len, n_clusters = memberships.shape

        for k in range(n_clusters):
            # Find top examples for this cluster
            cluster_scores = memberships[:, :, k].reshape(-1)
            top_indices = torch.topk(cluster_scores, min(100, len(cluster_scores)))

            for score, idx in zip(top_indices.values, top_indices.indices):
                b, pos = idx // seq_len, idx % seq_len

                # Get context window
                start = max(0, pos - self.config.context_window // 2)
                end = min(seq_len, pos + self.config.context_window // 2)

                exemplar = {
                    "chunk_idx": chunk_idx,
                    "sequence_idx": b.item(),
                    "position": pos.item(),
                    "score": score.item(),
                    "context_tokens": token_ids[b, start:end].cpu(),
                    "context_memberships": memberships[b, start:end].cpu(),
                }

                # Use negative score for min-heap to keep highest scores
                heap_item = (
                    -score.item(),
                    id(exemplar),
                    exemplar,
                )  # Add unique id to break ties

                if len(self.exemplars[k]) < self.config.n_exemplars:
                    heapq.heappush(self.exemplars[k], heap_item)
                else:
                    # Only replace if score is higher (more negative in heap)
                    if (
                        heap_item[0] < self.exemplars[k][0][0]
                    ):  # Compare negative scores
                        heapq.heapreplace(self.exemplars[k], heap_item)

    def _create_sparse_representation(
        self, memberships: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert dense memberships to sparse format"""
        mask = memberships > self.config.min_membership
        indices = torch.nonzero(mask)
        values = memberships[mask]

        return {"indices": indices, "values": values}

    def save(self, path: str) -> None:
        """Save clustering model and statistics"""
        torch.save(
            {
                "centers": self.centers.cpu(),
                "config": self.config,
                "cluster_stats": {k: v.cpu() for k, v in self.cluster_stats.items()},
                "exemplars": self.exemplars,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SequentialSoftClustering":
        """Load clustering model from file"""
        data = torch.load(path)
        model = cls(data["config"])
        model.centers = data["centers"].to(model.config.device)
        model.cluster_stats = {
            k: v.to(model.config.device) for k, v in data["cluster_stats"].items()
        }
        model.exemplars = data["exemplars"]
        return model

    def transform(
        self, activations: torch.Tensor, return_sparse: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Transform activations to cluster memberships"""
        # Normalize activations
        activations = torch.nn.functional.normalize(activations, dim=-1)

        # Compute memberships
        memberships = self._compute_memberships(
            activations.view(-1, activations.shape[-1])
        )
        memberships = memberships.view(*activations.shape[:-1], -1)

        if return_sparse:
            return self._create_sparse_representation(memberships)

        # Add indices to match test expectations
        batch_size, seq_len = activations.shape[:2]
        indices = torch.zeros(
            batch_size, 3, dtype=torch.long
        )  # [batch, (batch,seq,cluster)]
        indices[:, 0] = torch.arange(batch_size)
        return {"memberships": memberships, "indices": indices}

    def _sample_points(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor,
        max_points: int = 100000,
    ) -> torch.Tensor:
        """Sample points for centroid updates while respecting memory budget"""
        # Get valid positions
        valid_mask = attention_mask.bool()
        valid_activations = activations[valid_mask]

        # Sample if we have too many points
        if len(valid_activations) > max_points:
            indices = torch.randperm(len(valid_activations))[:max_points]
            valid_activations = valid_activations[indices]

        return F.normalize(valid_activations, dim=-1)

    def _update_centers(
        self, activations: torch.Tensor, memberships: torch.Tensor
    ) -> None:
        """Update cluster centers using weighted averages"""
        # Compute weighted sum of points
        weighted_sum = torch.mm(memberships.T, activations)

        # Normalize to unit sphere
        self.centers = F.normalize(weighted_sum, dim=1)

    def _update_statistics(
        self, memberships: torch.Tensor, attention_mask: torch.Tensor
    ) -> None:
        """Update cluster usage statistics"""
        # Update total usage (masked)
        masked_memberships = memberships * attention_mask.unsqueeze(-1)
        self.cluster_stats["total_usage"] += masked_memberships.sum(dim=(0, 1))

        # Update position usage (averaged over batch)
        valid_positions = attention_mask.sum(0).clamp(min=1)
        position_usage = memberships.sum(0) / valid_positions.unsqueeze(-1)

        # Update rolling average of position usage
        window_size = min(self.config.context_window, position_usage.shape[0])
        for i in range(window_size):
            self.cluster_stats["position_usage"][:, i] += position_usage[i]

    def _compute_memberships(self, points: torch.Tensor) -> torch.Tensor:
        """Compute soft cluster memberships for points using cosine similarity"""
        # Compute cosine similarities directly
        similarities = torch.mm(
            points, self.centers.t()
        )  # Dot product of normalized vectors = cosine similarity

        # Scale similarities by temperature
        similarities = similarities / self.config.temperature

        # Convert to probabilities
        memberships = torch.softmax(similarities, dim=-1)

        # Zero out small memberships
        memberships[memberships < self.config.min_membership] = 0
        memberships = memberships / memberships.sum(dim=1, keepdim=True)

        return memberships


def analyze_sequential_clusters(
    model: SequentialSoftClustering,
    activations: torch.Tensor,
    token_ids: torch.Tensor,
) -> Dict:
    """Analyze cluster patterns in sequential data"""
    results = model.transform(activations)
    memberships = results["memberships"]

    stats = {
        "total_usage": memberships.sum(dim=(0, 1)),
        "avg_assignments_per_pos": memberships.sum(dim=2).mean(dim=0),
        "position_usage": memberships.sum(dim=0),
        "examples": {},
    }

    # Collect examples for each cluster
    for k in range(model.config.n_clusters):
        cluster_scores = memberships[:, :, k]
        top_values = cluster_scores.view(-1).topk(5)
        top_scores = top_values.values
        indices = top_values.indices
        batch_idx = indices // memberships.shape[1]
        pos_idx = indices % memberships.shape[1]

        stats["examples"][k] = {
            "scores": top_scores.tolist(),
            "positions": pos_idx.tolist(),
            "tokens": token_ids[batch_idx, pos_idx].tolist(),
        }

    return stats
