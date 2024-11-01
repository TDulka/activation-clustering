from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch import nn
import numpy as np
import logging
from dataclasses import dataclass
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    n_clusters: int = 16384
    batch_size: int = 32
    max_iter: int = 100
    random_state: int = 42


class ActivationKMeans(nn.Module):
    def __init__(self, config: ClusteringConfig):
        super().__init__()
        self.config = config

        logger.info(f"Initializing KMeans with {config.n_clusters} clusters")
        self.kmeans = MiniBatchKMeans(
            n_clusters=config.n_clusters,
            batch_size=config.batch_size,
            max_iter=config.max_iter,
            random_state=config.random_state,
            verbose=1,  # Add verbosity to see iterations
        )

    def fit(self, activations: torch.Tensor):
        """Fit k-means on flattened activations using all data

        Args:
            activations: Tensor of any shape [..., n_features]
        """
        # Flatten all dimensions except the last
        flat_activations = activations.reshape(-1, activations.shape[-1])

        logger.info(f"Starting KMeans fit on tensor of shape {flat_activations.shape}")
        data = flat_activations.cpu().numpy()

        # Use a much larger batch size (about 5x n_clusters)
        batch_size = max(100000, self.config.n_clusters * 5)  # ~82k samples minimum
        n_batches = (len(data) + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc="Fitting KMeans"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            batch = data[start_idx:end_idx]
            self.kmeans.partial_fit(batch)

        logger.info("KMeans fitting completed")
        return self

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Get cluster assignments for activations"""
        original_shape = activations.shape[:-1]
        flat_activations = activations.reshape(-1, activations.shape[-1])

        # Process predictions in batches
        batch_size = 10000
        n_batches = (len(flat_activations) + batch_size - 1) // batch_size
        all_labels = []

        logger.info(f"Predicting clusters for tensor of shape {flat_activations.shape}")
        for i in tqdm(range(n_batches), desc="Predicting clusters"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(flat_activations))
            batch = flat_activations[start_idx:end_idx].cpu().numpy()
            labels = self.kmeans.predict(batch)
            all_labels.append(labels)

        labels = np.concatenate(all_labels)
        return torch.from_numpy(labels).reshape(original_shape)

    def save_checkpoint(self, path: Path):
        """Save clustering results"""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "centroids": torch.from_numpy(self.kmeans.cluster_centers_),
            },
            path / "clustering_results.pt",
        )

    @classmethod
    def load_checkpoint(cls, path: Path) -> "ActivationKMeans":
        """Load clustering results"""
        state = torch.load(path / "clustering_results.pt")
        instance = cls(state["config"])
        instance.kmeans.cluster_centers_ = state["centroids"].numpy()
        return instance

    def analyze_clusters(
        self,
        activations: torch.Tensor,
        token_ids: torch.Tensor,  # [batch_size, seq_len]
        max_examples_per_cluster: int = 5,
        context_window: int = 10,
    ) -> Dict[int, List[Tuple[int, List[int], int]]]:
        """Analyze clusters by showing tokens in their original context"""
        # Get cluster assignments
        labels = self.predict(activations)  # [batch_size, seq_len]

        # Create a dictionary to store examples for each cluster
        cluster_examples = defaultdict(list)

        # Process each sequence
        for batch_idx in range(activations.shape[0]):
            seq_token_ids = token_ids[batch_idx]  # [seq_len]

            for pos in range(activations.shape[1]):
                cluster_id = labels[batch_idx, pos].item()

                if len(cluster_examples[cluster_id]) >= max_examples_per_cluster:
                    continue

                # Get context window
                start_idx = max(0, pos - context_window)
                end_idx = min(len(seq_token_ids), pos + context_window + 1)

                # Get the specific token ID
                token_id = seq_token_ids[pos].item()

                # Get the context token IDs
                context_ids = seq_token_ids[start_idx:end_idx].tolist()

                # Store token position relative to start of context window
                token_pos = pos - start_idx

                cluster_examples[cluster_id].append((token_id, context_ids, token_pos))

        return dict(cluster_examples)


class ClusterVisualizer:
    """Utility class for visualizing cluster contents"""

    def __init__(
        self, cluster_examples: Dict[int, List[Tuple[int, List[int], int]]], tokenizer
    ):
        self.cluster_examples = cluster_examples
        self.tokenizer = tokenizer

    def print_cluster(self, cluster_id: int):
        """Print examples from a specific cluster"""
        if cluster_id not in self.cluster_examples:
            print(f"No examples found for cluster {cluster_id}")
            return

        print(f"\nCluster {cluster_id}:")
        print("-" * 80)

        for token_id, context_ids, token_pos in self.cluster_examples[cluster_id]:
            # Decode token
            token = self.tokenizer.decode([token_id])

            # Decode parts before and after the token separately
            prefix = self.tokenizer.decode(context_ids[:token_pos])
            suffix = self.tokenizer.decode(context_ids[token_pos + 1 :])

            print(f"{prefix}[[[{token}]]]{suffix}")
            print("-" * 40)

    def print_random_clusters(self, n: int = 5):
        """Print examples from n random clusters"""
        cluster_ids = list(self.cluster_examples.keys())
        selected_ids = np.random.choice(
            cluster_ids, min(n, len(cluster_ids)), replace=False
        )

        for cluster_id in selected_ids:
            self.print_cluster(cluster_id)

    def find_clusters_with_token(self, token: str) -> List[int]:
        """Find clusters that contain a specific token"""
        matching_clusters = []

        for cluster_id, examples in self.cluster_examples.items():
            if any(token == t for t, _, _ in examples):
                matching_clusters.append(cluster_id)

        return matching_clusters
