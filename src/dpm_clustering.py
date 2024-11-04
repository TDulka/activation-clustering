from typing import Dict, List, Tuple, Optional
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
from scipy.special import digamma, gammaln
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DPMConfig:
    alpha: float = 1.0  # Concentration parameter
    max_clusters: int = 256  # Upper bound for practical implementation
    batch_size: int = 32
    n_iterations: int = 100
    random_state: int = 42
    threshold: float = 1e-4  # Convergence threshold


class StreamingDPM(nn.Module):
    """Streaming Dirichlet Process Mixture model optimized for transformer activations"""

    def __init__(self, config: DPMConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.random_state)

        # Initialize cluster statistics as torch tensors on GPU
        self.cluster_means = []  # Will be converted to GPU tensor when first cluster is created
        self.cluster_covs = []  # Will be converted to GPU tensor when first cluster is created
        self.cluster_counts = []  # Will be converted to GPU tensor when first cluster is created
        self.n_points = 0

    def _compute_log_likelihood_batch(
        self, x: torch.Tensor, means: torch.Tensor, covs: torch.Tensor
    ) -> torch.Tensor:
        """Compute log likelihood for a batch of points across all clusters"""
        batch_size, d = x.shape
        n_clusters = means.shape[0]

        # Reshape for broadcasting
        x = x.unsqueeze(1)  # [batch_size, 1, d]
        means = means.unsqueeze(0)  # [1, n_clusters, d]

        # Compute deviations
        dev = x - means  # [batch_size, n_clusters, d]

        # Batch matrix solve
        solved = torch.linalg.solve(
            covs.unsqueeze(0), dev.transpose(-1, -2)
        )  # [batch_size, n_clusters, d]

        # Compute log determinant
        log_dets = torch.logdet(
            covs + 1e-10 * torch.eye(d, device=self.device).unsqueeze(0)
        )

        # Compute log probabilities
        log_probs = -0.5 * (
            d * np.log(2 * np.pi)
            + log_dets.unsqueeze(0)
            + torch.sum(dev * solved.transpose(-1, -2), dim=-1)
        )

        return log_probs

    def _create_new_cluster(self, x: torch.Tensor):
        """Initialize new cluster with single point"""
        if not self.cluster_means:  # First cluster
            self.cluster_means = [x.clone()]
            self.cluster_covs = [0.1 * torch.eye(x.shape[0], device=self.device)]
            self.cluster_counts = [torch.ones(1, device=self.device)]
        else:
            self.cluster_means.append(x.clone())
            self.cluster_covs.append(0.1 * torch.eye(x.shape[0], device=self.device))
            self.cluster_counts.append(torch.ones(1, device=self.device))

    def _update_cluster(self, cluster_idx: int, x: torch.Tensor):
        """Update cluster statistics with new point"""
        count = self.cluster_counts[cluster_idx]
        old_mean = self.cluster_means[cluster_idx]
        old_cov = self.cluster_covs[cluster_idx]

        # Update mean
        new_mean = (count * old_mean + x) / (count + 1)

        # Update covariance using online update formula
        dev = x - new_mean
        new_cov = (
            count * old_cov
            + count * torch.outer(old_mean - new_mean, old_mean - new_mean)
            + torch.outer(dev, dev)
        ) / (count + 1)

        self.cluster_means[cluster_idx] = new_mean
        self.cluster_covs[cluster_idx] = new_cov
        self.cluster_counts[cluster_idx] += 1

    def _compute_cluster_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Compute probability of point belonging to each cluster"""
        n_clusters = len(self.cluster_means)
        if n_clusters == 0:
            return torch.tensor([])

        # Ensure x is 2D [1, hidden_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        log_probs = torch.zeros(n_clusters, device=self.device)
        for i in range(n_clusters):
            mean = self.cluster_means[i].unsqueeze(0)  # [1, hidden_dim]
            cov = self.cluster_covs[i].unsqueeze(0)  # [1, hidden_dim, hidden_dim]

            log_probs[i] = (
                torch.log(self.cluster_counts[i] / (self.n_points + self.config.alpha))
                + self._compute_log_likelihood_batch(x, mean, cov).squeeze()
            )

        # Add probability of new cluster
        log_new_cluster = torch.log(
            self.config.alpha / (self.n_points + self.config.alpha)
        )
        log_probs = torch.cat([log_probs, log_new_cluster.unsqueeze(0)])

        # Normalize probabilities
        log_probs -= log_probs.max()  # For numerical stability
        probs = torch.exp(log_probs)
        probs /= probs.sum()

        return probs

    def partial_fit(self, activations: torch.Tensor, batch_size: int = 128):
        """Process activations in batches with memory-efficient GPU acceleration"""
        # Move data to GPU in chunks
        for chunk_idx in tqdm(
            range(0, len(activations), 10000), desc="Processing chunks"
        ):
            chunk = activations[chunk_idx : chunk_idx + 10000].to(self.device)

            # Handle first point if no clusters exist
            if len(self.cluster_means) == 0:
                self._create_new_cluster(chunk[0])
                self.n_points += 1
                chunk = chunk[1:]

            # Convert lists to tensors for batch processing
            means = torch.stack(self.cluster_means)
            covs = torch.stack(self.cluster_covs)

            # Process in smaller batches
            for idx in range(0, len(chunk), batch_size):
                batch = chunk[idx : idx + batch_size]

                # Free unused memory
                torch.cuda.empty_cache()

                # Compute probabilities batch-wise
                log_probs = []
                for cluster_idx in range(len(means)):
                    cluster_mean = means[cluster_idx]
                    cluster_cov = covs[cluster_idx]

                    # Compute log likelihood for this cluster
                    dev = batch - cluster_mean.unsqueeze(0)
                    solved = torch.linalg.solve(
                        cluster_cov.unsqueeze(0), dev.unsqueeze(-1)
                    ).squeeze(-1)

                    log_det = torch.logdet(
                        cluster_cov
                        + 1e-10 * torch.eye(cluster_cov.shape[0], device=self.device)
                    )

                    cluster_log_prob = -0.5 * (
                        batch.shape[1] * np.log(2 * np.pi)
                        + log_det
                        + torch.sum(dev * solved, dim=1)
                    )

                    log_probs.append(cluster_log_prob)

                # Stack log probabilities
                log_probs = torch.stack(log_probs, dim=1)

                # Add probability of new cluster
                log_new_cluster = torch.log(
                    torch.tensor(
                        self.config.alpha / (self.n_points + self.config.alpha),
                        device=self.device,
                    )
                ).expand(len(batch))

                # Combine and normalize
                log_probs = torch.cat([log_probs, log_new_cluster.unsqueeze(1)], dim=1)

                # Normalize probabilities
                log_probs = log_probs - log_probs.max(dim=1, keepdim=True)[0]
                probs = torch.exp(log_probs)
                probs = probs / probs.sum(dim=1, keepdim=True)

                # Sample cluster assignments
                assignments = torch.multinomial(probs, 1).squeeze()

                # Update clusters
                for i, (x, assignment) in enumerate(zip(batch, assignments)):
                    if (
                        assignment == len(self.cluster_means)
                        and len(self.cluster_means) < self.config.max_clusters
                    ):
                        self._create_new_cluster(x)
                    else:
                        if assignment >= len(self.cluster_means):
                            assignment = torch.randint(
                                0, len(self.cluster_means), (1,), device=self.device
                            )
                        self._update_cluster(assignment.item(), x)

                    self.n_points += 1

                # Free memory again
                torch.cuda.empty_cache()

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Assign activations to clusters"""
        batch_activations = activations.cpu().numpy()
        predictions = np.zeros(len(batch_activations), dtype=np.int64)

        for i, x in enumerate(batch_activations):
            probs = self._compute_cluster_probabilities(x)[:-1]  # Exclude new cluster
            predictions[i] = np.argmax(probs)

        return torch.from_numpy(predictions)

    def analyze_clusters(
        self,
        activations: torch.Tensor,
        token_ids: torch.Tensor,
        tokenizer,
        max_examples_per_cluster: int = 5,
        context_window: int = 10,
        probability_threshold: float = 0.1,  # Minimum probability to consider
    ) -> Dict[int, List[Tuple[int, List[int], int, float]]]:  # Added probability
        """Analyze clusters by showing tokens in their original context with probabilities

        Args:
            activations: [batch_size, seq_len, hidden_dim] activation tensor
            token_ids: [batch_size, seq_len] token ID tensor
            tokenizer: tokenizer for decoding tokens
            max_examples_per_cluster: maximum examples to store per cluster
            context_window: number of tokens before/after for context
            probability_threshold: minimum probability to consider cluster assignment

        Returns:
            Dictionary mapping cluster IDs to lists of (token_id, context_ids, position, probability)
        """
        # Reshape if needed
        if len(activations.shape) == 3:
            batch_size, seq_len, hidden_dim = activations.shape
            flat_activations = activations.reshape(-1, hidden_dim)
            flat_token_ids = token_ids.reshape(-1)
        else:
            flat_activations = activations
            flat_token_ids = token_ids

        # Store examples with their probabilities
        cluster_examples = defaultdict(list)

        # Process each token
        for idx in tqdm(range(len(flat_activations)), desc="Analyzing clusters"):
            # Move single activation to GPU and ensure it's the right shape
            x = flat_activations[idx].to(self.device)

            # Get probabilities for all clusters
            probs = self._compute_cluster_probabilities(x)[
                :-1
            ]  # Exclude new cluster prob

            # Get sequence position info
            if len(activations.shape) == 3:
                batch_idx = idx // seq_len
                pos = idx % seq_len
                seq_token_ids = token_ids[batch_idx]
            else:
                pos = idx
                seq_token_ids = token_ids

            # Get context window
            start_idx = max(0, pos - context_window)
            end_idx = min(len(seq_token_ids), pos + context_window + 1)

            # Get token and context
            token_id = flat_token_ids[idx].item()
            context_ids = seq_token_ids[start_idx:end_idx].tolist()
            token_pos = pos - start_idx

            # Store examples for each cluster with sufficient probability
            for cluster_id, prob in enumerate(probs):
                if prob >= probability_threshold:
                    if len(cluster_examples[cluster_id]) < max_examples_per_cluster:
                        cluster_examples[cluster_id].append(
                            (token_id, context_ids, token_pos, float(prob))
                        )

        return dict(cluster_examples)

    def print_cluster(self, cluster_id: int, tokenizer):
        """Print examples from a specific cluster with their probabilities"""
        if cluster_id not in self.cluster_examples:
            print(f"No examples found for cluster {cluster_id}")
            return

        print(f"\nCluster {cluster_id}:")
        print("-" * 80)

        for token_id, context_ids, token_pos, prob in self.cluster_examples[cluster_id]:
            # Decode token and context
            token = tokenizer.decode([token_id])
            prefix = tokenizer.decode(context_ids[:token_pos])
            suffix = tokenizer.decode(context_ids[token_pos + 1 :])

            print(f"P({prob:.3f}): {prefix}[[[{token}]]]{suffix}")
            print("-" * 40)
