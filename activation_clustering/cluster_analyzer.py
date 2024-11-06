from typing import Dict, List, Optional, Tuple, DefaultDict, Union
import torch
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots

from activation_clustering.soft_clustering import SequentialSoftClustering


@dataclass
class AnalysisConfig:
    """Configuration for cluster analysis"""

    min_membership: float = 0.01
    max_examples: int = 100
    context_window: int = 10
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ClusterExample:
    """Stores a single example from a cluster"""

    token_id: int
    context_ids: List[int]
    token_pos: int
    activation: np.ndarray
    membership_score: float


class ClusterAnalyzer:
    """Analyzes clustering results and provides visualization tools"""

    def __init__(
        self,
        model_path: str,
        results_dir: str,
        tokenizer,
        config: Optional[AnalysisConfig] = None,
    ):
        self.model = SequentialSoftClustering.load(model_path)
        self.results_dir = Path(results_dir)
        self.config = config or AnalysisConfig()
        self.tokenizer = tokenizer

        # Initialize cluster stats from model
        self.cluster_stats = {
            "total_usage": self.model.cluster_stats["total_usage"].clone(),
            "position_usage": self.model.cluster_stats["position_usage"].clone(),
        }

        # Initialize cluster sizes
        self.cluster_sizes = torch.zeros(
            self.model.config.n_clusters, device="cpu", dtype=torch.float32
        )

        # Load metadata
        self.chunk_metadata = self._load_chunk_metadata()
        self.cluster_examples = {}

    def _load_chunk_metadata(self) -> Dict:
        """Load metadata from all chunk results"""
        metadata = {}
        for chunk_file in self.results_dir.glob("chunk_*_results.npz"):
            chunk_id = int(chunk_file.stem.split("_")[1])
            with np.load(chunk_file, allow_pickle=True) as data:
                metadata[chunk_id] = dict(data["metadata"].item())
        return metadata

    def get_cluster_summary(self, cluster_id: int) -> Dict:
        """Get summary statistics and examples for a specific cluster"""
        summary = {
            "total_usage": self.model.cluster_stats["total_usage"][cluster_id].item(),
            "position_usage": self.model.cluster_stats["position_usage"][cluster_id]
            .cpu()
            .numpy(),
            "exemplars": self._get_cluster_exemplars(cluster_id),
        }
        return summary

    def _get_cluster_exemplars(self, cluster_id: int) -> List[Dict]:
        """Get sorted exemplars for a cluster"""
        exemplars = self.model.exemplars[cluster_id]
        # Extract exemplar from tuple (negative_score, id, exemplar)
        return [
            ex[2] for ex in sorted(exemplars)
        ]  # No need for reverse=True since scores are negative

    def visualize_cluster_context(
        self, cluster_id: int, example_idx: int, tokenizer=None
    ) -> Dict:
        """Visualize context around a specific cluster example"""
        exemplar = self._get_cluster_exemplars(cluster_id)[example_idx]

        # Load chunk data
        chunk_path = self.results_dir / f"chunk_{exemplar['chunk_idx']}_results.npz"
        chunk_data = np.load(chunk_path)

        context = {
            "position": exemplar["position"],
            "score": exemplar["score"],
            "context_memberships": exemplar["context_memberships"].cpu().numpy(),
        }

        if tokenizer:
            context["tokens"] = tokenizer.convert_ids_to_tokens(
                exemplar["context_tokens"].cpu().numpy()
            )

        return context

    def find_cluster_patterns(self, min_support: float = 0.1) -> Dict:
        """Find common patterns in cluster activations"""
        patterns = {}

        # Analyze position-wise patterns
        position_patterns = self._analyze_position_patterns(min_support)

        # Analyze cross-cluster correlations
        cluster_correlations = self._compute_cluster_correlations()

        patterns.update(
            {
                "position_patterns": position_patterns,
                "cluster_correlations": cluster_correlations,
            }
        )

        return patterns

    def _analyze_position_patterns(self, min_support: float) -> Dict:
        """Analyze common positional patterns in cluster usage"""
        position_usage = self.model.cluster_stats["position_usage"]

        # Normalize usage
        usage_prob = position_usage / position_usage.sum(dim=1, keepdim=True)

        patterns = {}
        for k in range(self.model.config.n_clusters):
            # Find positions where cluster k is commonly active
            active_positions = torch.where(usage_prob[k] > min_support)[0]

            if len(active_positions) > 0:
                patterns[k] = {
                    "active_positions": active_positions.cpu().numpy(),
                    "position_scores": usage_prob[k][active_positions].cpu().numpy(),
                }

        return patterns

    def _compute_cluster_correlations(self) -> np.ndarray:
        """Compute correlation matrix between cluster activations"""
        n_clusters = self.model.config.n_clusters
        correlations = torch.zeros((n_clusters, n_clusters), device=self.config.device)

        # Process chunks to compute correlations
        for chunk_file in self.results_dir.glob("chunk_*_results.npz"):
            chunk_data = np.load(chunk_file)
            indices = torch.from_numpy(chunk_data["indices"]).to(self.config.device)
            values = torch.from_numpy(chunk_data["values"]).to(self.config.device)

            # Update correlation matrix
            self._update_correlations(correlations, indices, values)

        # Normalize correlations
        correlations = correlations / correlations.diagonal().view(-1, 1)
        return correlations.cpu().numpy()

    def _update_correlations(
        self, correlations: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Update correlation matrix with batch of data"""
        batch_size = len(indices)

        # Create sparse correlation update
        for i in range(batch_size):
            cluster_idx = indices[i, 2]
            correlations[cluster_idx, cluster_idx] += values[i] ** 2

            # Update cross-correlations
            for j in range(i + 1, batch_size):
                if indices[i, 0] == indices[j, 0] and indices[i, 1] == indices[j, 1]:
                    ci, cj = indices[i, 2], indices[j, 2]
                    update = values[i] * values[j]
                    correlations[ci, cj] += update
                    correlations[cj, ci] += update

    def process_examples(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,
        batch_size: int = 64,
        max_positions: Optional[int] = None,
    ) -> None:
        """Process data to extract cluster examples efficiently"""
        total_batch, seq_len, _ = activations.shape

        # Initialize cluster sizes on CPU
        self.cluster_sizes = torch.zeros(
            self.model.config.n_clusters, device="cpu", dtype=torch.float32
        )

        # Initialize examples storage
        top_examples = defaultdict(list)

        # Process in batches
        for batch_start in tqdm(range(0, total_batch, batch_size)):
            batch_end = min(batch_start + batch_size, total_batch)
            batch_activations = activations[batch_start:batch_end]
            batch_tokens = tokens[batch_start:batch_end]

            # Process sequence chunks if needed
            chunk_size = max_positions or seq_len
            for seq_start in range(0, seq_len, chunk_size):
                seq_end = min(seq_start + chunk_size, seq_len)

                # Get chunk data
                chunk_acts = batch_activations[:, seq_start:seq_end]
                chunk_tokens = batch_tokens[:, seq_start:seq_end]

                # Get memberships
                with torch.no_grad():
                    results = self.model.transform(chunk_acts, return_sparse=False)
                    memberships = results["memberships"]

                # Update statistics
                self.cluster_sizes += memberships.sum(dim=(0, 1)).cpu()

                # Extract examples
                self._process_chunk_examples(
                    memberships,
                    chunk_acts,
                    chunk_tokens,
                    top_examples,
                    batch_offset=batch_start,
                    seq_offset=seq_start,
                )

                # Clear memory
                del memberships
                torch.cuda.empty_cache()

        # Finalize examples
        self._finalize_examples(top_examples)

        # Convert sizes to numpy
        self.cluster_sizes = self.cluster_sizes.numpy()

    def _process_chunk_examples(
        self,
        memberships: torch.Tensor,
        activations: torch.Tensor,
        tokens: torch.Tensor,
        top_examples: Dict,
        batch_offset: int,
        seq_offset: int,
    ) -> None:
        """Process examples from a single data chunk"""
        batch_size, seq_len, n_clusters = memberships.shape
        window = self.config.context_window

        for k in range(n_clusters):
            # Get scores for this cluster
            scores = memberships[:, :, k].reshape(-1)

            # Find top examples
            top_k = min(self.config.max_examples - len(top_examples[k]), len(scores))
            if top_k <= 0:
                continue

            top_indices = torch.topk(scores, k=top_k)

            for score, flat_idx in zip(top_indices.values, top_indices.indices):
                # Convert flat index
                batch_idx = flat_idx // seq_len
                seq_idx = flat_idx % seq_len

                # Get context window
                start_idx = max(0, seq_idx - window // 2)
                end_idx = min(seq_len, seq_idx + window // 2)

                # Create example
                example = ClusterExample(
                    token_id=tokens[batch_idx, seq_idx].item(),
                    context_ids=tokens[batch_idx, start_idx:end_idx].cpu().tolist(),
                    token_pos=seq_idx - start_idx,
                    activation=activations[batch_idx, seq_idx].cpu().numpy(),
                    membership_score=score.item(),
                )

                top_examples[k].append(example)

    def _finalize_examples(self, top_examples: Dict) -> None:
        """Sort and store final examples"""
        self.cluster_examples = {
            k: sorted(examples, key=lambda x: x.membership_score, reverse=True)
            for k, examples in top_examples.items()
        }

    def print_cluster_examples(
        self, cluster_id: int, n_examples: Optional[int] = None
    ) -> None:
        """Print readable examples from a cluster with context"""
        if cluster_id not in self.cluster_examples:
            print(f"No examples found for cluster {cluster_id}")
            return

        examples = self.cluster_examples[cluster_id]
        if n_examples:
            examples = examples[:n_examples]

        print(
            f"\nCluster {cluster_id} Examples (Size: {self.cluster_sizes[cluster_id]:.2f}):"
        )
        print("=" * 80)

        for i, example in enumerate(examples, 1):
            # Decode text
            token = self.tokenizer.decode([example.token_id])
            prefix = self.tokenizer.decode(example.context_ids[: example.token_pos])
            suffix = self.tokenizer.decode(example.context_ids[example.token_pos + 1 :])

            print(f"\nExample {i} (score: {example.membership_score:.3f}):")
            print("-" * 40)
            print(f"{prefix}[[[{token}]]]{suffix}")

            # Print activation stats
            activation_norm = np.linalg.norm(example.activation)
            print(f"Activation norm: {activation_norm:.3f}")
            print("-" * 40)

    def plot_cluster_sizes(self, top_k: Optional[int] = None) -> Figure:
        """Plot interactive distribution of cluster sizes"""
        sizes = self.cluster_sizes
        if top_k:
            # Sort and get top k
            sorted_indices = np.argsort(-sizes)[:top_k]
            sizes = sizes[sorted_indices]
            cluster_ids = sorted_indices
        else:
            cluster_ids = np.arange(len(sizes))

        # Create interactive bar plot
        fig = go.Figure(
            data=[
                go.Bar(
                    x=cluster_ids,
                    y=sizes,
                    hovertemplate=(
                        "Cluster %{x}<br>" + "Size: %{y:.2f}<br>" + "<extra></extra>"
                    ),
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title="Cluster Sizes (Total Membership)",
            xaxis_title="Cluster Index",
            yaxis_title="Size",
            yaxis_type="log",
            template="plotly_white",
            showlegend=False,
            hovermode="x",
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        return fig

    def plot_cluster_activation_patterns(self, cluster_id: int) -> Figure:
        """Plot interactive activation patterns for a specific cluster"""
        if cluster_id not in self.cluster_examples:
            print(f"No examples found for cluster {cluster_id}")
            return

        examples = self.cluster_examples[cluster_id]

        # Create figure
        fig = go.Figure()

        # Add traces for each example
        for i, example in enumerate(examples):
            token = self.tokenizer.decode([example.token_id])

            fig.add_trace(
                go.Scatter(
                    y=example.activation,
                    name=f'"{token}" (score: {example.membership_score:.3f})',
                    hovertemplate=(
                        "Dimension %{x}<br>" + "Value: %{y:.3f}<br>" + "<extra></extra>"
                    ),
                )
            )

        # Update layout
        fig.update_layout(
            title=f"Activation Patterns for Cluster {cluster_id}",
            xaxis_title="Dimension",
            yaxis_title="Activation Value",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def plot_cluster_correlations(self, min_correlation: float = 0.1) -> Figure:
        """Plot interactive correlation heatmap between clusters"""
        correlations = self._compute_cluster_correlations()

        # Mask low correlations for clarity
        correlations[np.abs(correlations) < min_correlation] = 0

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=correlations,
                x=np.arange(len(correlations)),
                y=np.arange(len(correlations)),
                colorscale="RdBu",
                zmid=0,
                hoverongaps=False,
                hovertemplate=(
                    "Cluster %{x} ↔ Cluster %{y}<br>"
                    + "Correlation: %{z:.3f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

        # Update layout
        fig.update_layout(
            title="Cluster Correlations",
            xaxis_title="Cluster Index",
            yaxis_title="Cluster Index",
            template="plotly_white",
            width=800,
            height=800,
        )

        return fig

    def plot_position_patterns(self, top_k: Optional[int] = None) -> Figure:
        """Plot interactive position-wise activation patterns"""
        position_usage = self.model.cluster_stats["position_usage"].cpu().numpy()

        # Normalize by position
        usage_prob = position_usage / position_usage.sum(axis=0, keepdims=True)

        if top_k:
            # Select top k clusters by total usage
            total_usage = usage_prob.sum(axis=1)
            top_indices = np.argsort(-total_usage)[:top_k]
            usage_prob = usage_prob[top_indices]
            cluster_ids = top_indices
        else:
            cluster_ids = np.arange(len(usage_prob))

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=usage_prob,
                x=np.arange(usage_prob.shape[1]),
                y=cluster_ids,
                colorscale="Viridis",
                hoverongaps=False,
                hovertemplate=(
                    "Cluster %{y}<br>"
                    + "Position: %{x}<br>"
                    + "Usage: %{z:.3f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

        # Update layout
        fig.update_layout(
            title="Position-wise Cluster Usage Patterns",
            xaxis_title="Position",
            yaxis_title="Cluster Index",
            template="plotly_white",
            width=1000,
            height=600,
        )

        return fig

    def create_cluster_dashboard(self, cluster_id: int) -> Figure:
        """Create an interactive dashboard for a specific cluster"""
        # Create subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Activation Patterns",
                "Position Usage",
                "Related Clusters",
                "Token Distribution",
            ),
        )

        # Add activation patterns
        examples = self.cluster_examples[cluster_id]
        for example in examples:
            token = self.tokenizer.decode([example.token_id])
            fig.add_trace(
                go.Scatter(
                    y=example.activation,
                    name=f'"{token}" ({example.membership_score:.3f})',
                ),
                row=1,
                col=1,
            )

        # Add position usage
        position_usage = (
            self.model.cluster_stats["position_usage"][cluster_id].cpu().numpy()
        )
        fig.add_trace(go.Scatter(y=position_usage, fill="tozeroy"), row=1, col=2)

        # Add correlation heatmap
        correlations = self._compute_cluster_correlations()[cluster_id]
        top_corr_idx = np.argsort(-np.abs(correlations))[:10]  # Top 10 related clusters
        fig.add_trace(
            go.Bar(x=top_corr_idx, y=correlations[top_corr_idx]), row=2, col=1
        )

        # Add token distribution
        token_counts = defaultdict(float)
        for ex in examples:
            token_counts[ex.token_id] += ex.membership_score

        tokens, scores = zip(*sorted(token_counts.items(), key=lambda x: -x[1])[:10])
        token_labels = [self.tokenizer.decode([t]) for t in tokens]

        fig.add_trace(go.Bar(x=token_labels, y=scores), row=2, col=2)

        # Update layout
        fig.update_layout(
            title=f"Cluster {cluster_id} Analysis Dashboard",
            template="plotly_white",
            showlegend=False,
            width=1200,
            height=800,
        )

        return fig

    def find_clusters_for_token(self, token: str) -> List[Tuple[int, float]]:
        """Find clusters that commonly activate for a given token"""
        token_id = self.tokenizer.encode(token)[0]
        matches = []

        for cluster_id, examples in self.cluster_examples.items():
            matching = [
                ex.membership_score for ex in examples if ex.token_id == token_id
            ]
            if matching:
                avg_score = np.mean(matching)
                matches.append((cluster_id, avg_score))

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def print_cluster_details(
        self, cluster_id: int, n_examples: int = 10, show_alignments: bool = True
    ) -> None:
        """
        Print detailed analysis of a cluster focusing on examples and their relationships.

        Args:
            cluster_id: ID of cluster to analyze
            n_examples: Number of examples to show
            show_alignments: Whether to show how similar examples are to each other
        """
        if cluster_id not in self.cluster_examples:
            print(f"No examples found for cluster {cluster_id}")
            return

        examples = self.cluster_examples[cluster_id][:n_examples]

        # Calculate cluster density
        total_positions = sum(
            meta["total_sequences"] * meta["sequence_length"]
            for meta in self.chunk_metadata.values()
        )
        cluster_activations = self.cluster_stats["total_usage"][cluster_id].item()
        density = cluster_activations / total_positions

        print(f"\nCluster {cluster_id} Analysis")
        print("=" * 80)
        print(f"Activation Density: {density:.3%} of positions")
        print(
            f"Average Activation When Active: {cluster_activations / len(examples):.3f}"
        )
        print("-" * 80)

        # Print examples with context
        print("\nTop Examples:")
        for i, example in enumerate(examples, 1):
            # Get context and decode
            token = self.tokenizer.decode([example.token_id])
            prefix = self.tokenizer.decode(
                example.context_ids[: example.token_pos]
                if example.token_pos > 0
                else []
            )
            suffix = self.tokenizer.decode(
                example.context_ids[example.token_pos + 1 :]
                if example.token_pos + 1 < len(example.context_ids)
                else []
            )

            print(f"\nExample {i} (score: {example.membership_score:.3f}):")
            print(f"{prefix}[[[{token}]]]{suffix}")

            if show_alignments and i < len(examples):
                # Show similarity to next few examples
                print("\nSimilar examples:")
                for j in range(i + 1, min(i + 4, len(examples))):
                    other = examples[j]
                    other_token = self.tokenizer.decode([other.token_id])
                    similarity = np.dot(
                        example.activation / np.linalg.norm(example.activation),
                        other.activation / np.linalg.norm(other.activation),
                    )
                    print(
                        f"- '{other_token}' (score: {other.membership_score:.3f}, similarity: {similarity:.3f})"
                    )

            print("-" * 40)

    def get_cluster_vectors(self, cluster_id: int) -> Dict[str, np.ndarray]:
        """Get vectors associated with a cluster."""
        if cluster_id not in self.cluster_examples:
            raise ValueError(f"No examples found for cluster {cluster_id}")

        if not self.cluster_examples[cluster_id]:
            raise ValueError(f"Cluster {cluster_id} has no examples")

        center = self.model.centers[cluster_id].cpu().numpy()
        examples = self.cluster_examples[cluster_id]

        return {
            "center": center,
            "examples": [ex.activation for ex in examples],
            "example_tokens": [ex.token_id for ex in examples],
            "example_scores": [ex.membership_score for ex in examples],
        }

    def get_cluster_density_stats(self, cluster_id: int) -> Dict[str, float]:
        """Get detailed activation density statistics for a cluster."""
        if cluster_id not in self.cluster_examples:
            raise ValueError(f"No examples found for cluster {cluster_id}")

        # Calculate total positions across all data
        total_positions = sum(
            meta["total_sequences"] * meta["sequence_length"]
            for meta in self.chunk_metadata.values()
        )

        # Get cluster activations
        total_activations = self.cluster_stats["total_usage"][cluster_id].item()
        n_examples = len(self.cluster_examples[cluster_id])

        # Calculate various density metrics
        stats = {
            "activation_density": total_activations / total_positions,
            "average_activation": total_activations / n_examples
            if n_examples > 0
            else 0,
            "total_activations": total_activations,
            "unique_examples": n_examples,
            "positions_analyzed": total_positions,
        }

        return stats
