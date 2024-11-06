import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import List

from activation_clustering.cluster_analyzer import ClusterAnalyzer, AnalysisConfig
from activation_clustering.soft_clustering import (
    SequentialSoftClustering,
    ClusteringConfig,
)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing"""

    class MockTokenizer:
        def encode(self, text: str) -> List[int]:
            return [hash(text) % 1000]  # Deterministic mock encoding

        def decode(self, token_ids: List[int]) -> str:
            if not token_ids:
                return ""  # Return empty string for empty list
            return f"token_{token_ids[0]}"  # Deterministic mock decoding

        def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
            return [f"token_{tid}" for tid in token_ids]

    return MockTokenizer()


@pytest.fixture
def mock_results(temp_data_dir, mock_data, mock_tokenizer):
    """Create mock clustering results"""
    # Create clustering model
    config = ClusteringConfig(n_clusters=3, d_model=8, device="cpu")
    model = SequentialSoftClustering(config)

    # Save model
    model_path = temp_data_dir / "model.pt"
    model.save(str(model_path))

    # Create mock results
    results_dir = temp_data_dir / "results"
    results_dir.mkdir()

    # Create mock chunk results
    for i in range(2):
        chunk_path = results_dir / f"chunk_{i}_results.npz"
        np.savez_compressed(
            chunk_path,
            indices=np.array([[0, 0, 0], [0, 1, 1]]),
            values=np.array([0.5, 0.5]),
            metadata={"total_sequences": 1, "sequence_length": 2},
        )

    return {
        "model_path": str(model_path),
        "results_dir": str(results_dir),
        "tokenizer": mock_tokenizer,
    }


def test_analyzer_initialization(mock_results):
    """Test proper initialization of analyzer"""
    analyzer = ClusterAnalyzer(
        mock_results["model_path"],
        mock_results["results_dir"],
        mock_results["tokenizer"],
    )

    assert len(analyzer.chunk_metadata) == 2
    assert all("total_sequences" in meta for meta in analyzer.chunk_metadata.values())


def test_print_cluster_details(mock_results):
    """Test cluster details printing"""
    analyzer = ClusterAnalyzer(
        mock_results["model_path"],
        mock_results["results_dir"],
        mock_results["tokenizer"],
    )

    # Process some mock examples
    activations = torch.randn(4, 6, 8)  # [batch, seq_len, d_model]
    tokens = torch.randint(0, 1000, (4, 6))  # [batch, seq_len]

    analyzer.process_examples(activations, tokens)

    # Test printing (should not raise errors)
    analyzer.print_cluster_details(0, n_examples=2)

    # Check cluster vectors
    vectors = analyzer.get_cluster_vectors(0)
    assert "center" in vectors
    assert "examples" in vectors
    assert "example_tokens" in vectors
    assert "example_scores" in vectors


def test_cluster_density_stats(mock_results):
    """Test density statistics computation"""
    analyzer = ClusterAnalyzer(
        mock_results["model_path"],
        mock_results["results_dir"],
        mock_results["tokenizer"],
    )

    # Process some mock examples
    activations = torch.randn(4, 6, 8)
    tokens = torch.randint(0, 1000, (4, 6))
    analyzer.process_examples(activations, tokens)

    stats = analyzer.get_cluster_density_stats(0)
    assert "activation_density" in stats
    assert "average_activation" in stats
    assert "total_activations" in stats
    assert "unique_examples" in stats
    assert "positions_analyzed" in stats

    assert 0 <= stats["activation_density"] <= 1
    assert stats["unique_examples"] > 0


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_data():
    """Generate mock activation data for testing"""
    batch_size, seq_len, d_model = 4, 6, 8

    activations = torch.randn(batch_size, seq_len, d_model)
    activations = torch.nn.functional.normalize(activations, dim=-1)

    attention_mask = torch.ones(batch_size, seq_len)
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))

    return {
        "activations": activations,
        "attention_mask": attention_mask,
        "token_ids": token_ids,
    }
