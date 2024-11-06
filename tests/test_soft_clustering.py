import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from activation_clustering.soft_clustering import (
    SequentialSoftClustering,
    ClusteringConfig,
    analyze_sequential_clusters,
)


@pytest.fixture
def clustering_config():
    """Basic clustering configuration for testing"""
    return ClusteringConfig(
        n_clusters=3,
        d_model=8,
        temperature=1.0,
        min_membership=0.01,
        device="cpu",  # Use CPU for testing
        n_exemplars=5,
        context_window=4,
    )


@pytest.fixture
def mock_data():
    """Generate mock activation data for testing"""
    batch_size, seq_len, d_model = 4, 6, 8

    # Create synthetic data with clear cluster structure
    activations = torch.randn(batch_size, seq_len, d_model)
    activations = torch.nn.functional.normalize(activations, dim=-1)

    attention_mask = torch.ones(batch_size, seq_len)
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))

    return {
        "activations": activations,
        "attention_mask": attention_mask,
        "token_ids": token_ids,
    }


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def create_mock_chunks(temp_dir: Path, mock_data: dict, n_chunks: int = 3):
    """Create mock chunk files for testing"""
    chunk_paths = []
    for i in range(n_chunks):
        chunk_path = temp_dir / f"chunk_{i}.npz"
        np.savez_compressed(
            chunk_path,
            activations=mock_data["activations"].numpy(),
            attention_mask=mock_data["attention_mask"].numpy(),
            token_ids=mock_data["token_ids"].numpy(),
        )
        chunk_paths.append(str(chunk_path))
    return chunk_paths


def test_clustering_initialization(clustering_config):
    """Test proper initialization of clustering model"""
    model = SequentialSoftClustering(clustering_config)

    assert model.centers.shape == (
        clustering_config.n_clusters,
        clustering_config.d_model,
    )
    assert torch.allclose(
        torch.norm(model.centers, dim=1), torch.ones(clustering_config.n_clusters)
    )
    assert len(model.exemplars) == 0
    assert model.cluster_stats["total_usage"].shape == (clustering_config.n_clusters,)


def test_chunk_processing(clustering_config, mock_data, temp_data_dir):
    """Test processing of data chunks"""
    model = SequentialSoftClustering(clustering_config)
    chunk_paths = create_mock_chunks(temp_data_dir, mock_data)
    output_dir = temp_data_dir / "results"

    # Process chunks
    model.fit_chunks(chunk_paths, str(output_dir))

    # Check output files exist
    for i in range(len(chunk_paths)):
        assert (output_dir / f"chunk_{i}_results.npz").exists()

    # Load and verify results
    chunk_results = np.load(output_dir / "chunk_0_results.npz", allow_pickle=True)
    assert "indices" in chunk_results
    assert "values" in chunk_results
    assert "metadata" in chunk_results


def test_exemplar_tracking(clustering_config, mock_data):
    """Test exemplar collection and management"""
    model = SequentialSoftClustering(clustering_config)

    # Process single chunk
    chunk_results = model._process_chunk(
        mock_data["activations"].numpy(),
        mock_data["attention_mask"].numpy(),
        mock_data["token_ids"].numpy(),
        chunk_idx=0,
    )

    # Check exemplars were collected
    for k in range(clustering_config.n_clusters):
        assert len(model.exemplars[k]) <= clustering_config.n_exemplars
        if len(model.exemplars[k]) > 0:
            exemplar = model.exemplars[k][0][
                2
            ]  # Get exemplar dict from tuple (score, idx, exemplar)
            assert "chunk_idx" in exemplar
            assert isinstance(exemplar["chunk_idx"], int)


def test_save_load(clustering_config, temp_data_dir):
    """Test model serialization"""
    model = SequentialSoftClustering(clustering_config)
    save_path = temp_data_dir / "model.pt"

    # Save model
    model.save(str(save_path))

    # Load model
    loaded_model = SequentialSoftClustering.load(str(save_path))

    # Check configuration
    assert loaded_model.config.n_clusters == clustering_config.n_clusters
    assert loaded_model.config.d_model == clustering_config.d_model

    # Check centers
    assert torch.allclose(loaded_model.centers, model.centers)

    # Check statistics
    for k, v in model.cluster_stats.items():
        assert torch.allclose(loaded_model.cluster_stats[k], v)


def test_sparse_representation(clustering_config, mock_data):
    """Test creation of sparse membership representation"""
    model = SequentialSoftClustering(clustering_config)

    # Get memberships
    memberships = model.transform(mock_data["activations"])["memberships"]

    # Create sparse representation
    sparse_results = model._create_sparse_representation(memberships)

    assert "indices" in sparse_results
    assert "values" in sparse_results
    assert sparse_results["indices"].shape[1] == 3  # (batch, seq, cluster) indices
    assert len(sparse_results["values"]) == len(sparse_results["indices"])
    assert torch.all(sparse_results["values"] > clustering_config.min_membership)


@pytest.mark.parametrize("n_clusters,d_model", [(2, 8), (5, 16), (10, 32)])
def test_different_dimensions(n_clusters, d_model):
    """Test clustering with different dimensions"""
    config = ClusteringConfig(n_clusters=n_clusters, d_model=d_model, device="cpu")
    model = SequentialSoftClustering(config)

    # Check dimensions
    assert model.centers.shape == (n_clusters, d_model)
    assert model.cluster_stats["total_usage"].shape == (n_clusters,)
    assert model.cluster_stats["position_usage"].shape == (
        n_clusters,
        config.context_window,
    )


def test_analyze_clusters(clustering_config, mock_data):
    """Test cluster analysis functionality"""
    model = SequentialSoftClustering(clustering_config)

    stats = analyze_sequential_clusters(
        model, mock_data["activations"], mock_data["token_ids"]
    )

    assert "total_usage" in stats
    assert "avg_assignments_per_pos" in stats
    assert "position_usage" in stats
    assert "examples" in stats

    assert len(stats["examples"]) == clustering_config.n_clusters
    for k in range(clustering_config.n_clusters):
        cluster_examples = stats["examples"][k]
        assert "scores" in cluster_examples
        assert "positions" in cluster_examples
        assert "tokens" in cluster_examples


def test_transform_method(clustering_config, mock_data):
    """Test activation transformation to memberships"""
    model = SequentialSoftClustering(clustering_config)

    # Test dense output
    results = model.transform(mock_data["activations"], return_sparse=False)
    memberships = results["memberships"]

    assert memberships.shape == (
        mock_data["activations"].shape[0],
        mock_data["activations"].shape[1],
        clustering_config.n_clusters,
    )
    assert torch.allclose(
        memberships.sum(dim=-1), torch.ones_like(memberships.sum(dim=-1))
    )

    # Test sparse output
    sparse_results = model.transform(mock_data["activations"], return_sparse=True)
    assert all(
        v.shape[0] == sparse_results["indices"].shape[0]
        for v in sparse_results.values()
    )


def test_memory_management(clustering_config):
    """Test memory-efficient point sampling"""
    model = SequentialSoftClustering(clustering_config)

    # Create large input
    large_batch = torch.randn(1000, 128, clustering_config.d_model)
    mask = torch.ones(1000, 128)

    # Sample points
    sampled = model._sample_points(large_batch, mask, max_points=1000)

    assert len(sampled) <= 1000
    assert sampled.shape[1] == clustering_config.d_model
    assert torch.allclose(
        torch.norm(sampled, dim=1), torch.ones(len(sampled)), atol=1e-6
    )


def test_statistics_tracking(clustering_config, mock_data):
    """Test cluster statistics updates"""
    model = SequentialSoftClustering(clustering_config)

    # Get initial stats
    initial_usage = model.cluster_stats["total_usage"].clone()
    initial_pos_usage = model.cluster_stats["position_usage"].clone()

    # Process data
    memberships = model.transform(mock_data["activations"], return_sparse=False)[
        "memberships"
    ]
    model._update_statistics(memberships, mock_data["attention_mask"])

    # Check stats were updated
    assert not torch.allclose(model.cluster_stats["total_usage"], initial_usage)
    assert not torch.allclose(model.cluster_stats["position_usage"], initial_pos_usage)

    # Check stats are reasonable
    assert torch.all(model.cluster_stats["total_usage"] >= 0)
    assert torch.all(model.cluster_stats["position_usage"] >= 0)


@pytest.mark.parametrize("batch_size,seq_len", [(2, 4), (4, 8), (8, 16)])
def test_different_sequence_lengths(clustering_config, batch_size, seq_len):
    """Test handling of different sequence lengths"""
    model = SequentialSoftClustering(clustering_config)

    # Create data with different dimensions
    activations = torch.randn(batch_size, seq_len, clustering_config.d_model)
    attention_mask = torch.ones(batch_size, seq_len)

    # Process data
    results = model.transform(activations)
    assert results["indices"].shape[1] == 3  # (batch, seq, cluster) indices
    assert torch.all(results["indices"][:, 0] < batch_size)
    assert torch.all(results["indices"][:, 1] < seq_len)
