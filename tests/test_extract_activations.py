import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.extract_activations import (
    ExtractionConfig,
    StreamingPileDataset,
    IncrementalWhitener,
    ActivationExtractor,
)


# Fixtures
@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 128, dtype=torch.long),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }
    return tokenizer


@pytest.fixture
def mock_model():
    model = Mock()
    model.config.hidden_size = 768
    # Mock hidden states output
    model.return_value.hidden_states = [torch.randn(32, 128, 768) for _ in range(12)]
    return model


@pytest.fixture
def basic_config():
    return ExtractionConfig(
        model_name="test_model",
        batch_size=32,
        max_length=128,
        num_samples=100,
        chunk_size=50,
        device="cpu",
        output_dir="./test_outputs",
    )


# Test ExtractionConfig
def test_extraction_config_initialization():
    config = ExtractionConfig()
    assert config.exclude_tokens == ["bos", "pad"]
    assert Path(config.output_dir).exists()


def test_extraction_config_custom_exclude_tokens():
    config = ExtractionConfig(exclude_tokens=["test"])
    assert config.exclude_tokens == ["test"]


# Test IncrementalWhitener
def test_incremental_whitener_initialization():
    whitener = IncrementalWhitener(hidden_dim=768)
    assert whitener.hidden_dim == 768
    assert torch.all(whitener.mean == 0)
    assert whitener.n == 0


def test_incremental_whitener_update():
    whitener = IncrementalWhitener(hidden_dim=2)
    batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mask = torch.ones(2, dtype=torch.bool)

    whitener.update(batch, mask)
    assert whitener.n == 2
    assert torch.allclose(whitener.mean, torch.tensor([2.0, 3.0]))


def test_whitener_save_load(tmp_path):
    whitener = IncrementalWhitener(hidden_dim=2)
    batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mask = torch.ones(2, dtype=torch.bool)

    whitener.update(batch, mask)
    whitener.finalize()

    save_path = tmp_path / "whitening_params.pt"
    whitener.save(str(save_path))

    new_whitener = IncrementalWhitener(hidden_dim=2)
    new_whitener.load(str(save_path))

    assert torch.allclose(whitener.mean, new_whitener.mean)
    assert torch.allclose(whitener.transform, new_whitener.transform)


# Test ActivationExtractor
@pytest.mark.asyncio
async def test_activation_extractor_initialization(
    basic_config, mock_model, mock_tokenizer
):
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ), patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        extractor = ActivationExtractor(basic_config)
        assert extractor.config == basic_config
        assert extractor.whitener is None


@pytest.mark.asyncio
async def test_compute_whitening_statistics(basic_config, mock_model, mock_tokenizer):
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ), patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    ), patch("src.extract_activations.StreamingPileDataset") as mock_dataset:
        # Mock dataset iteration
        mock_dataset.return_value = [
            {
                "input_ids": torch.ones(128, dtype=torch.long),
                "attention_mask": torch.ones(128, dtype=torch.long),
            }
            for _ in range(5)
        ]

        extractor = ActivationExtractor(basic_config)
        extractor.compute_whitening_statistics()

        assert extractor.whitener is not None
        assert Path(basic_config.output_dir).exists()
        assert Path(f"{basic_config.output_dir}/whitening_params.pt").exists()


# Test StreamingPileDataset
@pytest.mark.asyncio
async def test_streaming_dataset(mock_tokenizer):
    with patch("datasets.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = [{"text": "test text"} for _ in range(5)]

        dataset = StreamingPileDataset(
            tokenizer=mock_tokenizer, max_length=128, num_samples=3
        )

        items = list(dataset)
        assert len(items) == 3
        assert all(isinstance(item, dict) for item in items)
        assert all("input_ids" in item and "attention_mask" in item for item in items)


# Integration tests
def test_end_to_end_small_sample(tmp_path):
    """Test the entire pipeline with a small synthetic dataset"""
    config = ExtractionConfig(
        model_name="test_model",
        batch_size=2,
        max_length=4,
        num_samples=4,
        chunk_size=2,
        device="cpu",
        output_dir=str(tmp_path),
    )

    # Create synthetic model and tokenizer
    model = Mock()
    model.config.hidden_size = 4
    model.return_value.hidden_states = [torch.randn(2, 4, 4)]

    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    config.model = model
    config.tokenizer = tokenizer

    extractor = ActivationExtractor(config)
    extractor.compute_whitening_statistics()
    extractor.process_and_store_whitened()

    # Verify outputs
    assert Path(tmp_path / "whitening_params.pt").exists()
    chunk_files = list(tmp_path.glob("chunk_*.npz"))
    assert len(chunk_files) > 0
