import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from activation_clustering.extract_activations import (
    ExtractionConfig,
    StreamingPileDataset,
    IncrementalWhitener,
    ActivationExtractor,
)


# Fixtures
@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()

    def tokenize_func(*args, **kwargs):
        # Return consistent tensor shapes
        return {
            "input_ids": torch.ones(
                kwargs.get("batch_size", 1),
                kwargs.get("max_length", 128),
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                kwargs.get("batch_size", 1),
                kwargs.get("max_length", 128),
                dtype=torch.long,
            ),
        }

    tokenizer.side_effect = tokenize_func
    return tokenizer


@pytest.fixture
def mock_model():
    model = Mock()
    model.config.hidden_size = 768

    def forward(*args, **kwargs):
        batch_size = args[0].shape[0]  # Get batch size from input
        output = Mock()
        # Match hidden states shape with batch size
        output.hidden_states = [torch.randn(batch_size, 128, 768) for _ in range(12)]
        return output

    model.side_effect = forward
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


@pytest.fixture(autouse=True)
def cleanup_test_outputs():
    """Cleanup test outputs before and after each test."""
    test_outputs = Path("./test_outputs")
    if test_outputs.exists():
        for file in test_outputs.glob("*"):
            file.unlink()
        test_outputs.rmdir()

    yield

    if test_outputs.exists():
        for file in test_outputs.glob("*"):
            file.unlink()
        test_outputs.rmdir()


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
async def test_compute_whitening_statistics(
    basic_config, mock_model, mock_tokenizer, tmp_path
):
    # Update config to use temporary directory
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory explicitly
    basic_config.output_dir = str(output_dir)

    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    ), patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    ), patch(
        "activation_clustering.extract_activations.StreamingPileDataset"
    ) as mock_dataset_class:
        # Create a proper mock dataset that works with DataLoader
        class MockDataset:
            def __init__(self):
                self.data = [
                    {
                        "input_ids": torch.ones(
                            basic_config.max_length, dtype=torch.long
                        ),
                        "attention_mask": torch.ones(
                            basic_config.max_length, dtype=torch.long
                        ),
                    }
                    for _ in range(5)
                ]

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        mock_dataset_class.return_value = MockDataset()

        extractor = ActivationExtractor(basic_config)
        extractor.compute_whitening_statistics()

        assert extractor.whitener is not None
        assert Path(basic_config.output_dir).exists()
        assert Path(f"{basic_config.output_dir}/whitening_params.pt").exists()


# Test StreamingPileDataset
@pytest.mark.asyncio
async def test_streaming_dataset(mock_tokenizer):
    # Create a proper mock streaming dataset
    class MockStreamingDataset:
        def __init__(self):
            self.data = [{"text": f"test text {i}"} for i in range(5)]
            self._iter_counter = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._iter_counter >= len(self.data):
                raise StopIteration
            item = self.data[self._iter_counter]
            self._iter_counter += 1
            return item

        def take(self, n):
            """Implements datasets.IterableDataset.take"""
            self.data = self.data[:n]
            return self

    # Create a mock load_dataset function that returns our streaming dataset
    def mock_load(*args, split=None, streaming=None):
        assert args[0] == "monology/pile-uncopyrighted"
        assert split == "train"
        assert streaming is True
        return MockStreamingDataset()

    # Mock the tokenizer to return proper tensor outputs
    class MockTokenizerOutput:
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

    def mock_tokenize(*args, **kwargs):
        return MockTokenizerOutput(
            input_ids=torch.ones(1, kwargs.get("max_length", 128)),
            attention_mask=torch.ones(1, kwargs.get("max_length", 128)),
        )

    mock_tokenizer.side_effect = mock_tokenize

    # Use the patch where datasets.load_dataset is imported
    with patch("activation_clustering.extract_activations.load_dataset", mock_load):
        dataset = StreamingPileDataset(
            tokenizer=mock_tokenizer, max_length=128, num_samples=3
        )

        items = list(dataset)
        assert len(items) == 3
        assert all(isinstance(item, dict) for item in items)
        assert all("input_ids" in item and "attention_mask" in item for item in items)
        # Verify tensor shapes
        assert all(item["input_ids"].shape == torch.Size([128]) for item in items)
        assert all(item["attention_mask"].shape == torch.Size([128]) for item in items)


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

    def model_forward(*args, **kwargs):
        batch_size = args[0].shape[0]
        output = Mock()
        # Make sure hidden states match the expected shape
        output.hidden_states = [
            torch.randn(batch_size, config.max_length, model.config.hidden_size)
        ]
        return output

    model.side_effect = model_forward

    # Fix tokenizer mock
    class MockTokenizerOutput:
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

    def tokenize_func(*args, **kwargs):
        batch_size = kwargs.get("batch_size", 1)
        return MockTokenizerOutput(
            input_ids=torch.ones(batch_size, config.max_length, dtype=torch.long),
            attention_mask=torch.ones(batch_size, config.max_length, dtype=torch.long),
        )

    tokenizer = Mock()
    tokenizer.side_effect = tokenize_func

    config.model = model
    config.tokenizer = tokenizer

    # Mock the dataset loading
    with patch(
        "activation_clustering.extract_activations.load_dataset"
    ) as mock_load_dataset:
        # Create a proper mock dataset
        class MockStreamingDataset:
            def __iter__(self):
                return iter([{"text": f"test text {i}"} for i in range(4)])

            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {"text": f"test text {idx}"}

        def mock_load(*args, **kwargs):
            return MockStreamingDataset()

        mock_load_dataset.side_effect = mock_load

        extractor = ActivationExtractor(config)
        extractor.compute_whitening_statistics()
        extractor.process_and_store_whitened()

        # Verify outputs
        assert Path(tmp_path / "whitening_params.pt").exists()
        chunk_files = list(tmp_path.glob("chunk_*.npz"))
        assert len(chunk_files) > 0

        # New: Verify chunk contents include token IDs
        chunk_data = np.load(chunk_files[0])
        assert "token_ids" in chunk_data.files
        assert "activations" in chunk_data.files
        assert "attention_mask" in chunk_data.files

        # Check shapes match
        assert chunk_data["token_ids"].shape[1] == config.max_length
        assert chunk_data["activations"].shape[1] == config.max_length
        assert chunk_data["attention_mask"].shape[1] == config.max_length


# Add new test specifically for token ID storage
def test_save_chunk_includes_token_ids(tmp_path):
    # Create mock model and tokenizer
    mock_model = Mock()
    mock_model.config.hidden_size = 768
    mock_tokenizer = Mock()

    config = ExtractionConfig(
        model_name="test_model",  # Add model_name
        output_dir=str(tmp_path),
        model=mock_model,  # Add mock model
        tokenizer=mock_tokenizer,  # Add mock tokenizer
    )

    extractor = ActivationExtractor(config)

    # Create sample data
    batch_size = 2
    seq_length = 4
    hidden_dim = 768

    activations = [torch.randn(batch_size, seq_length, hidden_dim)]
    masks = [torch.ones(batch_size, seq_length)]
    token_ids = [torch.randint(0, 1000, (batch_size, seq_length))]

    # Save chunk
    extractor._save_chunk(activations, masks, token_ids, chunk_id=0)

    # Load and verify
    chunk_path = Path(tmp_path) / "chunk_0.npz"
    assert chunk_path.exists()

    chunk_data = np.load(chunk_path)
    assert "token_ids" in chunk_data.files
    assert chunk_data["token_ids"].shape == (batch_size, seq_length)
    assert chunk_data["activations"].shape[:2] == (batch_size, seq_length)
    assert chunk_data["attention_mask"].shape == (batch_size, seq_length)
