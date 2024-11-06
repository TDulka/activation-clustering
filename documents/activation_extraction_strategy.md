# Activation Extraction Strategy

## Overview
This document describes our approach to extracting and processing neural network activations from language models. The system extracts activations from specific layers, applies whitening transformations, and stores the results efficiently for further analysis.

## Core Components

### 1. ExtractionConfig
Configuration dataclass that controls the extraction process:
```python
@dataclass
class ExtractionConfig:
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_name: Optional[str] = None
    layer_index: int = -1  # Default to last layer
    batch_size: int = 32
    max_length: int = 128
    num_samples: int = 300_000
    chunk_size: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    output_dir: str = "./processed_activations"
    exclude_tokens: List[str] = ["bos", "pad"]
```

### 2. StreamingPileDataset
Handles efficient data streaming from the Pile dataset:
- Uses HuggingFace's streaming capabilities
- Tokenizes text on-the-fly
- Applies length constraints and padding
- Returns batches of token IDs and attention masks

### 3. IncrementalWhitener
Computes whitening statistics incrementally:
- Uses Welford's online algorithm for stable mean/covariance computation
- Supports saving/loading of whitening parameters
- Provides forward (whitening) and reverse transformations

### 4. ActivationExtractor
Main class orchestrating the extraction process:
- Manages model and tokenizer initialization
- Coordinates the two-pass extraction process
- Handles chunked storage of processed activations

## Extraction Process

### First Pass: Computing Whitening Statistics
1. Stream data through the model in batches
2. Extract activations from the specified layer
3. Update running statistics using IncrementalWhitener
4. Save whitening parameters for later use

### Second Pass: Processing and Storage
1. Apply whitening transformation to new batches
2. Store processed data in chunks containing:
   - Whitened activations
   - Attention masks
   - Token IDs
3. Use compressed NPZ format for efficient storage

## Data Format

### Chunk Storage Format
Each chunk is stored as an NPZ file containing:
```python
{
    'activations': array[batch_size, seq_length, hidden_dim],  # Whitened activations
    'attention_mask': array[batch_size, seq_length],  # Valid token masks
    'token_ids': array[batch_size, seq_length]  # Original token IDs
}
```

### Whitening Parameters
Stored separately as a PyTorch file containing:
```python
{
    'mean': tensor[hidden_dim],
    'transform': tensor[hidden_dim, hidden_dim],
    'reverse_transform': tensor[hidden_dim, hidden_dim],
    'n_samples': int
}
```

## Usage Example

```python
# Initialize configuration
config = ExtractionConfig(
    model_name="google/gemma-2b",
    layer_index=-1,
    num_samples=300_000,
    output_dir="./processed_activations"
)

# Create extractor
extractor = ActivationExtractor(config)

# First pass: compute whitening statistics
extractor.compute_whitening_statistics()

# Second pass: process and store whitened activations
extractor.process_and_store_whitened()
```

## Key Features

### Memory Efficiency
- Streaming dataset processing
- Incremental statistics computation
- Chunked storage of results
- Optional float16 precision

### Robustness
- Excludes special tokens (BOS, padding)
- Saves progress in chunks
- Includes token IDs for analysis

### Flexibility
- Configurable batch and chunk sizes
- Support for different model architectures
- Adjustable sequence lengths
- Custom token exclusion

## Implementation Notes

### Token Handling
- Attention masks track valid tokens
- Special tokens (BOS, padding) are excluded from statistics
- Original token IDs are preserved for analysis

### Storage Management
- Chunks are saved progressively during processing
- NPZ compression reduces storage requirements
- Each chunk contains complete context (masks, IDs)
- Whitening parameters stored separately for reuse

### Error Handling
- Validates configuration parameters
- Checks for required whitening statistics
- Handles empty batches gracefully
- Maintains processing state

## Future Considerations

1. **Optimization Opportunities**
   - Multi-GPU processing support
   - Advanced compression techniques
   - Parallel chunk processing

2. **Additional Features**
   - Support for multiple layers
   - Custom activation functions
   - Advanced token filtering
   - Progress tracking and reporting

3. **Analysis Tools**
   - Activation visualization
   - Statistical analysis utilities
   - Token-level correlations
   - Clustering integration