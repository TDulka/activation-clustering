# Activation Extraction and Storage Strategy

## Overview
This document outlines our approach to extracting, processing, and storing neural network activations for large-scale analysis. We will process a subset of the Pile Uncopyrighted dataset (available on HuggingFace) while managing memory and storage constraints.

## Dataset and System Requirements

### Dataset
- Source: [The Pile Uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted)
- Subset Size: 300k samples (approximately 33.6% of the training split)
- Sequence Length: 128 tokens per sample
- Hidden State Dimension: 2304

### System Constraints
- Available Storage: 200GB
- Expected Storage Requirements:
  - Raw Activations (float16): ~170GB (300k × 128 × 2304 × 2 bytes)
  - Final Compressed Size: ~85-100GB after whitening and compression

## Token Selection Strategy
- **Excluded Tokens:**
  - Beginning-of-Sequence (BOS) tokens: Have unique distributional properties
  - Padding tokens: Contain no meaningful information
- **Token Masking:**
  - Track valid token positions for each sample
  - Use masks during statistical computations
  - Store mask information for downstream analysis

## Processing Pipeline

### 1. First Pass: Computing Global Whitening Statistics
- Stream through data in fixed-size chunks (1000 samples)
- Filter out BOS and padding tokens
- Compute running statistics using Welford's online algorithm across all valid token positions
- Store minimal memory footprint:
  - Global mean vector (2304 dimensions)
  - Global covariance matrix (2304 × 2304)
  - Expected storage: ~20MB for parameters

### 2. Second Pass: Generating Whitened Activations
- Process same chunks of 1000 samples
- Apply global whitening transformation to valid tokens
- Maintain original positions and mask information
- Immediately save processed chunks to disk
- Use NPZ compression (expected 1.5-2x reduction)
- Expected final size: ~85-100GB

### 3. Dataset Creation
Convert processed activations into a HuggingFace dataset:
- Memory-mapped format for efficient access
- Streaming capabilities for large-scale processing
- Structure:
  ```python
  {
    'whitened_activations': array[300000, 128, 2304],
    'metadata': {
      'whitening_params': {
          'mean': array[2304],
          'transform_matrix': array[2304, 2304]
      },
      'attention_mask': array[300000, 128],  # Valid token masks
      'sample_ids': list[300000],
      'processing_config': dict
    }
  }
  ```

## Implementation Details

### Key Components

1. **IncrementalWhitener**
   - Handles online computation of global whitening statistics
   - Implements token filtering and masking
   - Saves/loads whitening parameters
   - Applies whitening transformations

2. **ChunkedActivationExtractor**
   - Manages batch processing of samples
   - Interfaces with model for activation extraction
   - Handles token selection and masking
   - Handles efficient disk I/O

3. **DatasetCreator**
   - Converts processed chunks to HuggingFace dataset
   - Implements memory mapping and streaming
   - Manages metadata and validation
   - Preserves mask information

### Storage Strategy
1. Temporary Storage:
   - Raw activation chunks (deleted after processing)
   - Whitening parameters (~20MB)

2. Permanent Storage:
   - Compressed whitened activations (~85-100GB)
   - HuggingFace dataset files with masks
   - Processing metadata and configuration

### Error Handling
- Checkpoint saving after each chunk
- Validation of processed data
- Recovery mechanisms for interrupted processing
- Mask validation and consistency checks

## Usage Workflow

1. **Initialization**
   ```python
   extractor = ChunkedActivationExtractor(
       model_name="model_id",
       output_dir="./processed_activations",
       chunk_size=1000,
       exclude_tokens=['bos', 'pad']
   )
   ```

2. **Computing Whitening Parameters**
   ```python
   whitening_params = extractor.compute_whitening_statistics(dataset)
   ```

3. **Processing and Storing Activations**
   ```python
   extractor.process_and_store_whitened(
       dataset,
       whitening_params,
       compression=True
   )
   ```

4. **Creating HuggingFace Dataset**
   ```python
   hf_dataset = extractor.to_huggingface_dataset()
   hf_dataset.push_to_hub("username/dataset_name")
   ```

## Monitoring and Validation

- Progress tracking per chunk
- Memory usage monitoring
- Storage space checking
- Data integrity validation
- Token mask validation
- Processing time estimation

## Future Optimizations

1. **Optional Dimensionality Reduction**
   - PCA after whitening
   - Configurable dimension reduction
   - Storage of transformation matrices

2. **Advanced Compression**
   - Quantization options
   - Custom compression algorithms
   - Sparse storage formats

3. **Parallel Processing**
   - Multi-GPU support
   - Distributed computation
   - Parallel I/O operations

4. **Advanced Token Selection**
   - Configurable token filtering criteria
   - Statistical outlier detection
   - Position-based selection options