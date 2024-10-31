# Gemma-2-2b Clustering vs SAE Analysis Project

## Project Overview

This project aims to compare clustering-based approaches with Sparse Autoencoders 
(SAEs) for analyzing and manipulating neural network representations in Gemma-2-2b 
Layer 12. We will compare against existing SAE results trained on the 
pile-uncopyrighted dataset.

## Technical Specifications

### Model Details
- Model: Gemma-2-2b
- Target Layer: 12
- Activation Dimension: 2304
- Activation Format: bfloat16 (2 bytes per value)
- Comparison SAE Sizes: Multiple variants available

### Dataset
- Source: monology/pile-uncopyrighted
- Full Size: 31.5GB text data
- Estimated Total Tokens: ~1.26 billion
- Development Sample: 4.34 million tokens (0.34% sampling rate)

### Storage Requirements
- Per-token Storage: 4.608 KB (2304 dims × bfloat16)
- Development Sample Target: 20GB
- Full Dataset Storage (theoretical): ~5.8 TB
- Expected Compression Ratio: 1.5-2x reduction

## Implementation Plan

### Phase 1: Initial Setup & Sampling ✓

1. [x] Set up environment and dependencies
   - Poetry configuration complete
   - Core dependencies specified
   - Development tools configured
2. [x] Initialize Gemma-2-2b with layer 12 activation hooks
   - Implemented in ActivationExtractor class
   - Supports configurable layer selection
3. [x] Implement sampling pipeline
   - Stratified random sampling across documents
   - Preserve 512 token context windows
   - Maintain source distribution
4. [x] Create activation storage architecture
   - Implemented efficient NPZ storage
   - Metadata tracking system in place

### Phase 2: Storage Architecture

1. **Storage Format** ✓
   - Primary: NPZ compressed arrays (implemented)
   - Metadata: Basic tracking implemented
   - Directory structure:
   ```
   activations/
   ├── chunks/
   │   ├── chunk_0000.npz
   │   ├── chunk_0001.npz
   │   └── ...
   └── metadata.json
   ```

2. **Implementation Stages**
   - [x] Basic NumPy storage implementation
   - [x] Metadata tracking system
   - [x] Compression support
   - [ ] Parallel processing
   - [ ] Caching layer
   - [x] Performance monitoring

### Phase 3: Clustering Implementation

1. [ ] Implement basic k-means clustering
2. [ ] Add density modeling
3. [ ] Create feature interpretation pipeline
4. [ ] Set up evaluation metrics

### Phase 4: Analysis & Scaling

1. [ ] Compare with existing SAE results
2. [ ] Evaluate computational efficiency
3. [ ] Document findings and insights
4. [ ] Plan scaling to larger sample if needed

## Key Experiments

1. **Clustering Quality Analysis**
   - Compare different numbers of clusters (64, 128, 256)
   - Evaluate against SAE feature dimensions
   - Measure silhouette scores and interpretation quality

2. **Computational Efficiency**
   - Training time comparison
   - Memory usage analysis
   - Inference speed benchmarks

3. **Feature Interpretation**
   - Analyze cluster centroids
   - Compare with SAE features
   - Evaluate interpretability metrics

## Success Metrics

- Clustering quality vs SAE reconstruction quality
- Computational resource usage comparison
- Feature interpretation clarity
- Implementation simplicity and maintainability

## Next Immediate Steps (Updated)

1. [ ] **Data Validation & Analysis**
   - Verify data integrity across chunks
   - Calculate overall activation statistics
   - Check for any anomalies or outliers

2. [ ] **Preprocessing Pipeline**
   - Implement activation normalization
   - Add whitening transformation
   - Set up data loading for clustering

3. [ ] **Initial Clustering Setup**
   - Start with small-scale k-means (n=256)
   - Implement basic evaluation metrics
   - Compare with existing SAE dimensionality

4. [ ] **Optimization & Scaling**
   - Profile memory usage
   - Optimize chunk loading
   - Plan for larger-scale runs if needed

## Technical Notes

### Data Collection Specifications
- Batch Size: 32
- Chunks Created: ~938 (30,000/32)
- Format: NPZ compressed files
- Content: Raw activations + token IDs

### Resource Usage
- Memory: Managed through batched processing
- Storage: Compressed NPZ format
- Processing: GPU-accelerated extraction

## Current Progress

### Initial Data Collection Complete ✓
- Extracted activations from 30,000 samples
- Model: Gemma-2-2b Layer 12
- Sequence Length: 128 tokens
- Total Tokens Processed: ~3.84M tokens (30k × 128)
- Storage Format: NPZ compressed chunks

### Preliminary Statistics
- Activation Shape: [batch_size, seq_length, 2304]
- Data Collection Runtime: Completed successfully
- Storage Location: /workspace/data/activations/chunks/
- Monitoring: Batch-level statistics (mean, std) collected

## Documentation Requirements

- Experimental setup details
- Performance comparisons
- Implementation guides
- Analysis results and insights
- Scaling considerations and limitations