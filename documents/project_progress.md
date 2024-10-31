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

### Phase 1: Initial Setup & Sampling

1. [ ] Set up environment and dependencies
2. [ ] Initialize Gemma-2-2b with layer 12 activation hooks
3. [ ] Implement sampling pipeline
    - Stratified random sampling across documents
    - Preserve 512 token context windows
    - Maintain source distribution
4. [ ] Create activation storage architecture

### Phase 2: Storage Architecture

1. **Storage Format**
   - Primary: Memory-mapped arrays using PyArrow
   - Metadata: SQLite database for token/sequence tracking
   - Directory structure:
   ```
   activations/
   ├── chunks/
   │   ├── chunk_0000.arrow
   │   ├── chunk_0001.arrow
   │   └── ...
   ├── metadata.sqlite
   └── index.json
   ```

2. **Implementation Stages**
   - [ ] Basic Arrow/NumPy storage implementation
   - [ ] Metadata tracking system
   - [ ] Compression support
   - [ ] Parallel processing
   - [ ] Caching layer
   - [ ] Performance monitoring

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

## Next Immediate Steps

1. [ ] Set up development environment
2. [ ] Implement dataset sampling pipeline
3. [ ] Create initial storage architecture
4. [ ] Build activation extraction script

## Documentation Requirements

- Experimental setup details
- Performance comparisons
- Implementation guides
- Analysis results and insights
- Scaling considerations and limitations