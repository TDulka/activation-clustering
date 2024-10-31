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
- Comparison SAE Sizes: Multiple variants available

### Dataset
- Source: monology/pile-uncopyrighted
- Type: Text corpus
- Usage: Activation extraction and analysis

## Implementation Plan

### Phase 1: Initial Setup

1. [ ] Set up environment and dependencies
2. [ ] Initialize Gemma-2-2b with layer 12 activation hooks
3. [ ] Create data loading pipeline for pile-uncopyrighted
4. [ ] Implement activation extraction and storage

### Phase 2: Clustering Implementation

1. [ ] Implement basic k-means clustering
2. [ ] Add density modeling
3. [ ] Create feature interpretation pipeline
4. [ ] Set up evaluation metrics

### Phase 3: Analysis

1. [ ] Compare with existing SAE results
2. [ ] Evaluate computational efficiency
3. [ ] Document findings and insights

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

## Next Immediate Steps

1. [ ] Set up development environment
2. [ ] Download and prepare pile-uncopyrighted dataset
3. [ ] Create activation extraction script
4. [ ] Implement basic clustering pipeline

## Success Metrics

- Clustering quality vs SAE reconstruction quality
- Computational resource usage comparison
- Feature interpretation clarity
- Implementation simplicity and maintainability

## Documentation Requirements

- Experimental setup details
- Performance comparisons
- Implementation guides
- Analysis results and insights