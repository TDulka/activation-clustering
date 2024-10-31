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

### Phase 2: Preprocessing & Storage ✓

1. **Storage Format** ✓
   - Primary: NPZ compressed arrays (implemented)
   - Metadata: Basic tracking implemented
   - Directory structure maintained

2. **Implementation Stages**
   - [x] Basic NumPy storage implementation
   - [x] Metadata tracking system
   - [x] Compression support
   - [x] Whitening transformation
   - [ ] Parallel processing
   - [ ] Caching layer
   - [x] Performance monitoring

3. **Whitening Implementation** ✓
   - Position-aware whitening transform
   - Per-position statistics computation
   - Reversible transformation support
   - Validation metrics implemented
   - GPU-accelerated processing

### Phase 3: Clustering Implementation (Updated)

1. **Two-Level Clustering Approach**
   - [ ] Global Feature Space Analysis
     * Cluster all positions together (n=256) for SAE comparison
     * Learn position-agnostic features first
     * Enables direct comparison with SAE features
   
   - [ ] Position-Specific Refinement
     * Analyze how global clusters manifest at each position
     * Track cluster usage patterns across positions
     * Identify position-specific variations

2. **Implementation Strategy**
   - [ ] Global Clustering
     * Mini-batch k-means on all positions (256 clusters)
     * Density modeling for cluster distributions
     * Feature interpretation pipeline
   
   - [ ] Positional Analysis
     * Track cluster assignment frequencies per position
     * Compute position-specific cluster statistics
     * Identify position-dependent patterns

3. **Evaluation Framework**
   - [ ] Global Metrics
     * Compare with SAE features directly
     * Measure clustering quality (silhouette, etc.)
     * Assess feature interpretability
   
   - [ ] Positional Metrics
     * Position-specific cluster usage patterns
     * Transition patterns between positions
     * Position-dependent feature analysis

### Phase 4: Analysis & Scaling

1. [ ] Compare with existing SAE results
2. [ ] Evaluate computational efficiency
3. [ ] Document findings and insights
4. [ ] Plan scaling to larger sample if needed

## Key Experiments (Updated)

1. **Clustering Analysis**
   - Global clustering (256 clusters) across all positions
   - Position-specific cluster usage patterns
   - Comparison with SAE feature directions
   - Analysis of position-dependent variations

2. **Feature Interpretation**
   - Global feature space analysis
   - Position-specific feature variations
   - Cluster usage patterns across positions
   - Comparison with SAE feature interpretations

## Success Metrics

- Clustering quality vs SAE reconstruction quality
- Computational resource usage comparison
- Feature interpretation clarity
- Implementation simplicity and maintainability

## Next Immediate Steps (Updated)

1. [ ] **Clustering Implementation**
   - Implement batched k-means (n=256)
   - Set up cluster quality metrics
   - Design feature interpretation pipeline

2. [ ] **Evaluation Framework**
   - Define comparison metrics with SAEs
   - Implement interpretability measures
   - Set up performance benchmarks

3. [ ] **Scaling & Optimization**
   - Profile memory usage patterns
   - Optimize data loading pipeline
   - Implement parallel processing

## Current Progress

### Preprocessing Pipeline Complete ✓
- Implemented position-aware whitening
- Processing 30,000 samples (3.84M tokens)
- GPU-accelerated transformation
- Validation metrics in place
- Reversible transformations supported

### Technical Implementation Details
- Architecture: Position-specific whitening transforms
- Batch Processing: 32 samples per batch
- Storage Format: NPZ compressed chunks
- Compute: CUDA-accelerated operations
- Validation: Mean and covariance checks per position

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