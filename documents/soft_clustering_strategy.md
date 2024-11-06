# Soft Clustering Strategy for Large-Scale Sequential Activations

## Challenge Overview
We need to perform soft clustering on whitened activations that are too large to fit in memory at once. The key requirements are:
1. Process data in chunks while maintaining clustering quality
2. Store results in a way that enables easy visualization and interpretation
3. Preserve contextual information for human analysis of cluster meanings

## Data Characteristics
- Total samples: 300k sequences
- Sequence length: 128 tokens
- Feature dimension: 2304 (whitened)
- Storage format: NPZ chunks containing:
  - Whitened activations
  - Attention masks
  - Token IDs

## Clustering Approach

### 1. Online Centroid Learning
- Initialize cluster centers randomly on unit hypersphere
- Process data in chunks:
  ```python
  for chunk in activation_chunks:
      # Sample positions to manage memory
      sampled_activations = sample_positions(chunk, max_points=100000)
      
      # Update centroids using current batch
      memberships = compute_memberships(sampled_activations)
      update_centroids(sampled_activations, memberships)
  ```
- Maintain running statistics for convergence monitoring
- Save intermediate centroids for potential restarts

### 2. Membership Computation and Storage
Process each chunk separately after centroid convergence:
```python
{
    "chunk_0": {
        # Sparse format for efficiency
        "indices": array[n_nonzero, 3],  # (sequence_idx, position, cluster_idx)
        "values": array[n_nonzero],      # Membership strengths
        "tokens": array[n_nonzero],      # Corresponding token IDs
        "metadata": {
            "total_sequences": int,
            "sequence_length": int
        }
    },
    ...
}
```

### 3. Cluster Analysis Data Structure
For each cluster, maintain:
```python
{
    "cluster_id": {
        "statistics": {
            "total_usage": float,          # Sum of all memberships
            "position_distribution": array, # Usage by sequence position
            "token_distribution": dict,     # Most common tokens
        },
        "vectors": {
            "center": array,               # Original cluster center
            "intervention": array,         # LDA-derived intervention direction
            "intervention_stats": {
                "separation_strength": float,  # Measure of cluster separation
                "within_cluster_variance": float,
                "cross_cluster_correlation": array  # Correlation with other clusters
            }
        },
        "exemplars": {
            "top_k": [                     # Best examples for visualization
                {
                    "chunk_id": int,
                    "sequence_idx": int,
                    "position": int,
                    "membership": float,
                    "context_tokens": array,# Window of surrounding tokens
                    "context_memberships": array # Other active clusters
                },
                ...
            ],
            "random_k": [...],             # Random examples for diversity
        }
    }
}
```

### 4. Intervention Vector Computation
- For each cluster, compute a secondary direction optimized for intervention:
  ```python
  # Weighted LDA-style computation
  - Positive examples: Activations with high membership in target cluster
  - Negative examples: Activations with high membership in other clusters
  - Weight samples by membership strengths
  - Use robust covariance estimation (Ledoit-Wolf)
  - Normalize and align with cluster center
  ```
- Store both representations:
  - Cluster centers for analysis/visualization
  - Intervention vectors for causal editing
- Track separation metrics to evaluate intervention direction quality

## Implementation Phases

### Phase 1: Centroid Learning
1. Initialize SequentialSoftClustering model
2. Stream through chunks for centroid updates
3. Save converged centroids

### Phase 2: Membership Computation
1. Load converged centroids
2. Process each chunk:
   - Compute sparse memberships
   - Store in efficient format
   - Update cluster statistics

### Phase 3: Exemplar Selection
1. Track top examples per cluster during membership computation
2. Store contextual information for selected examples
3. Include diverse random samples

### Phase 4: Intervention Vector Computation
1. For each cluster:
   - Select high-membership activations
   - Compute weighted statistics
   - Generate intervention direction
2. Validate separation quality:
   - Measure cluster separation
   - Compute cross-cluster effects
3. Store both representations:
   - Original centers for analysis
   - Intervention vectors for editing

## Storage Strategy

### 1. Chunk-Level Storage
- One NPZ file per chunk containing sparse memberships
- Include sequence metadata and token information
- Enables partial loading for analysis

### 2. Cluster Summary Storage
- JSON format for cluster statistics and exemplars
- Links to specific chunks for detailed analysis

## Memory Management

### 1. Chunk Processing
- Fixed memory budget for processing (e.g., 8GB)
- Adaptive batch sizes based on available memory
- Stream results to disk immediately

### 2. Exemplar Selection
- Maintain fixed-size priority queues per cluster
- Update incrementally during processing
- Store only necessary context

## Usage Workflow

1. **Training Phase**
```python
clustering = SequentialSoftClustering(n_clusters=k)
clustering.fit_chunks(chunk_paths)
```

2. **Analysis Phase**
```python
analyzer = ClusterAnalyzer(clustering_results_path)
analyzer.get_cluster_summary(cluster_id)
analyzer.visualize_cluster_context(cluster_id, example_idx)
```

## Future Extensions

1. **Hierarchical Analysis**
- Nested clustering for large numbers of clusters
- Multiple granularity levels for exploration

2. **Dynamic Visualization**
- Real-time computation of additional statistics
- Custom context window sizes
- Token pattern analysis

3. **Pattern Discovery**
- Automatic sequence pattern detection
- Cross-cluster relationship analysis
- Temporal dynamics study 