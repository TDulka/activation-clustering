# Soft Clustering Strategy for Neural Network Activations

## Overview
This document describes our approach to clustering neural network activations using a memory-efficient, sequential processing strategy. The implementation allows us to analyze patterns in neural network behavior by grouping similar activation patterns together.

## Key Components

### ClusteringConfig
Configuration dataclass that defines clustering parameters:
```python
@dataclass
class ClusteringConfig:
    n_clusters: int          # Number of clusters to create
    d_model: int            # Dimension of activation vectors
    temperature: float      # Controls softness of clustering (default: 1.0)
    min_membership: float   # Minimum threshold for cluster membership (default: 0.01)
    device: str            # Computing device ("cuda" or "cpu")
    n_exemplars: int       # Number of examples to store per cluster
    context_window: int    # Size of context window for examples
    n_iterations: int      # Number of training iterations
```

### SequentialSoftClustering
Main class that implements the clustering algorithm. Key features:

1. **Memory-Efficient Processing**
   - Processes data in chunks to handle large datasets
   - Uses sparse representations for cluster memberships
   - Maintains fixed-size exemplar storage

2. **Soft Clustering**
   - Points can belong to multiple clusters
   - Uses cosine similarity for cluster assignment
   - Applies temperature scaling for controlling assignment softness

3. **Exemplar Management**
   - Stores best examples for each cluster
   - Maintains context information around each example
   - Uses efficient priority queue for top-k tracking

## Data Flow

### 1. Input Data Format
Data is processed in chunks, where each chunk is an NPZ file containing:
```python
{
    "activations": array[batch_size, sequence_length, d_model],
    "attention_mask": array[batch_size, sequence_length],
    "token_ids": array[batch_size, sequence_length]
}
```

### 2. Processing Pipeline
```python
# Initialize clustering
config = ClusteringConfig(n_clusters=k, d_model=dim)
clustering = SequentialSoftClustering(config)

# Process data chunks
clustering.fit_chunks(chunk_paths, output_dir)
```

### 3. Output Format
For each processed chunk, creates an NPZ file containing:
```python
{
    "indices": array[n_nonzero, 3],  # (batch, sequence, cluster) indices
    "values": array[n_nonzero],      # Membership values
    "metadata": {
        "total_sequences": int,
        "sequence_length": int
    }
}
```

## Key Algorithms

### 1. Cluster Assignment
```python
def _compute_memberships(self, points):
    # Compute cosine similarities
    similarities = torch.mm(points, self.centers.t())
    
    # Apply temperature scaling
    similarities = similarities / self.config.temperature
    
    # Convert to probabilities
    memberships = torch.softmax(similarities, dim=-1)
    
    # Filter small values
    memberships[memberships < self.config.min_membership] = 0
    memberships = memberships / memberships.sum(dim=1, keepdim=True)
    
    return memberships
```

### 2. Centroid Updates
```python
def _update_centers(self, activations, memberships):
    # Weighted average of points
    weighted_sum = torch.mm(memberships.T, activations)
    
    # Normalize to unit sphere
    self.centers = F.normalize(weighted_sum, dim=1)
```

### 3. Exemplar Collection
For each cluster, maintains top examples based on membership strength:
- Stores context window around each example
- Includes token IDs and membership patterns
- Uses min-heap for efficient top-k tracking

## Usage Examples

### Basic Usage
```python
# Initialize clustering
config = ClusteringConfig(
    n_clusters=10,
    d_model=768,
    temperature=1.0,
    min_membership=0.01
)
clustering = SequentialSoftClustering(config)

# Process data
clustering.fit_chunks(chunk_paths, "output_dir")

# Save model
clustering.save("model.pt")
```

### Analysis
```python
# Load saved model
model = SequentialSoftClustering.load("model.pt")

# Analyze new data
results = model.transform(activations)
memberships = results["memberships"]

# Get cluster statistics
stats = analyze_sequential_clusters(model, activations, token_ids)
```

## Implementation Details

### Memory Management
- Uses batch processing for large datasets
- Implements sparse storage for memberships
- Maintains fixed-size exemplar storage per cluster

### Numerical Stability
- Normalizes vectors to unit length
- Uses temperature scaling for controlled softness
- Filters small membership values

### Performance Considerations
- Supports GPU acceleration
- Implements efficient matrix operations
- Uses sparse representations where beneficial

## Testing

The implementation includes comprehensive tests covering:
- Initialization and configuration
- Data processing and transformation
- Memory management
- Exemplar collection
- Model serialization
- Different input dimensions
- Edge cases and error handling

## Future Improvements

1. **Optimization**
   - Implement parallel processing for chunks
   - Add support for distributed training
   - Optimize memory usage further

2. **Analysis Tools**
   - Add visualization utilities
   - Implement cluster quality metrics
   - Add support for hierarchical clustering

3. **Features**
   - Add support for incremental updates
   - Implement cluster merging/splitting
   - Add more analysis capabilities