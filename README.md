# Activation Clustering for Neural Networks

A Python library for analyzing and clustering neural network activations, with a focus on comparing clustering-based approaches with Sparse Autoencoders (SAEs) for model interpretability.

## Installation

### Option 1: Install from GitHub (recommended)
```bash
pip install git+https://github.com/yourusername/activation-clustering.git
```

### Option 2: Install for Development
```bash
# Clone the repository
git clone https://github.com/yourusername/activation-clustering.git
cd activation-clustering

# Install with Poetry
poetry install
```

## Quick Start

```python
from activation_clustering import ActivationExtractor, ExtractionConfig, SequentialSoftClustering

# Extract and whiten activations
config = ExtractionConfig(model_name="google/gemma-2b")
extractor = ActivationExtractor(config)
whitened_activations = extractor.process_and_store_whitened()

# Perform soft clustering
clustering = SequentialSoftClustering(n_clusters=256, d_model=2304)
clustering.fit(whitened_activations)
```

## Requirements

- Python 3.8+
- PyTorch 2.1.0+
- Transformers
- NumPy
- Poetry (for development)

## License

MIT License
