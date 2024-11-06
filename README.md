# Activation Clustering for Neural Networks

A Python library for analyzing and clustering neural network activations, with a focus on comparing clustering-based approaches with Sparse Autoencoders (SAEs) for model interpretability.

## Overview

This project implements efficient activation extraction and clustering techniques for large language models, specifically tested with Gemma-2-2b. It provides tools for:

- Extracting and whitening neural network activations
- Implementing soft clustering algorithms for sequential data
- Comparing clustering approaches with SAE-based methods
- Supporting interpretability research

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/activation-clustering.git
cd activation-clustering

# Install dependencies using Poetry
poetry install
```

## Quick Start

```python
from src.extract_activations import ActivationExtractor
from src.soft_clustering import SequentialSoftClustering

# Extract and whiten activations
config = ExtractionConfig(model_name="google/gemma-2-2b")
extractor = ActivationExtractor(config)
whitened_activations = extractor.process_and_store_whitened()

# Perform soft clustering
clustering = SequentialSoftClustering(n_clusters=256, d_model=2304)
clustering.fit(whitened_activations)
```

## Features

- Memory-efficient activation extraction
- Soft clustering with temperature scaling

## Requirements

- Python 3.8+
- PyTorch 2.1.0+
- Transformers
- NumPy
- Poetry for dependency management

## License

MIT License
