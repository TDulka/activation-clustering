"""Activation Clustering for Neural Networks."""

from .extract_activations import ActivationExtractor, ExtractionConfig
from .whitening import ActivationWhitening
from .soft_clustering import SequentialSoftClustering

__all__ = [
    "ActivationExtractor",
    "ExtractionConfig",
    "ActivationWhitening",
    "SequentialSoftClustering",
]

__version__ = "0.1.0"
