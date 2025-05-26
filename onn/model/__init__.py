# ============================================================================
# onn/model/__init__.py
# ============================================================================

"""Neural network model implementations.

This module contains hybrid optical-electronic neural network architectures
for various classification tasks.
"""

from .optical_mnist import OpticalMNISTClassifier
from .optical_cancer import OpticalCancerClassifier
from .utils import create_optical_layers, create_reduction_layers

__all__ = [
    "OpticalMNISTClassifier",
    "OpticalCancerClassifier",
    "create_optical_layers",
    "create_reduction_layers",
]
