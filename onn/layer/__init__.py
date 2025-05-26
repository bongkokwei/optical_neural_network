# ============================================================================
# onn/layer/__init__.py
# ============================================================================

"""Optical computing layer implementations.

This module provides various optical processing layers including:
- Linear optical transformations using interferometers
- Adaptive optical layers with trainable parameters
- Segmented processing for large-scale optical devices
- Optical nonlinear activation functions
"""

from .optical_layer import OpticalLinearLayer
from .adaptive_optical_layer import AdaptiveOpticalLayer
from .segmented_optical_layer import SegmentedOpticalLayer
from .optical_activation_layer import OpticalNonlinearLayer

__all__ = [
    "OpticalLinearLayer",
    "AdaptiveOpticalLayer",
    "SegmentedOpticalLayer",
    "OpticalNonlinearLayer",
]
