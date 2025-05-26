# ============================================================================
# onn/__init__.py (Main package __init__.py)
# ============================================================================

"""Optical Neural Network (ONN) Package

A hybrid optical-electronic neural network implementation for classification tasks
using simulated optical interferometers and photonic computing elements.

Features:
- Optical linear layers with interferometer simulation
- Optical nonlinear activation functions (SHG, FWM)
- Adaptive and segmented optical processing
- MNIST and cancer classification models
- GDS layout generation for photonic circuits
- Fock state quantum optical simulation
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main model classes
from .model import (
    OpticalMNISTClassifier,
    OpticalCancerClassifier,
)

# Import layer implementations
from .layer import (
    OpticalLinearLayer,
    AdaptiveOpticalLayer,
    SegmentedOpticalLayer,
    OpticalNonlinearLayer,
)

# Import optics simulation components
from .optics import (
    Interferometer,
    Beamsplitter,
    square_decomposition,
    random_unitary,
    FockStateInterferometer,
)

# Import layout generation
from .layout import create_mesh_interferometer

# Define what gets imported with "from onn import *"
__all__ = [
    # Models
    "OpticalMNISTClassifier",
    "OpticalCancerClassifier",
    # Layers
    "OpticalLinearLayer",
    "AdaptiveOpticalLayer",
    "SegmentedOpticalLayer",
    "OpticalNonlinearLayer",
    # Optics
    "Interferometer",
    "Beamsplitter",
    "square_decomposition",
    "random_unitary",
    "FockStateInterferometer",
    # Layout
    "create_mesh_interferometer",
    # Metadata
    "__version__",
]
