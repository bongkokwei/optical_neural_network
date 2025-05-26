# ============================================================================
# onn/optics/__init__.py
# ============================================================================

"""Optical simulation and modeling components.

This module provides tools for simulating optical interferometers,
quantum optical states, and photonic device behavior including:
- Interferometer decomposition and simulation
- Fock state evolution
- Nonlinear optical effects
- Material properties and dispersion
"""

from .interferometer import (
    Interferometer,
    Beamsplitter,
    square_decomposition,
)
from .utils import random_unitary, fidelity
from .fock_state_simulation import FockStateInterferometer
from .sellmeier import sellmeier_mgln
from .output_fields import shg_out

# Optional imports (may have additional dependencies)
try:
    from .pulse_gen import generate_frequency_pulse, gen_pulse_freq

    _HAS_PULSE_GEN = True
except ImportError:
    _HAS_PULSE_GEN = False

try:
    from .gvm import calculate_gvm, n_eff_tfln

    _HAS_GVM = True
except ImportError:
    _HAS_GVM = False

__all__ = [
    # Core components
    "Interferometer",
    "Beamsplitter",
    "square_decomposition",
    "random_unitary",
    "fidelity",
    # Quantum simulation
    "FockStateInterferometer",
    # Material properties
    "sellmeier_mgln",
    "shg_out",
]

# Add optional components to __all__ if available
if _HAS_PULSE_GEN:
    __all__.extend(["generate_frequency_pulse", "gen_pulse_freq"])

if _HAS_GVM:
    __all__.extend(["calculate_gvm", "n_eff_tfln"])
