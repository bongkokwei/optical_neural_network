# ============================================================================
# onn/layout/__init__.py
# ============================================================================

"""Photonic layout generation and GDS tools.

This module provides functions for generating GDS layouts of photonic
circuits, particularly interferometer meshes and optical processing units.
"""

from .create_gds_nxn import (
    create_mesh_interferometer,
    match_beamsplitters,
    get_bottom_bs_indices,
    sort_ports,
)

__all__ = [
    "create_mesh_interferometer",
    "match_beamsplitters",
    "get_bottom_bs_indices",
    "sort_ports",
]
