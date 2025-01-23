import numpy as np
import math
from typing import Callable, Optional
import torch.nn as nn

from ..layer.optical_activation_layer import (
    OpticalNonlinearLayer,
)


def create_reduction_layers(
    input_size: int,
    target_size: int,
    num_steps: int = None,
) -> nn.Sequential:
    """
    Creates a sequential model that reduces dimensionality from input_size to target_size.
    If num_steps is provided, uses that many reduction steps. Otherwise, calculates optimal
    number of steps through halving.

    Args:
        input_size: Initial input dimension
        target_size: Desired output dimension
        num_steps: Optional; Number of reduction steps to use. If None, calculates automatically.

    Returns:
        nn.Sequential: Sequential model containing the reduction layers

    Raises:
        ValueError: If target_size is larger than input_size or if num_steps is invalid
    """
    if target_size > input_size:
        raise ValueError(
            f"Target size ({target_size}) must be smaller than input size ({input_size})"
        )

    # If sizes are equal, return identity model
    if target_size == input_size:
        return nn.Sequential(nn.Identity())

    # If num_steps not provided, calculate optimal number through halving
    if num_steps is None:
        num_steps = math.ceil(math.log2(input_size / target_size))
    elif num_steps <= 0:
        raise ValueError("num_steps must be positive")

    layers = []
    current_size = input_size

    # Calculate sizes for each step
    for i in range(num_steps):
        if i == num_steps - 1:
            # Last layer always goes to target_size
            next_size = target_size
        else:
            # Calculate intermediate size
            size_reduction = (current_size - target_size) / (num_steps - i)
            next_size = int(current_size - size_reduction)
            next_size = max(next_size, target_size)  # Don't go below target_size

        layers.extend([nn.Linear(current_size, next_size), nn.ReLU()])

        current_size = next_size

    return nn.Sequential(*layers)


def create_optical_layers(
    num_layers: int,
    initial_size: int,
    final_size: int,
    device_max_inputs: int = None,
    optical_layer: Optional[Callable[[int, int], nn.Module]] = None,
    dropout_rate=0.2,
) -> nn.Sequential:
    """
    Creates a Sequential module of optical layers with gradually reducing size,
    each followed by activation except for the final layer.

    Args:
        initial_size (int): Size of the first layer
        final_size (int): Size of the final layer output

    Returns:
        nn.Sequential: Sequential module containing optical layers and ReLU
    """
    layers = []

    # Calculate size reduction per layer
    sizes = np.linspace(initial_size, final_size, num_layers + 1, dtype=int)

    for i in range(num_layers - 1):
        current_size = sizes[i]
        next_size = sizes[i + 1]

        # Add optical layer
        layers.append(
            optical_layer(
                in_features=current_size,
                out_features=next_size,
                device_max_inputs=device_max_inputs,
            ),
        )

        # Add activation functions
        layers.append(
            OpticalNonlinearLayer(
                nonlinearity_type="SHG",
                chi=1.3e-22,  # d_33 of KTP crystal
                phase_matching=False,
                phase_mismatch=0.0,  # Assume perfectly phase-matched
            ),
        )
        layers.append(nn.Dropout(dropout_rate))

    # Final optical layer without nonlinearity
    layers.append(
        optical_layer(
            in_features=sizes[-2],
            out_features=final_size,
            device_max_inputs=device_max_inputs,
        ),
    )

    return nn.Sequential(*layers)
