import torch
import torch.nn as nn
import numpy as np
from ..optics.interferometer import Interferometer

from .adaptive_optical_layer import AdaptiveOpticalLayer
from .optical_layer import OpticalLinearLayer


class SegmentedOpticalLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_max_inputs: int = 16,
        num_modes: int = None,
        interferometer: Interferometer = None,
        additional_blocked_modes: list[int] = None,
    ):
        """
        Combines input segmentation with optical interferometer transformation.

        Args:
            in_features (int): Total number of input features
            out_features (int): Total number of output features
            device_max_inputs (int): Maximum inputs per optical device
            num_modes (int, optional): Number of modes per optical device
            interferometer (Interferometer, optional): Pre-configured interferometer
            additional_blocked_modes (list[int], optional): Additional modes to block
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device_max_inputs = device_max_inputs

        # Calculate number of segments needed (ceiling division)
        self.num_segments = (in_features + device_max_inputs - 1) // device_max_inputs

        # Calculate features per segment
        self.segment_out_features = min(
            device_max_inputs,
            (out_features + self.num_segments - 1) // self.num_segments,
        )

        # Create optical layers for each segment
        self.optical_segments = nn.ModuleList()
        for i in range(self.num_segments):
            # Calculate input size for this segment
            segment_in_size = min(
                device_max_inputs, in_features - i * device_max_inputs
            )

            # Ensure num_modes is at least as large as the larger of input/output
            segment_modes = max(segment_in_size, self.segment_out_features)
            if num_modes is not None:
                segment_modes = max(segment_modes, num_modes)

            optical_layer = AdaptiveOpticalLayer(
                in_features=segment_in_size,
                out_features=self.segment_out_features,
                num_modes=segment_modes,
                interferometer=interferometer.clone() if interferometer else None,
                additional_blocked_modes=additional_blocked_modes,
            )
            self.optical_segments.append(optical_layer)

        # Calculate total intermediate size
        total_intermediate_size = self.num_segments * self.segment_out_features

        # Add final transformation layer if needed
        if total_intermediate_size != out_features:
            final_modes = max(total_intermediate_size, out_features)
            if num_modes is not None:
                final_modes = max(final_modes, num_modes)

            self.output_transform = OpticalLinearLayer(
                in_features=total_intermediate_size,
                out_features=out_features,
                num_modes=final_modes,  # might cahnge to NONE
                interferometer=interferometer.clone() if interferometer else None,
                additional_blocked_modes=additional_blocked_modes,
            )
        else:
            self.output_transform = None

        # Register a parameter to ensure the layer has parameters
        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass splitting input into segments and processing through optical devices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.size(0)

        # Process each segment through its optical layer
        segment_outputs = []
        for i in range(self.num_segments):
            # Extract segment
            start_idx = i * self.device_max_inputs
            end_idx = min((i + 1) * self.device_max_inputs, self.in_features)
            segment = x[:, start_idx:end_idx]

            # Process through optical layer (will handle its own padding)
            segment_output = self.optical_segments[i](segment)
            segment_outputs.append(segment_output)

        # Combine segment outputs
        combined = torch.cat(segment_outputs, dim=1)

        # Apply final transform if needed
        if self.output_transform is not None:
            combined = self.output_transform(combined)

        # Add a small multiple of dummy parameter to maintain gradient flow
        combined = combined + 0 * self.dummy_parameter

        # Ensure output size matches exactly
        return combined[:, : self.out_features]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"device_max_inputs={self.device_max_inputs}, "
            f"num_segments={self.num_segments}, "
            f"segment_out_features={self.segment_out_features}"
        )
