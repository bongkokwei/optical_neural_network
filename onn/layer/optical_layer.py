import torch
import torch.nn as nn
import numpy as np
from ..optics.interferometer import (
    Interferometer,
    square_decomposition,
    random_unitary,
    Beamsplitter,
)


def normalize_optical_params(
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize theta and phi parameters to their valid ranges for optical interferometers.

    Args:
        theta: Beamsplitter angle tensor
        phi: Phase tensor

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized (theta, phi) within valid ranges:
            - theta in [0, π/2]
            - phi in [0, 2π]
    """
    # Create a mask for theta values > π/2
    theta = theta % np.pi  # First get it into [0, π]
    mask = theta > (np.pi / 2)

    # Where theta > π/2, use complementary angle and adjust phi
    theta = torch.where(mask, np.pi - theta, theta)
    phi = torch.where(mask, phi + np.pi, phi)

    # Normalize phi to [0, 2π]
    phi = phi % (2 * np.pi)

    return theta, phi


class OpticalLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_modes: int = None,
        interferometer: Interferometer = None,
        additional_blocked_modes: list[int] = None,
    ):
        """
        Initializes the OpticalLinearLayer with automatic mode blocking based on dimensions.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            num_modes (int, optional): Number of optical modes.
                                     Must be >= max(in_features, out_features).
            interferometer (Interferometer, optional): Pre-configured interferometer.
                                                     If None, a random unitary will be used.
            additional_blocked_modes (list[int], optional): Additional mode indices to block.
        """
        super().__init__()

        # Set and validate basic parameters
        self.in_features = in_features
        self.out_features = out_features
        self.num_modes = num_modes or max(in_features, out_features)

        if self.num_modes < max(in_features, out_features):
            raise ValueError(
                f"Number of modes ({self.num_modes}) must be at least max(in_features, out_features)"
            )

        # Process and validate blocked modes
        self.additional_blocked_modes = set(additional_blocked_modes or [])
        if invalid_modes := {
            m for m in self.additional_blocked_modes if not 0 <= m < self.num_modes
        }:
            raise ValueError(
                f"Blocked modes {invalid_modes} are out of range [0, {self.num_modes-1}]"
            )

        # Create masks as buffers (moved to a separate method for clarity)
        self._create_masks()

        # Initialize interferometer parameters
        self._init_interferometer(interferometer)

        # Register parameter normalization hook
        self.register_forward_hook(self._normalize_params_hook)

    def _create_masks(self):
        """Create input and output masks as buffers."""
        input_mask = torch.zeros(self.num_modes, dtype=torch.float32)
        input_mask[: self.in_features] = 1.0
        self.register_buffer("_input_mask", input_mask)

        output_mask = torch.zeros(self.num_modes, dtype=torch.float32)
        output_mask[: self.out_features] = 1.0
        output_mask[list(self.additional_blocked_modes)] = 0.0
        self.register_buffer("_output_mask", output_mask)

    def _init_interferometer(self, interferometer):
        """Initialize interferometer parameters."""
        if interferometer is None:
            U = random_unitary(self.num_modes)
            interferometer = square_decomposition(U)

        self.itf = interferometer

        # Extract and normalize parameters
        theta_params = [bs.theta for bs in self.itf.BS_list]
        phi_params = [bs.phi for bs in self.itf.BS_list]

        theta_init = torch.tensor(theta_params, dtype=torch.float32)
        phi_init = torch.tensor(phi_params, dtype=torch.float32)
        theta_init, phi_init = normalize_optical_params(theta_init, phi_init)

        self.theta = nn.Parameter(theta_init)
        self.phi = nn.Parameter(phi_init)

    @torch.no_grad()
    def _normalize_params_hook(self, module, input, output):
        """Hook to normalize parameters after each forward pass."""
        normalized_theta, normalized_phi = normalize_optical_params(
            self.theta, self.phi
        )
        self.theta.copy_(normalized_theta)
        self.phi.copy_(normalized_phi)

    def _get_current_interferometer(self, theta, phi):
        """Create interferometer with current parameters."""
        current_interferometer = Interferometer()

        # Add beamsplitters with current parameters
        for bs_orig, t, p in zip(self.itf.BS_list, theta, phi):
            current_interferometer.add_BS(
                Beamsplitter(bs_orig.mode1, bs_orig.mode2, t, p)
            )

        # Add original output phases
        for i, phase in enumerate(self.itf.output_phases):
            current_interferometer.add_phase(i + 1, phase)

        return current_interferometer

    def forward(self, x):
        """
        Forward pass of the OpticalLinearLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, out_features).
        """
        # Normalize and prepare parameters
        current_theta, current_phi = normalize_optical_params(self.theta, self.phi)
        theta_np = current_theta.detach().cpu().numpy()
        phi_np = current_phi.detach().cpu().numpy()

        # Get transformation matrix
        current_interferometer = self._get_current_interferometer(theta_np, phi_np)
        transformation = torch.tensor(
            current_interferometer.calculate_transformation().real,
            dtype=torch.float32,
            device=x.device,
        )

        # Apply masks
        transformation = (
            transformation
            * self._output_mask.unsqueeze(1)
            * self._input_mask.unsqueeze(0)
        )

        # Handle input padding if needed
        if x.shape[1] < self.num_modes:
            x = torch.nn.functional.pad(x, (0, self.num_modes - x.shape[1]))

        # Apply transformation and truncate if needed
        output = x @ transformation.T
        return (
            output[:, : self.out_features]
            if self.out_features < self.num_modes
            else output
        )

    def extra_repr(self):
        """Return extra representation string."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_modes={self.num_modes}, "
            f"additional_blocked_modes={list(self.additional_blocked_modes)}"
        )


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

            optical_layer = OpticalLinearLayer(
                in_features=segment_in_size,
                out_features=self.segment_out_features,
                num_modes=segment_modes,  # might cahnge to NONE
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
