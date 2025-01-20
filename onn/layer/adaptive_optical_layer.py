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


class AdaptiveOpticalLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_max_inputs: int = 16,
        interferometer: Interferometer = None,
        aux_loss_weight: float = 0.1,
    ):
        """
        Initializes the AdaptiveOpticalLayer with energy concentration instead of masking.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device_max_inputs (int): Maximum number of optical modes.
            interferometer (Interferometer, optional): Pre-configured interferometer.
            aux_loss_weight (float): Weight for auxiliary loss (default: 0.1).
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device_max_inputs = device_max_inputs
        self.aux_loss_weight = aux_loss_weight

        if self.device_max_inputs < max(in_features, out_features):
            raise ValueError(
                f"Number of modes ({self.device_max_inputs}) must be at least max(in_features, out_features)"
            )

        # Initialize dimension weights for adaptive reduction
        self.dimension_weights = nn.Parameter(
            torch.ones(device_max_inputs, dtype=torch.float32)
        )

        # Initialize interferometer parameters
        self._init_interferometer(interferometer)

        # Register parameter normalization hook
        self.register_forward_hook(self._normalize_params_hook)

    def _init_interferometer(self, interferometer):
        """Initialize interferometer parameters."""
        if interferometer is None:
            # Create structured unitary that concentrates energy in desired dimensions
            U = random_unitary(self.device_max_inputs)  # Using your existing function
            interferometer = square_decomposition(U)

        self.itf = interferometer

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

        for bs_orig, t, p in zip(self.itf.BS_list, theta, phi):
            current_interferometer.add_BS(
                Beamsplitter(bs_orig.mode1, bs_orig.mode2, t, p)
            )

        return current_interferometer

    def forward(self, x):
        """
        Forward pass maintaining the same interface as OpticalLinearLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Normalize parameters
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

        # Apply adaptive weights (replaces masking)
        weighted_transform = transformation * torch.sigmoid(
            self.dimension_weights
        ).unsqueeze(1)

        # Handle input padding if needed
        if x.shape[1] < self.device_max_inputs:
            x = torch.nn.functional.pad(x, (0, self.device_max_inputs - x.shape[1]))

        # Apply transformation
        output = x @ weighted_transform.T

        # Return only the needed outputs (maintaining original interface)
        return (
            output[:, : self.out_features]
            if self.out_features < self.device_max_inputs
            else output
        )

    def extra_repr(self):
        """Return extra representation string."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"device_max_inputs={self.device_max_inputs}"
        )
