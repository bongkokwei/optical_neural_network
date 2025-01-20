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


def construct_complementary_matrix(U_sub, total_size=32):
    # Calculate sizes: m is size of U_sub, n is size needed for complementary matrix
    m = U_sub.shape[0]
    n = total_size - m

    # Initialize with random complex matrix
    # A = X + iY where X,Y are real random matrices
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # First QR decomposition to get initial orthogonal matrix Q
    # Q is unitary, R is upper triangular
    Q, R = np.linalg.qr(A)

    # Iterative refinement to improve orthogonality
    for _ in range(3):
        # Each iteration of QR makes Q more unitary
        Q, R = np.linalg.qr(Q)

        # Ensure each column has unit norm
        for j in range(n):
            Q[:, j] = Q[:, j] / np.linalg.norm(Q[:, j])

    # Phase correction step:
    # 1. Calculate phase of diagonal elements
    # 2. Create diagonal matrix D = exp(-i*phase)
    # 3. Multiply Q by D to ensure proper phase alignment
    D = np.diag(np.exp(-1j * np.angle(np.diag(Q))))
    U_comp = Q @ D

    return U_comp


def verify_unitarity(U_comp):
    # Check U_comp * U_comp† = I
    prod = np.dot(U_comp, U_comp.conj().T)
    error = np.max(np.abs(prod - np.eye(U_comp.shape[0])))
    print(f"Maximum unitarity error: {error}")
    return error < 1e-10


class OpticalLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_max_inputs: int = 16,
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
        self.device_max_inputs = device_max_inputs or max(in_features, out_features)

        if self.device_max_inputs < max(in_features, out_features):
            raise ValueError(
                f"Number of modes ({self.device_max_inputs}) must be at least max(in_features, out_features)"
            )

        # Process and validate blocked modes
        self.additional_blocked_modes = set(additional_blocked_modes or [])
        if invalid_modes := {
            m
            for m in self.additional_blocked_modes
            if not 0 <= m < self.device_max_inputs
        }:
            raise ValueError(
                f"Blocked modes {invalid_modes} are out of range [0, {self.device_max_inputs-1}]"
            )

        # Create masks as buffers (moved to a separate method for clarity)
        self._create_masks()

        # Initialize interferometer parameters
        self._init_interferometer(interferometer)

        # Register parameter normalization hook
        self.register_forward_hook(self._normalize_params_hook)

    def _create_masks(self):
        """Create input and output masks as buffers."""
        input_mask = torch.zeros(self.device_max_inputs, dtype=torch.float32)
        input_mask[: self.in_features] = 1.0
        self.register_buffer("_input_mask", input_mask)

        output_mask = torch.zeros(self.device_max_inputs, dtype=torch.float32)
        output_mask[: self.out_features] = 1.0
        output_mask[list(self.additional_blocked_modes)] = 0.0
        self.register_buffer("_output_mask", output_mask)

    def _init_interferometer(self, interferometer):
        """Initialize interferometer parameters."""
        if interferometer is None:
            U = random_unitary(self.device_max_inputs)
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
        if x.shape[1] < self.device_max_inputs:
            x = torch.nn.functional.pad(x, (0, self.device_max_inputs - x.shape[1]))

        # Apply transformation and truncate if needed
        output = x @ transformation.T
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
            f"device_max_inputs={self.device_max_inputs}, "
            f"additional_blocked_modes={list(self.additional_blocked_modes)}"
        )


# Example usage:
if __name__ == "__main__":

    max_device_input = 32
    out_features = 8
    U_subset = random_unitary(out_features)  # only uses a portion of input/output
    U_comp = construct_complementary_matrix(
        U_subset,
        total_size=max_device_input,
    )  # complementary unitary, to ensure U_total is unitary

    verify_unitarity(U_comp)

    U_total = np.zeros(
        (max_device_input, max_device_input),
        dtype=complex,
    )

    U_total[:out_features, :out_features] = U_subset
    U_total[out_features:, out_features:] = U_comp

    verify_unitarity(U_total)
