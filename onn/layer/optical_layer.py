import torch
import torch.nn as nn
import numpy as np

from functools import lru_cache
from torch import Tensor
from typing import Optional, Tuple, Dict

from onn.optics.interferometer import (
    Interferometer,
    square_decomposition,
    random_unitary,
    Beamsplitter,
)


def normalise_optical_params(
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

    # Initialize with identity matrix
    A = np.eye(n)

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

    U_total = np.zeros(
        (total_size, total_size),
        dtype=complex,
    )

    U_total[:m, :m] = U_sub
    U_total[m:, m:] = U_comp

    return U_total


def verify_unitarity(U_comp):
    # Check U_comp * U_comp† = I
    prod = np.dot(U_comp, U_comp.conj().T)
    error = np.max(np.abs(prod - np.eye(U_comp.shape[0])))
    print(f"Maximum unitarity error: {error}")
    return error < 1e-10


""" Optimised version """


class OpticalLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_max_inputs: int = 16,
        interferometer: Optional[Interferometer] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device_max_inputs = device_max_inputs

        if self.device_max_inputs < max(in_features, out_features):
            raise ValueError(
                f"Number of modes ({self.device_max_inputs}) must be at least {(max(in_features, out_features))}"
            )

        # Initialise trainable parameter tracking
        self.trainable_params: Dict[str, Optional[torch.Tensor]] = {
            "theta": None,
            "phi": None,
        }

        # Store original states as tensors
        self.original_trainable_state: Dict[str, Optional[torch.Tensor]] = {
            "theta": None,
            "phi": None,
        }

        self._initialise_interferometer(interferometer)

        # Pre-calculate padding size
        self.padding_size = (
            self.device_max_inputs - self.in_features
            if self.in_features < self.device_max_inputs
            else 0
        )

        # Register output slice for efficiency
        self.register_buffer("output_slice", torch.arange(self.out_features))

    def _normalise_params(self, theta: Tensor, phi: Tensor) -> Tuple[Tensor, Tensor]:
        """Optimised parameter normalisation using torch operations"""
        pi = torch.pi
        theta = theta % pi
        mask = theta > (pi / 2)

        theta = torch.where(mask, pi - theta, theta)
        phi = torch.where(mask, phi + pi, phi % (2 * pi))

        return theta, phi

    def _initialise_interferometer(self, interferometer: Optional[Interferometer]):
        if interferometer is None:
            effective_size = max(self.in_features, self.out_features)
            if effective_size < self.device_max_inputs:
                U_sub = random_unitary(effective_size)
                U_total = construct_complementary_matrix(U_sub, self.device_max_inputs)
                interferometer = square_decomposition(U_total)
            else:
                U = random_unitary(self.device_max_inputs)
                interferometer = square_decomposition(U)

        self.itf = interferometer

        # Convert trainable flags to tensor for better efficiency
        self.trainable_params["theta"] = torch.tensor(
            [bs.trainable for bs in self.itf.BS_list], dtype=torch.bool
        )
        self.trainable_params["phi"] = torch.tensor(
            [bs.trainable for bs in self.itf.BS_list], dtype=torch.bool
        )

        # Store original state as tensors
        self.original_trainable_state["theta"] = self.trainable_params["theta"].clone()
        self.original_trainable_state["phi"] = self.trainable_params["phi"].clone()

        # Extract initial parameters
        theta_init = torch.tensor(
            [bs.theta for bs in self.itf.BS_list], dtype=torch.float32
        )
        phi_init = torch.tensor(
            [bs.phi for bs in self.itf.BS_list], dtype=torch.float32
        )

        # Initialise parameters
        self.theta = nn.Parameter(theta_init)
        self.phi = nn.Parameter(phi_init)

        self._update_parameter_masks()

    def _update_parameter_masks(self):
        """Optimised parameter mask updates using tensor operations"""
        self.theta_mask = self.trainable_params["theta"]
        self.phi_mask = self.trainable_params["phi"]

        theta = self.theta.data
        phi = self.phi.data

        # Update trainable parameters
        self.theta_trainable = nn.Parameter(theta[self.theta_mask])
        self.phi_trainable = nn.Parameter(phi[self.phi_mask])

        # Store fixed parameters in buffer
        self.register_buffer("theta_fixed", theta[~self.theta_mask])
        self.register_buffer("phi_fixed", phi[~self.phi_mask])

    @torch.no_grad()
    def _get_full_parameters(self) -> Tuple[Tensor, Tensor]:
        """Optimised parameter reconstruction using tensor operations"""
        if hasattr(self, "theta_mask"):
            # Pre-allocate tensors with correct size and type
            theta = torch.zeros_like(self.theta_mask, dtype=torch.float32)
            phi = torch.zeros_like(self.phi_mask, dtype=torch.float32)

            # Use masked operations for assignment
            theta[self.theta_mask] = self.theta_trainable
            phi[self.phi_mask] = self.phi_trainable
            theta[~self.theta_mask] = self.theta_fixed
            phi[~self.phi_mask] = self.phi_fixed
        else:
            theta = self.theta
            phi = self.phi

        return theta, phi

    @lru_cache(maxsize=128)
    def _get_current_interferometer(
        self, theta_key: Tuple[float, ...], phi_key: Tuple[float, ...]
    ) -> torch.Tensor:
        """Cached interferometer calculation with optimised tensor creation"""
        current_interferometer = Interferometer()

        # Batch create beamsplitters
        for bs_orig, t, p in zip(self.itf.BS_list, theta_key, phi_key):
            current_interferometer.add_BS(
                Beamsplitter(bs_orig.mode1, bs_orig.mode2, t, p)
            )

        # Batch add phases
        for i, phase in enumerate(self.itf.output_phases):
            current_interferometer.add_phase(i + 1, phase)

        return torch.tensor(
            current_interferometer.calculate_transformation(), dtype=torch.complex64
        )

    def forward(self, x: Tensor) -> Tensor:
        # Get parameters and normalize
        theta, phi = self._get_full_parameters()
        current_theta, current_phi = self._normalise_params(theta, phi)

        # Convert to tuples for caching
        theta_np = tuple(current_theta.detach().cpu().numpy())
        phi_np = tuple(current_phi.detach().cpu().numpy())

        # Get cached transformation
        transformation = self._get_current_interferometer(theta_np, phi_np).to(x.device)

        # Efficient padding if needed
        if self.padding_size > 0:
            x = torch.nn.functional.pad(x, (0, self.padding_size))

        # Compute output with type conversion
        output = torch.abs(x.to(torch.complex64) @ transformation.T)

        # Return sliced or full output
        return (
            output[:, self.output_slice]
            if self.out_features < self.device_max_inputs
            else output
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"device_max_input={self.device_max_inputs}"
        )
