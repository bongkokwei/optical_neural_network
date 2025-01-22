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


class OpticalLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device_max_inputs: int = 16,
        interferometer: Interferometer = None,
        trainable_params: dict = {},
        to_normalise=False,
    ):
        """
        Initializes the OpticalLinearLayer with support for fixed and trainable parameters.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device_max_inputs (int): Maximum number of optical modes supported by the device.
            interferometer (Interferometer, optional): Pre-configured interferometer.
            trainable_params (dict, optional): Dictionary specifying which parameters to train.
                Format: {
                    'theta': list[bool],  # True for trainable theta params
                    'phi': list[bool]     # True for trainable phi params
                }
                If None, all parameters are trainable.
            to_normalise (bool): Whether to normalize parameters after each forward pass.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device_max_inputs = device_max_inputs

        if self.device_max_inputs < max(in_features, out_features):
            raise ValueError(
                f"Number of modes ({self.device_max_inputs}) must be at least max(in_features, out_features)"
            )

        self._init_interferometer(interferometer, trainable_params)

        if to_normalise:
            self.register_forward_hook(self._normalize_params_hook)

    def _init_interferometer(self, interferometer, trainable_params):
        """Initialize interferometer with trainable and fixed parameters."""
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

        # Extract initial parameters
        theta_init = torch.tensor(
            [bs.theta for bs in self.itf.BS_list], dtype=torch.float32
        )
        phi_init = torch.tensor(
            [bs.phi for bs in self.itf.BS_list], dtype=torch.float32
        )

        # All parameters are trainable by default
        self.theta = nn.Parameter(theta_init)
        self.phi = nn.Parameter(phi_init)

        trainable_params = {
            "theta": [bs.trainable for bs in self.itf.BS_list],
            "phi": [bs.trainable for bs in self.itf.BS_list],
        }

        # Handle trainable parameters
        if trainable_params is not None:
            self._set_parameter_trainability(trainable_params)

    @torch.no_grad()
    def _get_full_parameters(self):
        """Reconstruct full parameter tensors from trainable and fixed parts."""
        if hasattr(self, "theta_mask"):
            # Initialize full parameter tensors
            theta = torch.zeros_like(self.theta_mask, dtype=torch.float32)
            phi = torch.zeros_like(self.phi_mask, dtype=torch.float32)

            # Fill in trainable parameters
            theta[self.theta_mask] = self.theta_trainable
            phi[self.phi_mask] = self.phi_trainable

            # Fill in fixed parameters
            theta[~self.theta_mask] = self.theta_fixed
            phi[~self.phi_mask] = self.phi_fixed
        else:
            # All parameters are trainable
            theta = self.theta
            phi = self.phi

        return theta, phi

    @torch.no_grad()
    def _set_parameter_trainability(self, trainable_params):
        # Create masks for trainable parameters
        theta_mask = torch.tensor(
            trainable_params.get("theta", [True] * len(self.theta)),
            dtype=torch.bool,
        )
        phi_mask = torch.tensor(
            trainable_params.get("phi", [True] * len(self.phi)), dtype=torch.bool
        )

        # Initialize trainable parameters
        self.theta_trainable = nn.Parameter(self.theta[theta_mask])
        self.phi_trainable = nn.Parameter(self.phi[phi_mask])

        # Initialize fixed parameters as buffers
        self.register_buffer("theta_fixed", self.theta[~theta_mask])
        self.register_buffer("phi_fixed", self.phi[~phi_mask])

        # Store masks for reconstruction
        self.register_buffer("theta_mask", theta_mask)
        self.register_buffer("phi_mask", phi_mask)

    @torch.no_grad()
    def _normalize_params_hook(self, module, input, output):
        """Hook to normalize parameters after each forward pass."""
        theta, phi = self._get_full_parameters()
        normalized_theta, normalized_phi = normalize_optical_params(theta, phi)

        if hasattr(self, "theta_mask"):
            # Update only trainable parameters
            self.theta_trainable.copy_(normalized_theta[self.theta_mask])
            self.phi_trainable.copy_(normalized_phi[self.phi_mask])
        else:
            # Update all parameters
            self.theta.copy_(normalized_theta)
            self.phi.copy_(normalized_phi)

    @torch.no_grad()
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
        """Forward pass using combined trainable and fixed parameters."""
        # Get full parameter sets
        theta, phi = self._get_full_parameters()

        # Normalize parameters
        current_theta, current_phi = normalize_optical_params(theta, phi)
        theta_np = current_theta.detach().cpu().numpy()
        phi_np = current_phi.detach().cpu().numpy()

        # Get transformation matrix
        current_interferometer = self._get_current_interferometer(theta_np, phi_np)
        transformation = torch.tensor(
            current_interferometer.calculate_transformation(),
            dtype=torch.complex64,
            device=x.device,
        )

        # Handle input padding if needed
        if x.shape[1] < self.device_max_inputs:
            x = torch.nn.functional.pad(x, (0, self.device_max_inputs - x.shape[1]))

        # Apply transformation and truncate if needed
        output = torch.abs(x.to(torch.complex64) @ transformation.T)
        return (
            output[:, : self.out_features]
            if self.out_features < self.device_max_inputs
            else output
        )


# Example usage:
if __name__ == "__main__":

    max_device_input = 16
    out_features = 8
    U_subset = random_unitary(out_features)  # only uses a portion of input/output
    U_total = construct_complementary_matrix(
        U_subset,
        total_size=max_device_input,
    )  # complementary unitary, to ensure U_total is unitary

    verify_unitarity(U_total)

    from ..optics.interferometer import fidelity

    itf = square_decomposition(U_total)
    U_recon = itf.calculate_transformation()
    print(
        f"Fidelity between U_total and U_recon: {np.abs(fidelity(U_total, U_recon)):.6f}"
    )
    itf.draw()
