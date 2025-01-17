import torch
import torch.nn as nn
import numpy as np

from typing import (
    Literal,
    Optional,
    Union,
)
from enum import Enum
from scipy.constants import epsilon_0, c, pi


class OpticalNonlinearLayer(nn.Module):
    def __init__(
        self,
        nonlinearity_type: Literal["FWM", "SHG"] = "FWM",
        chi: float = 1e-12,
        phase_matching: bool = False,
        phase_mismatch: float = 0.0,
        length: float = 1.0,
        eps: float = 1e-8,
        device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        self.nonlinearity_type = nonlinearity_type
        self.chi = chi
        self.phase_matching = phase_matching
        self.phase_mismatch = phase_mismatch
        self.length = length
        self.eps = eps
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Register parameters as buffers
        self.register_buffer(
            "_chi",
            torch.tensor(chi, device=self.device),
        )
        self.register_buffer(
            "_phase_mismatch",
            torch.tensor(phase_mismatch, device=self.device),
        )
        self.register_buffer(
            "_length",
            torch.tensor(length, device=self.device),
        )

    def _minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization to the input tensor."""
        batch_min = x.min(dim=1, keepdim=True)[0]
        batch_max = x.max(dim=1, keepdim=True)[0]

        # Handle case where max = min
        diff = batch_max - batch_min
        diff = torch.where(diff == 0, torch.ones_like(diff) + self.eps, diff)

        return (x - batch_min) / diff

    def _phase_matching_factor(self, delta_k: torch.Tensor) -> torch.Tensor:
        """Calculate the phase matching factor."""
        return self._chi * torch.sinc(delta_k * self._length / 2) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing the normalized nonlinear activation."""
        # Apply input normalization
        # x = self._minmax_normalize(x)

        if self.nonlinearity_type == "FWM":
            # Third-order nonlinearity: P ∝ χ(3)E³
            output = x**3
        else:  # SHG
            # Second-order nonlinearity: P ∝ χ(2)E²
            output = x**2

        if self.phase_matching:
            phase_factor = self._phase_matching_factor(self._phase_mismatch)
            output = output * phase_factor

        # Apply output normalization
        output = self._minmax_normalize(output)

        return output

    def extra_repr(self) -> str:
        return (
            f"nonlinearity_type={self.nonlinearity_type}, "
            f"chi={self.chi}, "
            f"phase_matching={self.phase_matching}, "
            f"phase_mismatch={self.phase_mismatch}, "
            f"length={self.length}"
        )


def calculate_output_shg_field(
    input_field,
    control_field=1,
    d2=1,
    L=1,
    n1=1,
    n2=1,
    n3=1,
    deltaK=0,
):
    """
    Calculate output field intensity based on input and control field intensities.

    Parameters:
    -----------
    input_field : float
        Input field intensity
    control_field : float
        Control field intensity
    d2 : float, optional
        Nonlinear coefficient (default: 1e-3)
    L : float, optional
        Interaction length (default: 1 m)
    n1, n2, n3 : float, optional
        Refractive indices (default: 1)
    deltaK : float, optional
        Phase mismatch parameter (default: 0)

    Returns:
    --------
    float
        The calculated output field intensity
    """

    # Calculate the prefactor
    prefactor = 8 * d2**2 * epsilon_0**2 * input_field * control_field
    denominator = n1 * n2 * n3 * epsilon_0 * c**2

    # Calculate the L^2 * sinc^2(ΔkL/2) term
    sinc_term = L**2 * np.sinc(deltaK * L / (2 * pi)) ** 2

    # Calculate final result
    output_field = (prefactor / denominator) * sinc_term

    return output_field
