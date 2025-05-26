import torch
import torch.nn as nn
import numpy as np

from typing import (
    Literal,
    Optional,
    Union,
)
from enum import Enum
from onn.optics.output_fields import shg_out


class OpticalNonlinearLayer(nn.Module):
    def __init__(
        self,
        nonlinearity_type: Literal["FWM", "SHG"] = "FWM",
        eps: float = 1e-8,
        device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        self.nonlinearity_type = nonlinearity_type
        self.eps = eps
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization to the input tensor."""
        batch_min = x.min(dim=1, keepdim=True)[0]
        batch_max = x.max(dim=1, keepdim=True)[0]

        # Handle case where max = min
        diff = batch_max - batch_min
        diff = torch.where(diff == 0, torch.ones_like(diff) + self.eps, diff)

        return (x - batch_min) / diff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing the normalized nonlinear activation."""
        # Apply input normalization
        # x = self._minmax_normalize(x)

        if self.nonlinearity_type == "FWM":
            # Third-order nonlinearity: P ∝ χ(3)E³
            output = x**3
        elif self.nonlinearity_type == "SHG":  # SHG
            # Implement RelU here for now
            # output = x**2
            output = shg_out(x)

        # Apply output normalization
        output = self._minmax_normalize(output)

        return output

    def extra_repr(self) -> str:
        return f"nonlinearity_type={self.nonlinearity_type}, "
