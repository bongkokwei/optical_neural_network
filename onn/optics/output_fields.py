import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.constants import epsilon_0, mu_0, c, pi
from onn.optics.sellmeier import sellmeier_mgln

eta_0 = np.sqrt(mu_0 / epsilon_0)


def shg_out(
    a: torch.Tensor,
    wave_in: float = 1550e-9,
    n_eff: callable = sellmeier_mgln,
    d: float = 0.36e-12,
    crystal_length: float = 50e-3,
) -> torch.Tensor:
    """
    Calculates the output intensity of the three-wave mixing medium (Eqn 22.4-32b, Saleh 2019)
    using PyTorch operations to maintain gradient information.

    Args:
        a (torch.Tensor): input field amplitude
        wave_in (float): center wavelength of input field
        n_eff (callable): effective refractive index
        d (float): effective non linear coefficient
        crystal_length (float): length of waveguide

    Returns:
        torch.Tensor: Output intensity with maintained gradient information
    """
    # Convert constants to tensors with the same device and dtype as input
    eta = torch.tensor(eta_0 / n_eff(wave_in), device=a.device, dtype=a.dtype)
    # c_tensor = torch.tensor(c, device=a.device, dtype=a.dtype)
    # pi_tensor = torch.tensor(pi, device=a.device, dtype=a.dtype)

    # Calculate intermediate values
    omega = (2 * pi) * (c / wave_in)
    g = torch.sqrt(8 * d**2 * eta**3 * omega**2)

    # Calculate output using PyTorch operations
    return 0.5 * torch.abs(a) ** 2 * torch.tanh(a * g * crystal_length / 2) ** 2


def fwm_out(a):
    pass


if __name__ == "__main__":

    a = np.linspace(0, 1, 100)
    L = np.arange(0, 5000e-3, 100e-3)
    a_out = shg_out(a)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for length in L:
        ax.plot(a, shg_out(a, crystal_length=length))
    ax.set_xlabel("input field intensity")
    ax.set_ylabel("output field intensity")
    ax.set_yscale("log")
    ax.grid()
    plt.show()
