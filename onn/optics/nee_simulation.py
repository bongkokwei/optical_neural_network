import numpy as np

from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.integrate import solve_ivp
from scipy.constants import epsilon_0, c, pi
from functools import partial
from typing import Callable
import matplotlib.pyplot as plt

from pulse_gen import gen_pulse_freq, generate_frequency_pulse
from sellmeier import sellmeier_mgln


def ang_freq_to_wav(omega: float):
    return (2 * pi * c) / omega


def wav_to_ang_freq(wave: float):
    return (2 * pi * c) / wave


def beta(omega: float, n_eff: Callable):
    wav = ang_freq_to_wav(omega)
    return 2 * pi * n_eff(wav * 1e6) / wav


def get_poling_period(
    pump_wave: float,
    signal_wave: float,
    idler_wave: float,
    n_eff: Callable,
) -> float:
    """Calculate quasi-phase-matching period."""
    k_p = 2 * pi * n_eff(pump_wave * 1e6) / pump_wave
    k_s = 2 * pi * n_eff(signal_wave * 1e6) / signal_wave
    k_i = 2 * pi * n_eff(idler_wave * 1e6) / idler_wave
    return (2 * pi) / abs(k_p - k_s - k_i)


def get_poling_config(z: float, poling_period: float) -> int:
    """Returns +1 or -1 based on position within polling period using modulo."""
    normalized_z = z % poling_period
    return 1 if normalized_z < poling_period / 2 else -1


def calculate_transmission(thickness_cm, attenuation_db_per_cm):
    """Calculate the transmission of light through a material."""
    # Calculate the transmission using the formula T = 10^(-total_attenuation_db / 10)
    return 10 ** (-attenuation_db_per_cm * thickness_cm / 10)


def normalise_field(field):
    """Normalise field"""
    return np.abs(field) / np.max(np.abs(field))


def solve_nee(
    z_max=1.0,
    n_z_steps=1000,
    n_points=1024,
    beta_func=None,
    loss_cm=0.1,
    v_ref=1.0,
    poling=None,
    initial_pulse="sech",
    pulse_params=None,
):
    """
    Solve the Nonlinear Envelope Equation using split-step Fourier method.

    Parameters:
    -----------
    z_max : float
        Maximum propagation distance
    n_z_steps : int
        Number of steps in z direction
    n_points : int
        Number of freq points
    t_max : float
        Maximum time window
    beta_func : callable
        Function that returns beta(omega)
    loss_cm : float
        Attenuation constant (dB/cm)
    v_ref : float
        Reference velocity
    d_pattern : array-like
        Quasi-phase matching pattern (±1)
    initial_pulse : str
        Type of initial pulse ('sech' or 'gaussian')
    pulse_params : dict
        Parameters for the initial pulse

    Returns:
    --------
    field : ndarray
        Complex field amplitude at each z step
    t : ndarray
        Time grid
    omega : ndarray
        Frequency grid
    """
    # Convert alpha from dB/cm to linear units
    alpha = partial(calculate_transmission, attenuation_db_per_cm=loss_cm)

    # Create grids
    z_array, dz = np.linspace(0, z_max, n_z_steps, retstep=True, endpoint=True)
    print(f"step size in z-dir: {dz*1e6:.2f} micron")

    # _, freq, field = gen_pulse_freq(
    #     center_wavelength=pulse_params.get("wave_ref", 1550e-9),
    #     fwhm=pulse_params.get("width", 3e-9),
    #     pulse_energy=pulse_params.get("pulse_energy", 1),
    #     pulse_type=initial_pulse,
    #     num_points=n_points,
    #     range_factor=20,
    # )

    # Get spectral profile
    wavelengths = np.linspace(700e-9, 1600e-9, n_points)
    freq = c / wavelengths
    center_wave = np.array([1550e-9, 775e-9])
    center_freq = c / center_wave
    widths = np.array([50e-9, 50e-9])  # in meter
    energies = [300e-15, 1e-15]

    field, _ = generate_frequency_pulse(
        f=freq,
        pulse_type=initial_pulse,
        center_frequencies=center_freq,
        energies=energies,
        width=c / center_wave**2 * widths,  # in freq
    )

    delta_freq = freq - freq[n_points // 2]
    df = np.abs(delta_freq[-1] - delta_freq[0]) / (n_points - 1)
    t = fftshift(fftfreq(n_points, np.abs(df)))
    omega = 2 * np.pi * freq
    omega_ref = wav_to_ang_freq(params.get("wave_ref", 1550e-9))

    # Initialize arrays for results
    field_z = np.ones((n_z_steps, n_points), dtype=complex)
    field_z[0] = field

    if beta_func is None:
        raise ValueError("beta_func must be a function")

    linear_op = lambda z: -1j * (
        beta_func(omega) - beta_func(omega_ref) - omega / v_ref - alpha(z) / 2
    )

    # Define nonlinear operator (in time domain)
    X0 = pulse_params.get("X0", 1.0)  # Effective nonlinear coefficient

    def nonlinear_step(a_omega: np.ndarray, z_span):
        # Compute nonlinear term
        a_t_curr = ifft(a_omega)

        def nl_term(z, a_t: np.ndarray):
            # Convert to complex if needed
            # a_t = ifft(a_omega)
            if np.isscalar(a_t):
                a_t = complex(a_t)
            else:
                a_t = a_t.astype(np.complex128)

            phi = omega_ref * t - (beta_func(omega_ref) - omega_ref / v_ref) * z
            prefactor = 1j * ((omega + omega_ref) * epsilon_0 * X0 / 8) * poling(z)

            return prefactor * (
                a_t**2 * np.exp(1j * phi) + 2 * a_t * np.conj(a_t) * np.exp(-1j * phi)
            )

        sol = solve_ivp(nl_term, z_span, a_t_curr, method="RK45")
        return fft(sol.y[:, -1].reshape(-1))

    # Main propagation loop
    linear_propagator = np.exp(linear_op(dz) * dz / 2)

    for ii in range(1, n_z_steps):
        # Half step in frequency domain (linear effects)
        field_z[ii] = field_z[ii - 1] * linear_propagator

        # # Full nonlinear step in time domain
        z_curr = z_array[ii]
        field_z[ii] = nonlinear_step(
            field_z[ii],
            [z_curr, z_curr + dz],
        )

        # # Another half step in frequency domain
        field_z[ii] *= linear_propagator
        plt.show()

    return field_z, t, freq


# Example usage
if __name__ == "__main__":

    n_eff_tfln = partial(sellmeier_mgln, polarisation="e")
    beta_tfln = partial(beta, n_eff=n_eff_tfln)
    poling_period = get_poling_period(
        pump_wave=775e-9,
        signal_wave=1550e-9,
        idler_wave=1550e-9,
        n_eff=n_eff_tfln,
    )

    print(f"poling period: {poling_period*1e6:.2f} micron")
    poling_config = partial(get_poling_config, poling_period=poling_period)

    # Parameters from the paper
    params = {
        "width": 100e-9,  # 70 fs pulse width
        "beta": 1.0,  # Example dispersion
        "pulse_energy": 10e-10,  # joules
        "X0": 0.36e-12,  # Nonlinear coefficient (V²)
        "wave_ref": 1550e-9,
    }

    # Solve equation
    field_z, t, freq = solve_nee(
        z_max=25e-3,  # 2.5 mm propagation
        n_z_steps=20000,
        n_points=2**14,
        loss_cm=3,  # 0.1 dB/cm loss
        v_ref=c / n_eff_tfln(1.55),
        pulse_params=params,
        beta_func=beta_tfln,
        poling=poling_config,
    )

    # Plot results
    fig, (ax_t, ax_f) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax_t.plot(
        t * 1e12,
        np.abs(ifftshift(ifft(field_z[0]))),
        label="Input",
    )
    ax_t.plot(
        t * 1e12,
        np.abs(ifftshift(ifft(field_z[-1]))),
        "--",
        label="Output",
    )
    ax_t.set_xlabel("Time (ps)")
    ax_t.set_ylabel("Power (normalized)")
    ax_t.legend()
    ax_t.grid()

    ax_f.plot(
        freq * 1e-12,
        np.abs(field_z[0]) ** 2,
        label="Input",
    )
    ax_f.plot(
        freq * 1e-12,
        np.abs(field_z[-1]) ** 2,
        "--",
        label="Output",
    )
    ax_f.set_xlabel("Frequency (THz)")
    ax_f.set_ylabel("Spectrum (normalized)")
    ax_f.legend()
    ax_f.grid()

    # ax_f.set_yscale("log")
    # ax_t.set_yscale("log")

    fig.tight_layout()
    plt.show()
