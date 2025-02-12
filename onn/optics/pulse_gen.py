import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.constants import c, pi
from scipy.integrate import quad


def generate_frequency_pulse(
    f,
    pulse_type="gaussian",
    center_frequencies=None,
    energies=None,
    width=1.0,
    center=0.0,
):
    """
    Generate a multi-peak Gaussian or hyperbolic secant pulse in frequency domain
    with specified pulse energies.

    Parameters:
    -----------
    f : array_like
        Frequency array in Hz
    pulse_type : str, optional
        Type of pulse ('gaussian' or 'sech'), default is 'gaussian'
    center_frequencies : array_like
        List of center frequencies for each peak in Hz
    energies : array_like, optional
        List of pulse energies for each peak. If None, all peaks have energy 1.
        Energies are normalized so their sum equals the total desired energy.
    width : float or array_like, optional
        Spectral width parameter for each peak (FWHM).
        If single value, same width applied to all peaks.
    center : float, optional
        Center frequency offset applied to all peaks

    Returns:
    --------
    array_like
        Complex spectrum of the multi-peak pulse
    dict
        Dictionary containing peak parameters and actual energies
    """

    # Input validation and defaults
    if center_frequencies is None:
        center_frequencies = [0]  # Default to single peak at zero frequency

    center_frequencies = np.array(center_frequencies)
    num_peaks = len(center_frequencies)

    if energies is None:
        energies = np.ones(num_peaks)
    else:
        energies = np.array(energies)
        if len(energies) != num_peaks:
            raise ValueError(
                "Number of energies must match number of center frequencies"
            )

    # Handle width parameter
    if np.isscalar(width):
        width = np.full(num_peaks, width)
    else:
        width = np.array(width)
        if len(width) != num_peaks:
            raise ValueError("Number of widths must match number of center frequencies")

    # Initialize spectrum and actual energy tracking
    spectrum = np.zeros_like(f, dtype=complex)
    actual_energies = {}

    # Frequency step for integration
    df = np.abs(f[1] - f[0])

    # Generate each spectral peak
    for idx, (freq, energy, w) in enumerate(zip(center_frequencies, energies, width)):
        # Shift frequency array by center frequency and overall offset
        f_shifted = f - (freq + center)

        # Generate unnormalized spectral shape based on pulse type
        if pulse_type.lower() == "gaussian":
            # Convert FWHM to standard deviation (σ)
            # FWHM = 2.355 * σ for Gaussian
            sigma = w / 2.355
            peak = np.exp(-0.5 * (f_shifted / sigma) ** 2)

            # Calculate normalization factor for desired energy
            # Energy = ∫|A(f)|²df
            current_energy = np.sum(np.abs(peak) ** 2) * df
            norm_factor = np.sqrt(energy / current_energy)
            peak *= norm_factor

        elif pulse_type.lower() == "sech":
            # For sech^2, FWHM ≈ 1.763 * w
            w_scaled = w / 1.763
            peak = (1 / np.cosh(2 * (f_shifted / w_scaled))) ** 2
            # print(np.max(np.cosh(2 * (f_shifted / w_scaled))))

            # Calculate normalization factor for desired energy
            current_energy = np.sum(np.abs(peak) ** 2) * df
            norm_factor = np.sqrt(energy / current_energy)
            peak *= norm_factor

        else:
            raise ValueError("Pulse type must be 'gaussian' or 'sech'")

        # Add normalized peak to spectrum
        spectrum += peak

        # Store actual energy of this peak
        actual_energies[f"peak_{idx}"] = {
            "center_freq": freq + center,
            "width": w,
            "target_energy": energy,
            "actual_energy": np.sum(np.abs(peak) ** 2) * df,
        }

    # Calculate total spectrum energy
    total_energy = np.sum(np.abs(spectrum) ** 2) * df
    actual_energies["total"] = total_energy

    return spectrum, actual_energies


def gen_pulse_freq(
    center_wavelength,
    fwhm,
    pulse_energy,
    pulse_type="gaussian",
    num_points=1000,
    unit=1e0,
    range_factor=100,
):
    """
    Simulate an optical pulse in frequency domain

    Parameters:
    -----------
    center_wavelength : float
        Center wavelength in nanometers
    fwhm : float
        Full Width at Half Maximum in nanometers
    pulse_energy : float
        Total pulse energy in arbitrary units
    pulse_type : str
        Type of pulse shape ('gaussian' or 'sech2')
    num_points : int
        Number of points for simulation
    unit : float
        adds suffix to parameter values (m)

    Returns:
    --------
    wavelengths : numpy array
        Wavelength array in nm
    frequencies : numpy array
        Frequency array in Hz
    spectrum : numpy array
        Spectral intensity
    """
    # Convert wavelength to frequency domain parameters
    center_freq = 3e8 / (center_wavelength * unit)  # Hz
    freq_width = 3e8 / ((center_wavelength - fwhm / 2) * unit) - 3e8 / (
        (center_wavelength + fwhm / 2) * unit
    )

    # Create wavelength array centered around center_wavelength
    wavelength_range = range_factor * fwhm  # simulate over 5x FWHM range
    wavelengths = np.linspace(
        center_wavelength - wavelength_range / 2,
        center_wavelength + wavelength_range / 2,
        num_points,
    )

    # Convert to frequency array
    frequencies = 3e8 / (wavelengths * unit)

    # Calculate spectrum based on pulse type
    if pulse_type.lower() == "gaussian":
        spectrum = np.exp(
            -4 * np.log(2) * ((frequencies - center_freq) / freq_width) ** 2
        )
    elif pulse_type.lower() == "sech":
        width_factor = 2 * np.log(1 + np.sqrt(2))
        spectrum = (
            1
            / np.cosh(
                (np.pi / width_factor) * (frequencies - center_freq) / (freq_width / 2)
            )
            ** 2
        )
    else:
        raise ValueError("Pulse type must be either 'gaussian' or 'sech'")

    # Normalize to given pulse energy
    spectrum = spectrum * np.abs(pulse_energy / np.trapezoid(spectrum, frequencies))

    return wavelengths, frequencies, spectrum


def calculate_temporal_profile(
    frequencies,
    spectrum,
    pad_factor: int = 1,
):
    """
    Calculate temporal profile using inverse Fourier transform

    Parameters:
    -----------
    frequencies : numpy array
        Frequency array in Hz
    spectrum : numpy array
        Spectral intensity

    Returns:
    --------
    time : numpy array
        Time array in femtoseconds
    temporal_intensity : numpy array
        Temporal intensity profile
    """
    # Take square root of spectrum to get electric field
    E_omega = np.sqrt(spectrum)

    # Perform inverse Fourier transform
    n_points = len(frequencies)
    frequencies = frequencies - frequencies[n_points // 2]
    n_pad = pad_factor * n_points
    df = (frequencies[-1] - frequencies[0]) / (n_points - 1)
    time = np.fft.fftshift(np.fft.fftfreq(n_pad, np.abs(df)))

    # Perform inverse Fourier transform
    E_omega_padded = np.zeros(n_pad, dtype=np.complex128)
    pad_start = (n_pad - n_points) // 2
    E_omega_padded[pad_start : pad_start + n_points] = E_omega
    E_t = np.fft.ifftshift(np.fft.ifft(E_omega_padded))

    # Calculate intensity
    temporal_intensity = np.abs(E_t) ** 2

    # Convert time to femtoseconds for plotting
    time_fs = time * 1e15

    return time_fs, temporal_intensity


def plot_pulse_domains(
    center_wavelength,
    fwhm,
    pulse_energy,
    pulse_type="sech",
    num_points=2**12,
):
    """
    Plot both spectral and temporal domains
    """
    # # Get spectral profile
    # wavelengths, frequencies, spectrum = gen_pulse_freq(
    #     center_wavelength,
    #     fwhm,
    #     pulse_energy,
    #     pulse_type,
    #     num_points=num_points,
    #     range_factor=10,
    # )

    # Get spectral profile
    wavelengths = np.linspace(700e-9, 1600e-9, num_points)
    frequencies = c / wavelengths
    center_wave = np.array([1550e-9, 775e-9])
    center_freq = c / center_wave
    widths = np.array([15e-9, 15e-9])  # in meter
    energies = [30e-15, 10e-15]

    spectrum, _ = generate_frequency_pulse(
        f=frequencies,
        pulse_type=pulse_type,
        center_frequencies=center_freq,
        energies=energies,
        width=c / center_wave**2 * widths,  # in freq
    )

    # Calculate temporal profile
    time_fs, temporal_intensity = calculate_temporal_profile(
        frequencies,
        spectrum,
        pad_factor=500,
    )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot spectral domain
    ax1.plot(wavelengths * 1e9, np.abs(spectrum))
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Spectral Intensity (a.u.)")
    ax1.set_title(f"{pulse_type.capitalize()} Pulse - Spectral Domain")
    ax1.grid(True)

    for c_wave, width in zip(center_wave, widths):
        ax1.axvline(
            c_wave * 1e9,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Center wavelength = {c_wave*1e9} nm",
        )
        ax1.axvline(
            (c_wave - width) * 1e9,
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"FWHM = {width*1e9:.2f} nm",
        )

        ax1.axvline(
            (c_wave + width) * 1e9,
            color="g",
            linestyle="--",
            alpha=0.5,
        )
    ax1.legend()

    # Plot temporal domain
    ax2.plot(time_fs, temporal_intensity / np.max(temporal_intensity))
    ax2.set_xlabel("Time (fs)")
    ax2.set_ylabel("Normalized Intensity")
    ax2.set_title(f"{pulse_type.capitalize()} Pulse - Temporal Domain")
    ax2.grid(True)

    # Calculate and display temporal FWHM
    temporal_fwhm = np.sum(temporal_intensity > np.max(temporal_intensity) / 2) * (
        time_fs[1] - time_fs[0]
    )
    ax2.text(
        0.05,
        0.95,
        f"Temporal FWHM: {temporal_fwhm:.4f} fs",
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # results = analyse_optical_pulse(
    #     pulse_width=1e-12,
    #     center_wavelength=1550e-9,
    #     time_window=100e-12,
    #     num_points=2**15,
    #     plot=True,
    # )

    plot_pulse_domains(
        center_wavelength=1550e-9,
        fwhm=3e-9,
        pulse_energy=1e-15,
        pulse_type="sech",
    )
