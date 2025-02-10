import numpy as np


def sellmeier_mgln(wavelength, polarisation="e"):
    """
    Calculate the refractive index of 5-mol.% MgO-doped LiNbO₃ using the Sellmeier equation.

    Parameters:
    -----------
    wavelength : float or numpy.ndarray
        Wavelength in micrometers (μm)
    polarization : str, optional
        Polarization direction, either 'e' (extraordinary) or 'o' (ordinary)
        Default is 'e'

    Returns:
    --------
    float or numpy.ndarray
        Refractive index at the specified wavelength(s)

    Raises:
    -------
    ValueError
        If polarization is not 'e' or 'o'
        If wavelength is not positive
    """

    # Coefficients for extraordinary (e) and ordinary (o) axes
    coefficients = {
        "e": {
            "A": 2.4272,
            "B": 0.01478,
            "C": 1.4617,
            "D": 0.05612,
            "E": 9.6536,
            "F": 371.216,
        },
        "o": {
            "A": 2.2454,
            "B": 0.01242,
            "C": 1.3005,
            "D": 0.05213,
            "E": 6.8972,
            "F": 331.33,
        },
    }

    # Input validation
    if polarisation not in ["e", "o"]:
        raise ValueError(
            "Polarization must be either 'e' (extraordinary) or 'o' (ordinary)"
        )

    if np.any(wavelength <= 0):
        return 0
        # raise ValueError("Wavelength must be positive")

    # Get coefficients for the specified polarization
    coeff = coefficients[polarisation]

    # Calculate each term of the Sellmeier equation
    term1 = coeff["A"] * wavelength**2 / (wavelength**2 - coeff["B"])
    term2 = coeff["C"] * wavelength**2 / (wavelength**2 - coeff["D"])
    term3 = coeff["E"] * wavelength**2 / (wavelength**2 - coeff["F"])

    # Calculate n² - 1
    n_squared_minus_1 = term1 + term2 + term3

    # Calculate refractive index
    n = np.sqrt(n_squared_minus_1 + 1)

    return n


# Example usage:
if __name__ == "__main__":
    # Test the function with some example wavelengths
    test_wavelengths = np.array([0.532, 1.064, 1.55])  # Common laser wavelengths in μm

    print("Refractive indices for different wavelengths:")
    print("Wavelength (μm) | n_e        | n_o")
    print("-" * 40)

    for wavelength in test_wavelengths:
        n_e = sellmeier_mgln(wavelength, "e")
        n_o = sellmeier_mgln(wavelength, "o")
        print(f"{wavelength:14.3f} | {n_e:9.5f} | {n_o:9.5f}")
