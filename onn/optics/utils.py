import numpy as np
from scipy.linalg import qr


def custom_arctan(x1, x2):
    if x2 != 0:
        return np.arctan(abs(x1 / x2))
    else:
        return np.pi / 2


def custom_angle(x1, x2):
    """
    Computes the relative angle (phase difference) between two complex numbers.
    """
    if x2 != 0:
        return np.angle(x1 / x2)
    else:
        return 0


def fidelity(U, U_exp):

    return np.trace(U.conj().T @ U_exp) / np.sqrt(
        U.shape[0] * np.trace(U_exp.conj().T @ U)
    )


def random_unitary(N):
    """
    Generate a Haar random NxN unitary matrix.

    Args:
        N (int): Dimension of the unitary matrix.

    Returns:
        np.ndarray: Random unitary matrix.
    """
    # Create a random complex matrix with Gaussian entries
    X = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)

    # Perform QR decomposition
    Q, R = qr(X)

    # Ensure unitarity by adjusting the phase of the diagonal of R
    R = np.diag(np.divide(np.diag(R), np.abs(np.diag(R))))
    return Q @ R
