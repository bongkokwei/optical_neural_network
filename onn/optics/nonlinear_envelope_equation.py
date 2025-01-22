import numpy as np
from scipy.fft import fft, ifft
from scipy.integrate import solve_ivp
from scipy.constants import epsilon_0, c, pi


class NEESimulator:
    def __init__(self, z_points=1000, t_points=1024, t_window=10e-12):
        """
        Initialize the NEE simulator.

        Args:
            z_points (int): Number of points in the propagation direction
            t_points (int): Number of points in the time domain
            t_window (float): Time window in seconds
        """
        self.z_points = z_points
        self.t_points = t_points
        self.t_window = t_window

        # Setup the time and frequency grids
        self.dt = t_window / t_points
        self.t = np.linspace(-t_window / 2, t_window / 2, t_points)
        self.omega = 2 * np.pi * np.fft.fftfreq(t_points, self.dt)

    def simulate_nee(self, A_initial, beta, alpha, X0, d_z, v_ref, omega0):
        """
        Simulate the nonlinear envelope equation.

        Args:
            A_initial (np.array): Initial field amplitude
            beta (callable): Dispersion relation β(ω)
            alpha (float): Attenuation constant
            X0 (float): Effective nonlinear coefficient
            d_z (float): Propagation step size
            v_ref (float): Reference velocity
            omega0 (float): Central frequency

        Returns:
            np.array: Field evolution along z
        """
        z = np.linspace(0, d_z * self.z_points, self.z_points)
        A = np.zeros((self.z_points, self.t_points), dtype=complex)
        A[0] = A_initial

        def rhs(z, A_vec):
            """Right-hand side of the NEE."""
            A = A_vec.reshape(-1)

            # Linear operator in frequency domain
            A_freq = fft(A)
            Omega = self.omega - omega0

            # Apply dispersion and attenuation
            L = -1j * (beta(self.omega) - beta(omega0) - Omega / v_ref) - alpha / 2
            A_freq = A_freq * np.exp(L * d_z)
            A = ifft(A_freq)

            # Nonlinear term
            a_t = ifft(A_freq)
            d = 1  # Assuming uniform poling pattern for simplicity
            phi = omega0 * self.t - (beta(omega0) - omega0 / v_ref) * z

            NL = (
                -1j
                * omega0
                * X0
                / 8
                * d
                * (
                    np.abs(a_t) ** 2 * a_t * np.exp(1j * phi)
                    + 2 * a_t * np.conj(a_t) * a_t * np.exp(-1j * phi)
                )
            )

            return NL.reshape(-1)

        # Solve using split-step Fourier method
        for i in range(1, self.z_points):
            A_vec = A[i - 1].reshape(-1)

            # RK4 for nonlinear step
            sol = solve_ivp(rhs, [z[i - 1], z[i]], A_vec, method="RK45", t_eval=[z[i]])
            A[i] = sol.y[:, -1].reshape(-1)

        return A


def create_sech_pulse(t, t0, P0):
    """Create a hyperbolic secant pulse."""
    return np.sqrt(P0) * 1 / np.cosh(t / t0)


# Define dispersion relation (example: including up to β2)
def beta(omega):
    beta0 = 4e6  # m^-1
    beta1 = 1 / 2e8  # s/m
    beta2 = 1e-26  # s^2/m
    omega0 = 2 * np.pi * 3e14  # Central frequency
    return beta0 + beta1 * (omega - omega0) + beta2 / 2 * (omega - omega0) ** 2


# Example usage
if __name__ == "__main__":
    # Setup simulation parameters
    t_window = 10e-12  # 10 ps window
    t_points = 1024
    z_points = 1000

    # Create simulator instance
    simulator = NEESimulator(z_points, t_points, t_window)

    # Create initial pulse
    t0 = 0.1e-12  # 100 fs pulse duration
    P0 = 1e3  # Peak power
    A_initial = create_sech_pulse(simulator.t, t0, P0)

    # Simulation parameters
    alpha = 0.1 * 100  # 0.1 dB/cm converted to m^-1
    X0 = 0.36e-12  # Effective nonlinear coefficient
    d_z = 1e-6  # 1 μm step size
    v_ref = 3e8 / 2  # Reference velocity (c/2)
    omega0 = 2 * np.pi * 3e14  # Central frequency

    # Run simulation
    A_evolution = simulator.simulate_nee(A_initial, beta, alpha, X0, d_z, v_ref, omega0)

    # Example plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.abs(A_evolution) ** 2,
        aspect="auto",
        extent=[simulator.t[0] * 1e12, simulator.t[-1] * 1e12, 0, z_points * d_z * 1e3],
    )
    plt.colorbar(label="|A|²")
    plt.xlabel("Time (ps)")
    plt.ylabel("Distance (mm)")
    plt.title("Pulse Evolution")
    plt.show()
