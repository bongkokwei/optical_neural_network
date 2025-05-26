import numpy as np
from onn.optics.utils import custom_angle, custom_arctan, random_unitary, fidelity


class Beamsplitter:
    """
    Defines a beamsplitter for mode transformations.

    The transformation matrix is:

        [ e^{i*phi} * cos(theta)    -sin(theta)        ]
        [ e^{i*phi} * sin(theta)     cos(theta)        ]

    Args:
        mode1 (int): Index of the first mode (1-based).
        mode2 (int): Index of the second mode.
        theta (float): Beamsplitter angle.
        phi (float): Beamsplitter phase.
    """

    def __init__(self, mode1, mode2, theta, phi):
        self.mode1 = mode1
        self.mode2 = mode2
        self.theta = theta
        self.phi = phi
        self.trainable = self._is_trainable()

    def _is_trainable(self):
        """
        Determines if the beamsplitter parameters are trainable.
        Returns False if both theta and phi are integer multiples of pi,
        True otherwise.
        """

        def is_pi_multiple(value):
            eps = 1e-10
            return abs(value % np.pi) < eps or abs(value % np.pi - np.pi) < eps

        return not (is_pi_multiple(self.theta) and is_pi_multiple(self.phi))

    def __repr__(self):
        return (
            f"\n Beamsplitter between modes {self.mode1} and {self.mode2}: "
            f"\n Theta angle: {self.theta:.2f} \n Phase: {self.phi:.2f}"
            f"\n Trainable parameters: {self.trainable}"
        )


class Interferometer:
    """
    Defines an interferometer consisting of a series of beamsplitters.

    Attributes:
        BS_list (list): List of Beamsplitter objects.
        output_phases (list): List of phase shifts for each mode.
    """

    def __init__(self):
        self.BS_list = []
        self.output_phases = []

    def add_BS(self, BS):
        """Add a beamsplitter to the interferometer."""
        self.BS_list.append(BS)

    def add_phase(self, mode, phase):
        """Add a phase shift to a specific mode."""
        while mode > len(self.output_phases):
            self.output_phases.append(0)
        self.output_phases[mode - 1] = phase

    def count_modes(self):
        """Count the number of modes in the interferometer."""
        return max(max(BS.mode1, BS.mode2) for BS in self.BS_list)

    def calculate_transformation(self):
        """
        Calculate the unitary matrix representing the interferometer.

        Returns:
            numpy.ndarray: Unitary matrix of the interferometer.
        """
        N = self.count_modes()
        U = np.eye(N, dtype=np.complex128)

        for BS in self.BS_list:
            T = np.eye(N, dtype=np.complex128)
            T[BS.mode1 - 1, BS.mode1 - 1] = np.exp(1j * BS.phi) * np.cos(BS.theta)
            T[BS.mode1 - 1, BS.mode2 - 1] = -np.sin(BS.theta)
            T[BS.mode2 - 1, BS.mode1 - 1] = np.exp(1j * BS.phi) * np.sin(BS.theta)
            T[BS.mode2 - 1, BS.mode2 - 1] = np.cos(BS.theta)
            U = T @ U

        while len(self.output_phases) < N:
            self.output_phases.append(0)

        D = np.diag(np.exp(1j * np.array(self.output_phases)))
        return D @ U

    def draw(self, show_plot=True):
        """
        Visualize the interferometer using Matplotlib.

        Args:
            show_plot (bool): Whether to display the plot.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 10))
        N = self.count_modes()
        mode_tracker = np.zeros(N)

        for ii in range(N):
            plt.plot((-1, 0), (ii, ii), lw=1, color="blue")

        for ii, BS in enumerate(self.BS_list):
            x = max(mode_tracker[BS.mode1 - 1], mode_tracker[BS.mode2 - 1])
            plt.plot((x + 0.3, x + 1), (N - BS.mode1, N - BS.mode2), lw=2, color="red")
            plt.plot((x, x + 0.3), (N - BS.mode1, N - BS.mode1), lw=1, color="green")
            plt.plot((x, x + 0.3), (N - BS.mode2, N - BS.mode2), lw=1, color="orange")
            plt.plot((x + 0.3, x + 1), (N - BS.mode2, N - BS.mode1), lw=2, color="pink")
            reflectivity = f"BS{ii}: r = {np.cos(BS.theta)**2:.2f}"
            plt.text(
                x + 0.9,
                N - (BS.mode1 + BS.mode2) / 2,
                reflectivity,
                color="green",
                fontsize=7,
            )
            mode_tracker[BS.mode1 - 1] = x + 1
            mode_tracker[BS.mode2 - 1] = x + 1

        max_x = np.max(mode_tracker)
        for ii in range(N):
            plt.plot(
                (mode_tracker[ii], max_x + 1),
                (N - ii - 1, N - ii - 1),
                lw=1,
                color="blue",
            )

        plt.axis("off")
        if show_plot:
            plt.show()


def square_decomposition(U):
    """
    Returns a rectangular mesh of beam splitters implementing matrix U

    This code implements the decomposition algorithm in:
    Clements, William R., et al. "Optimal design for universal multiport interferometers."
    Optica 3.12 (2016): 1460-1465.

    Args:
        U (np.ndarray): Unitary matrix to decompose.

    Returns:
        Interferometer: An instance of the Interferometer class containing the decomposition.
    """
    I = Interferometer()
    N = int(np.sqrt(U.size))  # Determine the size of the unitary matrix
    left_T = []  # Store intermediate transformations for later processing
    # Iterate over the layers of the interferometer
    for ii in range(N - 1):
        if np.mod(ii, 2) == 0:  # Even layer (forward sweep)
            for jj in range(ii + 1):
                # Identify the modes being interfered
                modes = [ii - jj + 1, ii + 2 - jj]
                # Compute the parameters for the beamsplitter
                theta = custom_arctan(
                    U[N - 1 - jj, ii - jj],
                    U[N - 1 - jj, ii - jj + 1],
                )
                phi = custom_angle(
                    U[N - 1 - jj, ii - jj],
                    U[N - 1 - jj, ii - jj + 1],
                )
                # Create the inverse transformation matrix
                invT = np.eye(N, dtype=np.complex128)
                invT[modes[0] - 1, modes[0] - 1] = np.exp(-1j * phi) * np.cos(theta)
                invT[modes[0] - 1, modes[1] - 1] = np.exp(-1j * phi) * np.sin(theta)
                invT[modes[1] - 1, modes[0] - 1] = -np.sin(theta)
                invT[modes[1] - 1, modes[1] - 1] = np.cos(theta)
                # Apply the transformation to the unitary matrix
                U = np.matmul(U, invT)
                # Append the beamsplitter to the interferometer
                I.BS_list.append(Beamsplitter(modes[0], modes[1], theta, phi))
        else:  # Odd layer (backward sweep)
            for jj in range(ii + 1):
                # Identify the modes being interfered
                modes = [N + jj - ii - 1, N + jj - ii]
                # Compute the parameters for the beamsplitter
                theta = custom_arctan(
                    U[N + jj - ii - 1, jj],
                    U[N + jj - ii - 2, jj],
                )
                phi = custom_angle(
                    -U[N + jj - ii - 1, jj],
                    U[N + jj - ii - 2, jj],
                )
                # Create the transformation matrix
                T = np.eye(N, dtype=np.complex128)
                T[modes[0] - 1, modes[0] - 1] = np.exp(1j * phi) * np.cos(theta)
                T[modes[0] - 1, modes[1] - 1] = -np.sin(theta)
                T[modes[1] - 1, modes[0] - 1] = np.exp(1j * phi) * np.sin(theta)
                T[modes[1] - 1, modes[1] - 1] = np.cos(theta)
                # Apply the transformation to the unitary matrix
                U = np.matmul(T, U)
                # Store the beamsplitter for later processing
                left_T.append(Beamsplitter(modes[0], modes[1], theta, phi))

    # Process the stored transformations in reverse order
    for BS in np.flip(left_T, 0):
        modes = [int(BS.mode1), int(BS.mode2)]
        # Create the inverse transformation matrix
        invT = np.eye(N, dtype=np.complex128)
        invT[modes[0] - 1, modes[0] - 1] = np.exp(-1j * BS.phi) * np.cos(BS.theta)
        invT[modes[0] - 1, modes[1] - 1] = np.exp(-1j * BS.phi) * np.sin(BS.theta)
        invT[modes[1] - 1, modes[0] - 1] = -np.sin(BS.theta)
        invT[modes[1] - 1, modes[1] - 1] = np.cos(BS.theta)
        # Apply the transformation to the unitary matrix
        U = np.matmul(invT, U)
        # Recompute the parameters
        theta = custom_arctan(
            U[modes[1] - 1, modes[0] - 1], U[modes[1] - 1, modes[1] - 1]
        )
        phi = custom_angle(U[modes[1] - 1, modes[0] - 1], U[modes[1] - 1, modes[1] - 1])
        # Update the transformation matrix
        invT[modes[0] - 1, modes[0] - 1] = np.exp(-1j * phi) * np.cos(theta)
        invT[modes[0] - 1, modes[1] - 1] = np.exp(-1j * phi) * np.sin(theta)
        invT[modes[1] - 1, modes[0] - 1] = -np.sin(theta)
        invT[modes[1] - 1, modes[1] - 1] = np.cos(theta)
        # Apply the transformation
        U = np.matmul(U, invT)
        # Append the beamsplitter to the interferometer
        I.BS_list.append(Beamsplitter(modes[0], modes[1], theta, phi))

    # Extract the phases from the diagonal elements of the matrix
    phases = np.diag(U)
    I.output_phases = [np.angle(i) for i in phases]
    return I


# Example usage
if __name__ == "__main__":

    U = random_unitary(32)
    print("Random unitary matrix:")

    interferometer = square_decomposition(U)
    print("Decomposed Interferometer:")
    for i, bs in enumerate(interferometer.BS_list):
        print(bs)
        print(f"BS{i+1}")

    # print("Output phases:", interferometer.output_phases)
    reconstructed_U = interferometer.calculate_transformation()
    print("Reconstructed Unitary:")
    # print(reconstructed_U)

    print(f"fidelity: {np.abs(fidelity(U, reconstructed_U)):.4f}")

    interferometer.draw()
