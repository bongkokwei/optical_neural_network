import numpy as np
from scipy.linalg import expm
from itertools import combinations_with_replacement, permutations
from typing import List, Tuple
import numpy.typing as npt
import math  # Added explicit math import


class FockStateInterferometer:
    def __init__(self, num_modes: int):
        """Initialize interferometer simulation with specified number of modes."""
        self.num_modes = num_modes
        self.U = np.eye(num_modes, dtype=complex)  # Identity by default

    def set_unitary(
        self,
        U: npt.NDArray[complex],  # type: ignore
    ):
        """Set the unitary transformation matrix for the interferometer."""
        assert U.shape == (
            self.num_modes,
            self.num_modes,
        ), "Invalid unitary matrix dimensions"
        assert np.allclose(
            U @ U.conj().T, np.eye(self.num_modes)
        ), "Matrix must be unitary"
        self.U = U

    def generate_random_unitary(self) -> None:
        """Generate a random unitary matrix using QR decomposition."""
        # Generate random complex matrix
        H = np.random.normal(
            size=(self.num_modes, self.num_modes)
        ) + 1j * np.random.normal(size=(self.num_modes, self.num_modes))
        # Make it Hermitian
        H = H + H.conj().T
        # Convert to unitary via matrix exponential
        self.U = expm(1j * H)

    def permanent(
        self,
        matrix: npt.NDArray[complex],  # type: ignore
    ) -> complex:
        """Calculate the permanent of a matrix using Ryser's algorithm."""
        n = matrix.shape[0]
        if n == 1:
            return matrix[0, 0]

        result = 0
        for k in range(2**n):
            term = 1
            sign = 1 if bin(k).count("1") % 2 == 0 else -1
            for i in range(n):
                row_sum = 0
                for j in range(n):
                    if k & (1 << j):  # Check bit j of k, not i
                        row_sum += matrix[i, j]
                term *= row_sum
            result += sign * term
        return result

    def get_submatrix(
        self,
        input_modes: List[int],
        output_modes: List[int],
    ) -> npt.NDArray[complex]:  # type: ignore
        """Extract submatrix corresponding to specific input and output modes."""
        return self.U[np.ix_(output_modes, input_modes)]

    def compute_transition_amplitude(
        self, input_state: List[int], output_state: List[int]
    ) -> complex:
        """
        Compute transition amplitude between input and output Fock states.

        Args:
            input_state: List of photon numbers in each input mode
            output_state: List of photon numbers in each output mode

        Returns:
            Complex transition amplitude
        """
        assert len(input_state) == self.num_modes
        assert len(output_state) == self.num_modes
        assert sum(input_state) == sum(output_state), "Photon number must be conserved"

        # Get list of occupied modes
        input_modes = [i for i, n in enumerate(input_state) for _ in range(n)]
        output_modes = [i for i, n in enumerate(output_state) for _ in range(n)]

        # Get relevant submatrix
        subU = self.get_submatrix(input_modes, output_modes)

        # Calculate normalization factor
        input_norm = np.prod([math.factorial(n) for n in input_state])
        output_norm = np.prod([math.factorial(n) for n in output_state])
        normalization = np.sqrt(input_norm * output_norm)

        return self.permanent(subU) / normalization

    def simulate_evolution(
        self,
        input_state: List[int],
        max_outputs: int = 10,
    ) -> List[Tuple[List[int], complex]]:
        """
        Simulate the evolution of an input Fock state through the interferometer.

        Args:
            input_state: List of photon numbers in each input mode
            max_outputs: Maximum number of output states to return

        Returns:
            List of (output_state, amplitude) tuples
        """
        total_photons = sum(input_state)

        # Generate all possible output configurations
        possible_outputs = []
        for partition in combinations_with_replacement(
            range(self.num_modes), total_photons
        ):
            output_state = [0] * self.num_modes
            for mode in partition:
                output_state[mode] += 1
            possible_outputs.append(output_state)

        # Calculate amplitudes for each output configuration
        results = []
        for output_state in possible_outputs:
            amplitude = self.compute_transition_amplitude(input_state, output_state)
            if abs(amplitude) > 1e-10:  # Filter out negligible amplitudes
                results.append((output_state, amplitude))

        # Sort by probability (|amplitude|^2) and return top results
        results.sort(key=lambda x: abs(x[1]) ** 2, reverse=True)
        return results[:max_outputs]


def example_simulation():
    """Example usage of the MultimodeInterferometer class."""
    # Create a 3-mode interferometer
    interferometer = FockStateInterferometer(num_modes=4)

    # Generate a random unitary transformation
    interferometer.generate_random_unitary()

    # Generate an input state
    input_state = np.ones(4, dtype=int)

    # Simulate evolution
    results = interferometer.simulate_evolution(
        input_state,
        max_outputs=-1,
    )

    # Print results
    print("\nSimulation Results:")
    print("Input state:", input_state)
    print("\nTop output states:")
    for state, amplitude in results:
        probability = abs(amplitude) ** 2
        print(
            f"State {state}: amplitude = {amplitude:.3f}, probability = {probability:.3f}"
        )


if __name__ == "__main__":
    example_simulation()
