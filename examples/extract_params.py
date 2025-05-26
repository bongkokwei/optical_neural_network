from onn.layer.optical_layer import OpticalLinearLayer
from onn.model import OpticalMNISTClassifier


import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def extract_optical_parameters(model, save_plots=True):
    """Extract optical parameters from all optical layers in the model."""

    # Dictionary to store parameters for all optical layers
    params = {}

    # Get all optical layers (skipping the nonlinearity layers)
    optical_layers = [
        layer for layer in model.optical_layers if isinstance(layer, OpticalLinearLayer)
    ]

    for idx, layer in enumerate(optical_layers):
        params[f"optical_layer_{idx}"] = {}

        # Extract parameters from optical layer
        interferometer = layer.itf
        params[f"optical_layer_{idx}"][
            "transfer_matrix"
        ] = interferometer.calculate_transformation()
        params[f"optical_layer_{idx}"]["beam_splitters"] = []

        for bs in interferometer.BS_list:
            params[f"optical_layer_{idx}"]["beam_splitters"].append(
                {
                    "modes": (bs.mode1, bs.mode2),
                    "theta": bs.theta,
                    "phi": bs.phi,
                    "reflectivity": np.cos(bs.theta) ** 2,
                }
            )

        params[f"optical_layer_{idx}"]["output_phases"] = interferometer.output_phases

        # Generate visualizations if requested
        if save_plots:
            # Create figures directory if it doesn't exist
            figures_dir = "./data/figures"
            os.makedirs(figures_dir, exist_ok=True)

            # Plot interferometer configuration
            plt.figure()
            interferometer.draw(show_plot=False)
            plt.title(f"Optical Layer {idx} Configuration")
            plt.tight_layout()
            plt.savefig(
                os.path.join(figures_dir, f"interferometer_configuration_{idx}.png")
            )
            plt.close()

            # Plot transfer matrix magnitude
            plt.figure(figsize=(16, 10))
            plt.imshow(
                np.abs(params[f"optical_layer_{idx}"]["transfer_matrix"]),
                cmap="viridis",
            )
            plt.colorbar()
            plt.title(f"Transfer Matrix {idx} (Magnitude)")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"transfer_matrix_{idx}.png"))
            plt.close()

    return params


def select_model_file():
    """Open a file dialog to select the model file."""
    import tkinter as tk
    from tkinter import filedialog

    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()

    # Get the absolute path to ./data
    default_dir = os.path.abspath("./data")

    # Create data directory if it doesn't exist
    os.makedirs(default_dir, exist_ok=True)

    # Open file dialog
    file_path = filedialog.askopenfilename(
        initialdir=default_dir,
        title="Select model checkpoint file",
        filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
    )

    # Close the root window
    root.destroy()

    return file_path


def extract_and_analyze_model(model_path=None):
    """Load a trained model and extract its optical parameters."""

    # If no model_path provided, open file dialog
    if model_path is None:
        model_path = select_model_file()
        if not model_path:  # User cancelled
            print("No file selected")
            return None

    # Load the checkpoint
    checkpoint = torch.load(model_path)

    # Function to infer architecture from state dict
    def infer_architecture(state_dict):
        # Print all keys for debugging
        print("\nState dict keys:")
        for key in sorted(state_dict.keys()):
            if key.startswith("optical_layers"):
                print(key)

        reduction_sizes = []
        current_size = 784  # Input size

        # Find reduction layer sizes
        i = 0
        while f"reduction_layers.{i}.weight" in state_dict:
            weight = state_dict[f"reduction_layers.{i}.weight"]
            if i % 2 == 0:  # Only count Linear layers, skip ReLU
                reduction_sizes.append(weight.shape[0])
            i += 2

        # Find max optical layer index
        max_optical_index = max(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("optical_layers") and key.split(".")[1].isdigit()
        )

        return {
            "num_reduction_layers": len(reduction_sizes),
            "num_optical_input": reduction_sizes[-1],
            "num_optical_layers": (max_optical_index + 1)
            // 2,  # Each optical layer has nonlinearity
        }

    # Get architecture parameters from state dict
    arch_params = infer_architecture(checkpoint["model_state_dict"])
    print("\nInferred architecture parameters:")
    print(f"num_optical_input: {arch_params['num_optical_input']}")
    print(f"num_optical_layers: {arch_params['num_optical_layers']}")
    print(f"num_layers: {arch_params['num_layers']}")

    # Initialize model with inferred parameters
    model = OpticalMNISTClassifier(
        num_optical_input=arch_params["num_optical_input"],
        num_optical_layers=arch_params["num_optical_layers"],
        num_layers=arch_params["num_layers"],
    )

    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Print training info
    print(f"\nModel training information:")
    print(f"Epochs trained: {checkpoint['epoch']}")
    print(f"Final training loss: {checkpoint['train_loss']:.4f}")
    print(f"Final test accuracy: {checkpoint['test_accuracy']:.2f}%")
    print(f"Total training time: {checkpoint['total_training_time']:.2f} seconds")

    # Extract parameters
    params = extract_optical_parameters(model, save_plots=True)

    return params


if __name__ == "__main__":
    extract_and_analyze_model()
