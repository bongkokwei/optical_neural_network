import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import time
import os
import argparse

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm.auto import tqdm
from pathlib import Path
from onn.model.optical_mnist import OpticalMNISTClassifier


def train_mnist_optical_network(
    epochs=15,
    batch_size=64,
    learning_rate=0.001,
    num_optical_input=32,
    num_layers=3,
    num_optical_layers=2,
    device_max_inputs=32,
    dropout_rate=0.2,
    optical_dropout=0.2,
    save_dir="saved_models",
    mnist_data_dir="./data",
    save_checkpoint=False,
):
    """
    Train the optical neural network on MNIST dataset.

    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Optimizer learning rate
        num_optical_input (int): Number of optical inputs
        save_dir (str): Directory to save the model
        save_checkpoint (bool): To save checkpoints into file

    Returns:
        model: The trained model
        str: Path to the saved model
    """
    os.makedirs(save_dir, exist_ok=True)

    # Record start time
    total_start_time = time.time()
    setup_start_time = time.time()

    # Define image transformations: convert to tensor and normalize with MNIST dataset statistics
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL image to tensor and scale to [0,1]
            transforms.Normalize(
                (0.1307,),  # Normalize using MNIST mean
                (0.3081,),  # and standard deviation
            ),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=mnist_data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=mnist_data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer

    model = OpticalMNISTClassifier(
        num_optical_input=num_optical_input,
        num_layers=num_layers,
        num_optical_layers=num_optical_layers,
        device_max_inputs=device_max_inputs,
        dropout_rate=dropout_rate,
    )
    print(model)

    # model = AllOpticalMNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    setup_time = time.time() - setup_start_time
    print(f"\nSetup time: {setup_time:.2f}s")

    # Track best model performance
    best_test_accuracy = 0
    best_epoch = 0

    # Training loop with tqdm for epochs
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_times = []

        # Training phase with tqdm for batches
        train_start_time = time.time()
        train_pbar = tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}", leave=False, unit="batch"
        )

        for batch_idx, (data, target) in enumerate(train_pbar):
            batch_start = time.time()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            total_loss += loss.item()

            # Update progress bar with current loss and accuracy
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100 * correct / total
            train_pbar.set_postfix(
                {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"}
            )

            batch_times.append(time.time() - batch_start)

        train_pbar.close()
        train_time = time.time() - train_start_time
        train_accuracy = 100 * correct / total

        # Validation phase
        val_start_time = time.time()
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_accuracy = 100 * test_correct / test_total
        validation_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time

        # Create a timestamp for the save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            save_dir,
            f"optical_mnist_model_best_{timestamp}.pth",
        )

        best_test_accuracy = test_accuracy
        best_epoch = epoch

        # Save model if it has the best test accuracy
        if test_accuracy > best_test_accuracy and save_checkpoint:
            # Save the checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimizer.state_dict(),
                    "train_loss": total_loss / len(train_loader),
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss / len(test_loader),
                    "test_accuracy": test_accuracy,
                    "num_optical_input": num_optical_input,
                    "setup_time": setup_time,
                    "train_time": train_time,
                    "validation_time": validation_time,
                    "batch_times_mean": np.mean(batch_times),
                    "timestamp": timestamp,
                },
                save_path,
            )
            print(f"\nSaved new best model with test accuracy: {test_accuracy:.2f}%")

        # Print epoch statistics with timing info
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(
            f"Training Loss: {total_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%"
        )
        print(
            f"Epoch Time: {epoch_time:.2f}s (Train: {train_time:.2f}s, Validation: {validation_time:.2f}s)"
        )
        print(f"Average Batch Time: {np.mean(batch_times):.3f}s")

    # Calculate and print total time
    total_time = time.time() - total_start_time
    print("\nFinal Training Summary:")
    print(f"Total Training Time: {total_time:.2f}s")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}% (Epoch {best_epoch + 1})")

    if save_checkpoint:
        # Save final model regardless of performance
        final_save_path = os.path.join(
            save_dir,
            f"optical_mnist_model_final_{timestamp}.pth",
        )
        torch.save(
            {
                "epoch": epochs - 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimizer.state_dict(),
                "train_loss": total_loss / len(train_loader),
                "train_accuracy": train_accuracy,
                "test_loss": test_loss / len(test_loader),
                "test_accuracy": test_accuracy,
                "num_optical_input": num_optical_input,
                "total_training_time": total_time,
                "timestamp": timestamp,
            },
            final_save_path,
        )

    return model, save_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MNIST Optical Network with customizable parameters"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    # Network architecture
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers for dimension reduction",
    )
    parser.add_argument(
        "--num-optical-input",
        type=int,
        default=32,
        help="Number of optical inputs",
    )
    parser.add_argument(
        "--num-optical-layers",
        type=int,
        default=2,
        help="Number of optical layers",
    )
    parser.add_argument(
        "--device-max-inputs",
        type=int,
        default=32,
        help="Maximum number of device inputs",
    )

    # Regularization
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for regularization",
    )

    # File paths
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data/optical_mnist_models",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--mnist-data-dir",
        type=str,
        default="./data",
        help="Directory containing MNIST dataset",
    )

    # Additional options
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Whether to save model checkpoints",
    )

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Call the training function with parsed arguments
    model, save_path = train_mnist_optical_network(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        num_optical_input=args.num_optical_input,
        num_optical_layers=args.num_optical_layers,
        device_max_inputs=args.device_max_inputs,
        dropout_rate=args.dropout_rate,
        save_dir=args.save_dir,
        mnist_data_dir=args.mnist_data_dir,
        save_checkpoint=args.save_checkpoint,
    )


if __name__ == "__main__":
    main()
