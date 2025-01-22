import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import math
import time
import os
from datetime import datetime
from tqdm.auto import tqdm

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

        # Save model if it has the best test accuracy
        if test_accuracy > best_test_accuracy and save_checkpoint:
            best_test_accuracy = test_accuracy
            best_epoch = epoch

            # Save the checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
                "optimizer_state_dict": optimizer.state_dict(),
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


# Example usage
if __name__ == "__main__":
    model, save_path = train_mnist_optical_network(
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        num_layers=4,  # num layers of dimension reduction
        num_optical_input=16,
        num_optical_layers=1,
        device_max_inputs=16,
        dropout_rate=0.35,
        save_dir="./data/optical_mnist_models",
        mnist_data_dir="./data",
        save_checkpoint=False,
    )
