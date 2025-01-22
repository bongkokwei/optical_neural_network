import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
from datetime import datetime
from tqdm.auto import tqdm

from onn.model.optical_cancer import OpticalCancerClassifier


class BreastCancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_breast_cancer_model(
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    device_max_inputs=16,
    num_optical_layers=3,
    save_dir="saved_models",
    save_checkpoint=True,
):
    """
    Train a neural network on the Wisconsin Breast Cancer dataset.

    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout rate for regularization
        save_dir (str): Directory to save model checkpoints
        save_checkpoint (bool): Whether to save model checkpoints

    Returns:
        model: The trained model
        str: Path to the saved model
    """
    os.makedirs(save_dir, exist_ok=True)

    # Record start time
    total_start_time = time.time()
    setup_start_time = time.time()

    # Load and preprocess data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets and dataloaders
    train_dataset = BreastCancerDataset(X_train_scaled, y_train)
    test_dataset = BreastCancerDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    input_dim = X_train.shape[1]
    model = OpticalCancerClassifier(
        input_dim,
        device_max_inputs=device_max_inputs,
        num_optical_layers=num_optical_layers,
    )
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

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
            predicted = (output >= 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            total_loss += loss.item()

            # Update progress bar
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
                predicted = (output >= 0.5).float()
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_accuracy = 100 * test_correct / test_total
        validation_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time

        # Create timestamp for save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"breast_cancer_model_best_{timestamp}.pth")

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
                    "setup_time": setup_time,
                    "train_time": train_time,
                    "validation_time": validation_time,
                    "batch_times_mean": np.mean(batch_times),
                    "timestamp": timestamp,
                },
                save_path,
            )
            print(f"\nSaved new best model with test accuracy: {test_accuracy:.2f}%")

        # Print epoch statistics
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
        # Save final model
        final_save_path = os.path.join(
            save_dir, f"breast_cancer_model_final_{timestamp}.pth"
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
                "total_training_time": total_time,
                "timestamp": timestamp,
            },
            final_save_path,
        )

    return model, save_path


if __name__ == "__main__":
    model, save_path = train_breast_cancer_model(
        epochs=20,
        batch_size=8,
        learning_rate=0.001,
        device_max_inputs=32,
        num_optical_layers=5,
        save_dir="./data/saved_models",
        save_checkpoint=False,
    )
