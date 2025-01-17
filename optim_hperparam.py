import optuna
from train_mnist_model import train_mnist_optical_network
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os
import json


def objective(trial):
    # Define the hyperparameter search space
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "num_optical_input": trial.suggest_int("num_optical_input", 16, 64, step=8),
        "num_optical_layers": trial.suggest_int("num_optical_layers", 1, 3),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }

    # Fixed number of epochs for optimization
    params["epochs"] = 5  # Reduced epochs for faster optimization

    try:
        # Train model with current hyperparameters
        model, _ = train_mnist_optical_network(
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            num_optical_input=params["num_optical_input"],
            num_layers=params["num_layers"],
            num_optical_layers=params["num_optical_layers"],
            dropout_rate=params["dropout_rate"],
            save_checkpoint=False,  # Don't save intermediate models
        )

        # Evaluate model on validation set
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total

        # Save trial results
        trial_results = {
            "params": params,
            "accuracy": accuracy,
            "trial_number": trial.number,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        # Save to JSON file
        os.makedirs("optimization_results", exist_ok=True)
        with open(f"optimization_results/trial_{trial.number}.json", "w") as f:
            json.dump(trial_results, f, indent=4)

        return accuracy

    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        return float("-inf")


def optimize_hyperparameters(n_trials=50):
    """
    Run hyperparameter optimization using Optuna

    Args:
        n_trials: Number of optimization trials to run
    """
    study_name = (
        f"./data/optical_mnist_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials)

    # Print optimization results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Save study results
    study_results = {
        "study_name": study_name,
        "best_trial": {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
        },
        "n_trials": n_trials,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    os.makedirs("optimization_results", exist_ok=True)
    with open(f"optimization_results/{study_name}_summary.json", "w") as f:
        json.dump(study_results, f, indent=4)

    return study


if __name__ == "__main__":
    study = optimize_hyperparameters(n_trials=50)
