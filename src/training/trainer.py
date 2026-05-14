import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: Path = Path("outputs/best_model.pth"),
) -> Path:
    """Train a model with validation-based best-checkpoint selection.

    Args:
    - model: PyTorch model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - loss_function: Loss criterion.
    - optimizer: Optimizer instance.
    - num_epochs: Number of training epochs.
    - device: Device to run training on.
    - save_path: Path to save the best model checkpoint.

    Returns:
    - Path to the saved best model checkpoint.
    """
    model.to(device)
    best_val_loss: float = float("inf")
    best_model_path = save_path

    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        model.eval()
        val_loss: float = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_targets).item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_path
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

    logging.info("Training complete.")
    return best_model_path
