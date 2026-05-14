import logging
from typing import Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a trained model on a test set and return metrics.

    Args:
    - model: Trained PyTorch model.
    - test_loader: DataLoader for the test dataset.
    - loss_function: Loss criterion.
    - device: Device to run evaluation on.

    Returns:
    - Dict with keys: loss, accuracy, precision, recall, f1.
    """
    model.eval()
    test_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += loss_function(outputs, targets).item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    test_loss /= len(test_loader)
    metrics = {
        "loss": test_loss,
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
    }

    logging.info(
        f"Test Results — Loss: {metrics['loss']:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )
    return metrics
