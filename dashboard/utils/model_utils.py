"""Utilities for loading and running inference with CifarCNN."""
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn


def load_model(
    checkpoint_path: str,
    n_classes: int = 10,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a CifarCNN from a saved state dict.

    Args:
    - checkpoint_path: Path to the .pth checkpoint file.
    - n_classes: Number of output classes.
    - device: Target device; defaults to CPU.

    Returns:
    - Loaded model in eval mode.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.cnn import CifarCNN

    if device is None:
        device = torch.device("cpu")

    model = CifarCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def run_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[int, List[float]]:
    """Run inference on a single pre-processed image tensor.

    Args:
    - model: Loaded model in eval mode.
    - image_tensor: Float tensor of shape (3, H, W), already normalized.
    - device: Target device.

    Returns:
    - Tuple of (predicted class index, list of softmax probabilities).
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())

    return pred, probs


# Allow 'Optional' import needed above
from typing import Optional  # noqa: E402
