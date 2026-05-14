"""Utilities for loading models, running inference, and Grad-CAM."""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])


def load_model(
    checkpoint_path: str,
    n_classes: int = 10,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a CifarCNN from a saved state dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.cnn import CifarCNN

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CifarCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    logger.info("Model loaded from %s to %s", checkpoint_path, device)
    return model


def run_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[int, List[float]]:
    """Run inference on a single pre-processed image tensor (3, H, W)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs


def run_inference_batch(
    model: nn.Module,
    df,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on all rows of a DataFrame with image_path + label columns.

    Returns:
        y_true: (N,) ground-truth labels
        y_pred: (N,) predicted labels
        y_conf: (N,) confidence (max softmax prob) of the prediction
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.dataset import ImageDataset, eval_transform
    from torch.utils.data import DataLoader

    if device is None:
        device = next(model.parameters()).device

    dataset = ImageDataset(df, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true_list, y_pred_list, y_conf_list = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)
            confs = probs.max(dim=1).values
            y_true_list.append(labels.numpy())
            y_pred_list.append(preds.numpy())
            y_conf_list.append(confs.numpy())

    logger.info("Batch inference complete: %d samples", len(df))
    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_list),
        np.concatenate(y_conf_list),
    )


def preprocess_uploaded_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert an uploaded PIL image to a normalized (3, 32, 32) tensor."""
    return _EVAL_TRANSFORM(pil_image.convert("RGB"))


def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for the target class.

    Uses the last Conv2d layer (model.features[7]: 64→128 channels, output 16×16).

    Args:
        model: CifarCNN in eval mode.
        image_tensor: Pre-processed (3, 32, 32) float tensor.
        target_class: Class index to explain; defaults to the top prediction.
        device: Computation device.

    Returns:
        cam: (32, 32) float array in [0, 1], resized to input resolution.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    inp = image_tensor.unsqueeze(0).to(device)
    inp.requires_grad_(False)

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(module, input, output):
        activations.append(output.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_layer = model.features[7]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(inp)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        model.zero_grad()
        score = logits[0, target_class]
        score.backward()
    finally:
        h1.remove()
        h2.remove()

    acts = activations[0].squeeze(0)   # (C, H, W)
    grads = gradients[0].squeeze(0)    # (C, H, W)

    weights = grads.mean(dim=(1, 2))   # (C,)
    cam = (weights[:, None, None] * acts).sum(dim=0)  # (H, W)
    cam = torch.relu(cam).cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to 32×32 using PIL
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (32, 32), Image.BILINEAR
    )
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0
    return cam_resized
