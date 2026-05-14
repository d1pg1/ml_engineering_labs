"""Utilities for loading and inspecting dataset splits."""
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_split_dataframes(
    labels_path: str = "data/cifar-10-batches-py",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CIFAR-10 labels and return (train_df, val_df, test_df).

    Reuses the same split logic and random_state so splits are always consistent.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.dataset import load_labels, train_test_split

    logger.info("Loading labels from %s", labels_path)
    labels_df = load_labels(labels_path)

    train_val_df, test_df = train_test_split(labels_df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)
    logger.info("Splits — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def get_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, int]:
    return {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(train_df) + len(val_df) + len(test_df),
    }


def get_image_for_display(image_path: str) -> np.ndarray:
    """Read an image from disk and return as HWC uint8 numpy array."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return np.array(img)
