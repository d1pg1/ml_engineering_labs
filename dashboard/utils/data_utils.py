"""Utilities for loading and inspecting dataset splits."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def load_split_dataframes(
    data_dir: str = "data",
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CIFAR-10 image manifest and return train/val/test DataFrames.

    Assumes images have already been extracted by the training pipeline.
    Re-uses the same split logic with the same random_state for consistency.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.dataset import load_labels, train_test_split

    cifar_batches = str(Path(data_dir) / "cifar-10-batches-py")
    labels_df = load_labels(cifar_batches)

    train_val_df, test_df = train_test_split(labels_df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)
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
