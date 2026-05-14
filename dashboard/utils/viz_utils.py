"""Visualization helpers for the Streamlit dashboard."""
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution(
    df: pd.DataFrame,
    class_names: List[str],
    title: str = "Class Distribution",
) -> plt.Figure:
    """Bar chart of label counts for the given DataFrame."""
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar([class_names[i] for i in counts.index], counts.values)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> plt.Figure:
    """Heatmap of a confusion matrix computed from true and predicted labels."""
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig
