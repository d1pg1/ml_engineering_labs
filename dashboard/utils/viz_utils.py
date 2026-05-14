"""Visualization helpers for the Streamlit dashboard."""
import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def plot_class_distribution(
    df: pd.DataFrame,
    class_names: List[str],
    title: str = "Class Distribution",
):
    """Plotly bar chart of label counts."""
    counts = df["label"].value_counts().sort_index()
    fig = px.bar(
        x=[class_names[i] for i in counts.index],
        y=counts.values,
        labels={"x": "Class", "y": "Count"},
        title=title,
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(xaxis_tickangle=-35, height=350)
    return fig


def plot_split_sizes(stats: Dict[str, int]):
    """Plotly horizontal bar chart showing train/val/test sizes."""
    splits = ["Train", "Val", "Test"]
    sizes = [stats["train"], stats["val"], stats["test"]]
    fig = px.bar(
        x=sizes,
        y=splits,
        orientation="h",
        labels={"x": "Samples", "y": "Split"},
        title="Dataset Split Sizes",
        color_discrete_sequence=["#72B7B2"],
        text=sizes,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=250, xaxis_range=[0, max(sizes) * 1.15])
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> plt.Figure:
    """Matplotlib heatmap of the confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_per_class_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
):
    """Plotly bar chart showing number of errors per true class."""
    error_mask = y_true != y_pred
    error_labels = y_true[error_mask]
    counts = np.bincount(error_labels, minlength=len(class_names))
    fig = px.bar(
        x=class_names,
        y=counts,
        labels={"x": "True Class", "y": "Error Count"},
        title="Misclassifications per Class",
        color=counts,
        color_continuous_scale="Reds",
    )
    fig.update_layout(xaxis_tickangle=-35, height=350, coloraxis_showscale=False)
    return fig


def plot_probability_distribution(probs: List[float], class_names: List[str]):
    """Plotly horizontal bar chart of softmax probabilities."""
    sorted_pairs = sorted(zip(probs, class_names), reverse=True)
    sorted_probs, sorted_names = zip(*sorted_pairs)
    colors = ["#E45756" if i == 0 else "#4C78A8" for i in range(len(sorted_probs))]
    fig = go.Figure(go.Bar(
        x=list(sorted_probs),
        y=list(sorted_names),
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in sorted_probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis=dict(title="Probability", range=[0, 1.1]),
        yaxis=dict(title="Class"),
        height=350,
    )
    return fig


def plot_gradcam_overlay(
    original_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> plt.Figure:
    """Side-by-side figure: original image | Grad-CAM overlay."""
    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = (1 - alpha) * (original_img.astype(np.float32) / 255.0) + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    return fig
