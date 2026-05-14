"""Lab 6 — Streamlit interactive model analysis dashboard.

Tabs:
  1. Dataset Exploration  — dataset statistics, sample inspection, class filtering
  2. Error Analysis       — MLflow run comparison, model error breakdown
  3. Prediction & Explainability — per-sample inference with Grad-CAM or LIME
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CIFAR10_CLASSES

st.set_page_config(page_title="MLOps Dashboard", layout="wide")
st.title("MLOps Lab 6 — Model Analysis Dashboard")

tab_dataset, tab_errors, tab_explain = st.tabs([
    "Dataset Exploration",
    "Error Analysis",
    "Prediction & Explainability",
])

# ── Tab 1: Dataset Exploration ────────────────────────────────────────────────
with tab_dataset:
    st.header("Dataset Exploration")
    st.info("Implement in Lab 6: dataset statistics, sample inspection, class filtering.")

    # TODO (lab 6):
    #   from dashboard.utils.data_utils import load_split_dataframes, get_split_stats
    #   from dashboard.utils.viz_utils import plot_class_distribution
    #   train_df, val_df, test_df = load_split_dataframes()
    #   stats = get_split_stats(train_df, val_df, test_df)
    #   st.metric("Total samples", stats["total"])
    #   st.plotly_chart(plot_class_distribution(train_df, CIFAR10_CLASSES, "Train set"))

# ── Tab 2: Error Analysis ─────────────────────────────────────────────────────
with tab_errors:
    st.header("Error Analysis")
    st.info("Implement in Lab 6: load MLflow runs, compare metrics, display mispredictions.")

    # TODO (lab 6):
    #   from dashboard.utils.mlflow_utils import get_mlflow_runs
    #   runs_df = get_mlflow_runs(experiment_name="cifar10")
    #   st.dataframe(runs_df)
    #   from dashboard.utils.viz_utils import plot_confusion_matrix
    #   ...

# ── Tab 3: Prediction & Explainability ───────────────────────────────────────
with tab_explain:
    st.header("Prediction & Explainability")
    st.info("Implement in Lab 6: per-sample inference with Grad-CAM or LIME overlays.")

    # TODO (lab 6):
    #   from dashboard.utils.model_utils import load_model, run_inference
    #   model = load_model("outputs/best_model.pth")
    #   pred, probs = run_inference(model, image_tensor)
    #   st.bar_chart({CIFAR10_CLASSES[i]: probs[i] for i in range(10)})
    #   # Apply Grad-CAM and display overlay
