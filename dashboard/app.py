"""Lab 6 — Streamlit interactive model analysis dashboard.

Run from repo root:
    streamlit run dashboard/app.py

Tabs:
  1. Dataset Exploration  — dataset statistics, sample inspection, class filtering
  2. Error Analysis       — MLflow run selection, confusion matrix, misclassification browser
  3. Prediction & Explainability — per-sample inference + uploaded image + Grad-CAM
"""
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.utils.config import load_config
from dashboard.utils.data_utils import get_image_for_display, get_split_stats, load_split_dataframes
from dashboard.utils.mlflow_utils import (
    get_experiment_names,
    get_mlflow_runs,
    get_run_metric_history,
    load_model_from_run,
)
from dashboard.utils.model_utils import (
    compute_gradcam,
    load_model,
    preprocess_uploaded_image,
    run_inference,
    run_inference_batch,
)
from dashboard.utils.viz_utils import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_gradcam_overlay,
    plot_per_class_errors,
    plot_probability_distribution,
    plot_split_sizes,
)
from src.data.dataset import CIFAR10_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MLOps Lab 6 — Dashboard", layout="wide", page_icon="🔬")
st.title("🔬 MLOps Lab 6 — Model Analysis Dashboard")

cfg = load_config()
MLFLOW_URI = str(ROOT / cfg["mlflow"]["tracking_uri"])
EXPERIMENT = cfg["mlflow"]["experiment_name"]
LABELS_PATH = str(ROOT / cfg["data"]["labels_path"])
CHECKPOINT = str(ROOT / cfg["model"]["checkpoint_path"])
N_CLASSES = cfg["model"]["n_classes"]
BATCH_SIZE = cfg["model"]["batch_size"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset splits…")
def _load_splits():
    return load_split_dataframes(
        labels_path=LABELS_PATH,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
    )


@st.cache_data(show_spinner="Loading MLflow runs…")
def _load_runs(experiment: str):
    return get_mlflow_runs(experiment, tracking_uri=MLFLOW_URI)


@st.cache_resource(show_spinner="Loading model…")
def _load_default_model():
    return load_model(CHECKPOINT, n_classes=N_CLASSES, device=DEVICE)


@st.cache_resource(show_spinner="Loading model from MLflow run…")
def _load_run_model(run_id: str):
    return load_model_from_run(run_id, tracking_uri=MLFLOW_URI, n_classes=N_CLASSES, device=DEVICE)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dataset, tab_errors, tab_explain = st.tabs([
    "📊 Dataset Exploration",
    "🔍 Error Analysis",
    "💡 Prediction & Explainability",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Dataset Exploration
# ─────────────────────────────────────────────────────────────────────────────
with tab_dataset:
    st.header("Dataset Exploration")

    try:
        train_df, val_df, test_df = _load_splits()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    stats = get_split_stats(train_df, val_df, test_df)

    # ── Overview metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Samples", f"{stats['total']:,}")
    c2.metric("Classes", str(N_CLASSES))
    c3.metric("Train", f"{stats['train']:,}")
    c4.metric("Validation", f"{stats['val']:,}")
    c5.metric("Test", f"{stats['test']:,}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(plot_split_sizes(stats), use_container_width=True)
    with col_r:
        split_choice = st.selectbox("Split for class distribution", ["Train", "Val", "Test"], key="dist_split")
        df_map = {"Train": train_df, "Val": val_df, "Test": test_df}
        st.plotly_chart(
            plot_class_distribution(df_map[split_choice], CIFAR10_CLASSES, f"{split_choice} — Class Distribution"),
            use_container_width=True,
        )

    st.divider()

    # ── Sample Inspection ─────────────────────────────────────────────────────
    st.subheader("Sample Inspection")
    ins_col1, ins_col2, ins_col3 = st.columns(3)
    with ins_col1:
        split_sel = st.selectbox("Select split", ["Train", "Val", "Test"], key="ins_split")
    with ins_col2:
        class_filter = st.selectbox("Filter by class", ["All"] + CIFAR10_CLASSES, key="ins_class")
    with ins_col3:
        pass  # index slider depends on filtered df — computed below

    df_sel = df_map[split_sel].copy().reset_index(drop=True)
    if class_filter != "All":
        class_idx = CIFAR10_CLASSES.index(class_filter)
        df_sel = df_sel[df_sel["label"] == class_idx].reset_index(drop=True)

    if df_sel.empty:
        st.warning("No samples match the selected filter.")
    else:
        sample_idx = st.slider(
            "Sample index", 0, len(df_sel) - 1, 0, key="ins_idx",
            help=f"{len(df_sel)} samples match your filter.",
        )
        row = df_sel.iloc[sample_idx]
        img_arr = get_image_for_display(row["image_path"])
        label_name = CIFAR10_CLASSES[int(row["label"])]

        img_col, info_col = st.columns([1, 2])
        with img_col:
            # Upscale tiny 32×32 for display
            from PIL import Image
            display_img = Image.fromarray(img_arr).resize((192, 192), Image.NEAREST)
            st.image(display_img, caption=f"Label: {label_name}", use_container_width=False)
        with info_col:
            st.markdown(f"**Split:** {split_sel}")
            st.markdown(f"**True label:** `{label_name}` (class {int(row['label'])})")
            st.markdown(f"**Image path:** `{row['image_path']}`")
            st.markdown(f"**Image size:** 32 × 32 × 3 (CIFAR-10)")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_errors:
    st.header("Error Analysis")

    # ── MLflow run selection ──────────────────────────────────────────────────
    st.subheader("MLflow Experiment Selection")
    exp_names = get_experiment_names(tracking_uri=MLFLOW_URI)
    if not exp_names:
        st.error("No MLflow experiments found. Run `python scripts/setup_mlflow.py` first.")
        st.stop()

    exp_name = st.selectbox("Experiment", exp_names, index=0, key="err_exp")
    runs_df = _load_runs(exp_name)

    if runs_df.empty:
        st.warning(f"No runs found in experiment '{exp_name}'.")
    else:
        run_labels = {
            row["run_name"]: row["run_id"]
            for _, row in runs_df.iterrows()
        }
        selected_run_name = st.selectbox("Run", list(run_labels.keys()), key="err_run")
        selected_run_id = run_labels[selected_run_name]

        # Show run metrics table
        metric_cols = [c for c in runs_df.columns if c.startswith("metric.")]
        param_cols  = [c for c in runs_df.columns if c.startswith("param.")]
        run_row = runs_df[runs_df["run_id"] == selected_run_id].iloc[0]

        with st.expander("Run details", expanded=False):
            col_m, col_p = st.columns(2)
            with col_m:
                st.markdown("**Metrics**")
                st.dataframe(
                    run_row[metric_cols].rename(lambda c: c.replace("metric.", "")).to_frame("value"),
                    use_container_width=True,
                )
            with col_p:
                st.markdown("**Parameters**")
                st.dataframe(
                    run_row[param_cols].rename(lambda c: c.replace("param.", "")).to_frame("value"),
                    use_container_width=True,
                )

        # Show loss curves if available
        train_hist = get_run_metric_history(selected_run_id, "train_loss", MLFLOW_URI)
        val_hist   = get_run_metric_history(selected_run_id, "val_loss", MLFLOW_URI)
        if train_hist and val_hist:
            import plotly.graph_objects as go_mod
            fig_loss = go_mod.Figure()
            fig_loss.add_trace(go_mod.Scatter(y=train_hist, name="Train Loss", mode="lines+markers"))
            fig_loss.add_trace(go_mod.Scatter(y=val_hist,   name="Val Loss",   mode="lines+markers"))
            fig_loss.update_layout(title="Training Curves", xaxis_title="Epoch", yaxis_title="Loss", height=300)
            st.plotly_chart(fig_loss, use_container_width=True)

        st.divider()

        # ── Error extraction ──────────────────────────────────────────────────
        st.subheader("Error Analysis")
        if st.button("▶ Run Error Analysis on Test Split", key="run_err_btn"):
            model = _load_run_model(selected_run_id)
            if model is None:
                st.error("Could not load model artifact from this run.")
            else:
                _, _, test_df_e = _load_splits()
                with st.spinner("Running inference on test split…"):
                    y_true, y_pred, y_conf = run_inference_batch(
                        model, test_df_e, batch_size=BATCH_SIZE, device=DEVICE
                    )
                st.session_state["err_results"] = {
                    "y_true": y_true, "y_pred": y_pred, "y_conf": y_conf,
                    "test_df": test_df_e.reset_index(drop=True),
                }
                acc = (y_true == y_pred).mean()
                st.success(f"Inference complete — Test accuracy: {acc:.2%}")

        if "err_results" in st.session_state:
            res = st.session_state["err_results"]
            y_true = res["y_true"]
            y_pred = res["y_pred"]
            y_conf = res["y_conf"]
            test_df_r = res["test_df"]

            cm_col, err_col = st.columns(2)
            with cm_col:
                st.pyplot(plot_confusion_matrix(y_true, y_pred, CIFAR10_CLASSES), use_container_width=True)
            with err_col:
                st.plotly_chart(plot_per_class_errors(y_true, y_pred, CIFAR10_CLASSES), use_container_width=True)

            # ── Browse misclassified samples ──────────────────────────────────
            st.subheader("Misclassified Samples")
            error_mask = y_true != y_pred
            err_indices = np.where(error_mask)[0]

            if len(err_indices) == 0:
                st.success("No misclassifications found!")
            else:
                sort_by = st.selectbox(
                    "Sort errors by", ["confidence (high→low)", "predicted class", "true class"],
                    key="err_sort"
                )
                if sort_by == "confidence (high→low)":
                    order = np.argsort(-y_conf[err_indices])
                elif sort_by == "predicted class":
                    order = np.argsort(y_pred[err_indices])
                else:
                    order = np.argsort(y_true[err_indices])
                err_indices_sorted = err_indices[order]

                err_page = st.slider("Error sample index", 0, len(err_indices_sorted) - 1, 0, key="err_page")
                sample_i = err_indices_sorted[err_page]

                row = test_df_r.iloc[sample_i]
                img_arr = get_image_for_display(row["image_path"])
                from PIL import Image as _PIL
                disp = _PIL.fromarray(img_arr).resize((160, 160), _PIL.NEAREST)

                e_col1, e_col2 = st.columns([1, 2])
                with e_col1:
                    st.image(disp, use_container_width=False)
                with e_col2:
                    st.markdown(f"**True label:** `{CIFAR10_CLASSES[y_true[sample_i]]}` (#{y_true[sample_i]})")
                    st.markdown(f"**Predicted:**  `{CIFAR10_CLASSES[y_pred[sample_i]]}` (#{y_pred[sample_i]})")
                    st.markdown(f"**Confidence:** `{y_conf[sample_i]:.2%}`")
                    st.markdown(f"**Error #{err_page + 1} of {len(err_indices_sorted)}**")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Prediction & Explainability
# ─────────────────────────────────────────────────────────────────────────────
with tab_explain:
    st.header("Prediction & Explainability")

    # ── Model selection ───────────────────────────────────────────────────────
    exp_names_e = get_experiment_names(tracking_uri=MLFLOW_URI)
    runs_df_e = _load_runs(exp_names_e[0]) if exp_names_e else None

    use_mlflow_model = False
    selected_run_id_e = None
    if runs_df_e is not None and not runs_df_e.empty:
        run_labels_e = {row["run_name"]: row["run_id"] for _, row in runs_df_e.iterrows()}
        model_source = st.radio(
            "Model source",
            ["Default checkpoint (outputs/lab04_main.pth)", "MLflow run"],
            horizontal=True,
            key="exp_src",
        )
        if model_source == "MLflow run":
            use_mlflow_model = True
            selected_run_name_e = st.selectbox("Run", list(run_labels_e.keys()), key="exp_run")
            selected_run_id_e = run_labels_e[selected_run_name_e]
    else:
        st.info("Using default checkpoint (no MLflow runs available).")

    if use_mlflow_model and selected_run_id_e:
        model_exp = _load_run_model(selected_run_id_e)
        if model_exp is None:
            st.warning("Could not load from run — falling back to default checkpoint.")
            model_exp = _load_default_model()
    else:
        model_exp = _load_default_model()

    st.divider()

    # ── Dataset sample inference ──────────────────────────────────────────────
    st.subheader("Inference on Dataset Sample")
    try:
        train_df_e, val_df_e, test_df_e2 = _load_splits()
    except Exception as exc:
        st.error(f"Dataset load failed: {exc}")
        st.stop()

    df_map_e = {"Train": train_df_e, "Val": val_df_e, "Test": test_df_e2}

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        split_e = st.selectbox("Split", ["Train", "Val", "Test"], key="exp_split")
    with s_col2:
        class_filter_e = st.selectbox("Filter class", ["All"] + CIFAR10_CLASSES, key="exp_cls")

    df_e = df_map_e[split_e].reset_index(drop=True)
    if class_filter_e != "All":
        cls_idx_e = CIFAR10_CLASSES.index(class_filter_e)
        df_e = df_e[df_e["label"] == cls_idx_e].reset_index(drop=True)

    if df_e.empty:
        st.warning("No samples match the filter.")
    else:
        with s_col3:
            sample_idx_e = st.slider("Sample index", 0, len(df_e) - 1, 0, key="exp_idx")

        row_e = df_e.iloc[sample_idx_e]
        img_arr_e = get_image_for_display(row_e["image_path"])
        true_label_e = CIFAR10_CLASSES[int(row_e["label"])]

        from src.data.dataset import eval_transform
        from torchvision.io import read_image as _ri
        img_tensor_e = eval_transform(_ri(row_e["image_path"]).float() / 255.0)

        pred_e, probs_e = run_inference(model_exp, img_tensor_e, device=DEVICE)

        from PIL import Image as _PIL2
        disp_e = _PIL2.fromarray(img_arr_e).resize((160, 160), _PIL2.NEAREST)

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.image(disp_e, caption=f"True: {true_label_e}", use_container_width=False)
        with p_col2:
            pred_name_e = CIFAR10_CLASSES[pred_e]
            correct = "✅" if pred_e == int(row_e["label"]) else "❌"
            st.markdown(f"**Predicted:** `{pred_name_e}` {correct}  (confidence {probs_e[pred_e]:.2%})")
            st.plotly_chart(plot_probability_distribution(probs_e, CIFAR10_CLASSES), use_container_width=True)

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Grad-CAM Explainability")
        explain_class_options = ["Top prediction"] + CIFAR10_CLASSES
        explain_choice = st.selectbox("Class to explain", explain_class_options, key="exp_cls_explain")
        target_cls = None if explain_choice == "Top prediction" else CIFAR10_CLASSES.index(explain_choice)

        if st.button("🔥 Generate Grad-CAM", key="gradcam_btn"):
            with st.spinner("Computing Grad-CAM…"):
                cam = compute_gradcam(model_exp, img_tensor_e, target_class=target_cls, device=DEVICE)
            effective_class = target_cls if target_cls is not None else pred_e
            st.caption(f"Explaining class: **{CIFAR10_CLASSES[effective_class]}**")
            st.pyplot(plot_gradcam_overlay(img_arr_e, cam), use_container_width=True)

    st.divider()

    # ── Upload & inference ────────────────────────────────────────────────────
    st.subheader("Inference on Uploaded Image")
    uploaded = st.file_uploader(
        "Upload an image (JPG/PNG) for classification",
        type=["jpg", "jpeg", "png"],
        key="upload_file",
    )
    if uploaded is not None:
        from PIL import Image as _PIL3
        try:
            pil_img = _PIL3.open(uploaded).convert("RGB")
            img_tensor_up = preprocess_uploaded_image(pil_img)
            pred_up, probs_up = run_inference(model_exp, img_tensor_up, device=DEVICE)
            pred_name_up = CIFAR10_CLASSES[pred_up]

            up_col1, up_col2 = st.columns([1, 2])
            with up_col1:
                st.image(pil_img.resize((160, 160), _PIL3.BILINEAR), caption="Uploaded image")
            with up_col2:
                st.markdown(f"**Predicted class:** `{pred_name_up}` ({probs_up[pred_up]:.2%})")
                st.plotly_chart(
                    plot_probability_distribution(probs_up, CIFAR10_CLASSES),
                    use_container_width=True,
                )

            # Grad-CAM for uploaded image
            if st.button("🔥 Grad-CAM for uploaded image", key="gradcam_upload"):
                explain_cls_up_opt = st.selectbox(
                    "Class to explain (upload)", ["Top prediction"] + CIFAR10_CLASSES,
                    key="exp_cls_upload"
                )
                target_up = None if explain_cls_up_opt == "Top prediction" else CIFAR10_CLASSES.index(explain_cls_up_opt)
                with st.spinner("Computing Grad-CAM…"):
                    cam_up = compute_gradcam(model_exp, img_tensor_up, target_class=target_up, device=DEVICE)
                img_arr_up = np.array(pil_img.resize((32, 32), _PIL3.BILINEAR))
                st.pyplot(plot_gradcam_overlay(img_arr_up, cam_up), use_container_width=True)

        except Exception as exc:
            st.error(f"Failed to process uploaded image: {exc}")
            logger.error("Upload inference error: %s", exc)
