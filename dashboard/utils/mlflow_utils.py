"""Utilities for querying the MLflow tracking server."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_client(tracking_uri: str):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient()


def get_experiment_names(tracking_uri: str = "outputs/mlruns") -> List[str]:
    """Return list of all experiment names on the tracking server."""
    client = _get_client(tracking_uri)
    try:
        exps = client.search_experiments()
        return [e.name for e in exps]
    except Exception as exc:
        logger.error("Failed to list experiments: %s", exc)
        return []


def get_mlflow_runs(
    experiment_name: str,
    tracking_uri: str = "outputs/mlruns",
) -> pd.DataFrame:
    """Return a DataFrame of all runs for the given experiment."""
    client = _get_client(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment '%s' not found.", experiment_name)
        return pd.DataFrame()

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    records = []
    for run in runs:
        row: Dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            "status": run.info.status,
        }
        row.update({f"param.{k}": v for k, v in run.data.params.items()})
        row.update({f"metric.{k}": v for k, v in run.data.metrics.items()})
        records.append(row)

    logger.info("Retrieved %d runs for experiment '%s'.", len(records), experiment_name)
    return pd.DataFrame(records)


def get_run_artifacts(run_id: str, tracking_uri: str = "outputs/mlruns") -> List[str]:
    """List artifact paths for a given run."""
    client = _get_client(tracking_uri)
    try:
        artifacts = client.list_artifacts(run_id)
        return [a.path for a in artifacts]
    except Exception as exc:
        logger.error("Failed to list artifacts for run %s: %s", run_id, exc)
        return []


def get_run_metric_history(
    run_id: str,
    metric_name: str,
    tracking_uri: str = "outputs/mlruns",
) -> List[float]:
    """Return per-step values for a metric in a specific run."""
    client = _get_client(tracking_uri)
    try:
        history = client.get_metric_history(run_id, metric_name)
        return [m.value for m in sorted(history, key=lambda m: m.step)]
    except Exception as exc:
        logger.error("Failed to get metric history: %s", exc)
        return []


def load_model_from_run(
    run_id: str,
    tracking_uri: str = "outputs/mlruns",
    n_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Optional[nn.Module]:
    """Download the model artifact from a run and return loaded CifarCNN."""
    import sys
    import mlflow
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.cnn import CifarCNN

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client = _get_client(tracking_uri)
    artifacts = client.list_artifacts(run_id, path="model")
    if not artifacts:
        logger.warning("No model artifacts found for run %s", run_id)
        return None

    pth_artifacts = [a for a in artifacts if a.path.endswith(".pth")]
    if not pth_artifacts:
        logger.warning("No .pth artifact in model/ for run %s", run_id)
        return None

    artifact_uri = f"runs:/{run_id}/{pth_artifacts[0].path}"
    logger.info("Downloading artifact: %s", artifact_uri)
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        model = CifarCNN(n_classes=n_classes)
        model.load_state_dict(torch.load(local_path, map_location=device))
        model.eval()
        model.to(device)
        logger.info("Model loaded from run %s to %s", run_id, device)
        return model
    except Exception as exc:
        logger.error("Failed to load model from run %s: %s", run_id, exc)
        return None
