"""Utilities for querying the MLflow tracking server."""
from typing import Any, Dict, List, Optional

import pandas as pd


def get_mlflow_runs(
    experiment_name: str,
    tracking_uri: str = "outputs/mlruns",
) -> pd.DataFrame:
    """Return a DataFrame of all runs for the given MLflow experiment.

    Args:
    - experiment_name: Name of the MLflow experiment.
    - tracking_uri: Path or URI to the MLflow tracking server.

    Returns:
    - DataFrame with run params, metrics, and tags columns.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    records = []
    for run in runs:
        row: Dict[str, Any] = {"run_id": run.info.run_id, "status": run.info.status}
        row.update({f"param.{k}": v for k, v in run.data.params.items()})
        row.update({f"metric.{k}": v for k, v in run.data.metrics.items()})
        records.append(row)

    return pd.DataFrame(records)


def get_run_artifacts(run_id: str, tracking_uri: str = "outputs/mlruns") -> List[str]:
    """List artifact paths for a given run."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    return [a.path for a in artifacts]
