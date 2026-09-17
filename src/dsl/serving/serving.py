import os
import logging
import mlflow 
from typing import Any

logger = logging.getLogger(__name__)


def load_best_model(
        experiment_name: str,
        metric: str = "metrics.val_auc",
        tracking_uri: str | None = None,
)-> tuple[Any, str]:
    """Loads the best evaluated model for an experiment in MLflow.

    Args:
        experiment_name: Name of the experiment registry.
        metric: Metric name to sort by (e.g., 'val_auc' or 'metrics.val_auc').
        tracking_uri: MLflow db path.

    Returns:
        Tuple of (Loaded Model object, Run ID string).

    Raises:
        ValueError: If no runs are found for the given experiment name.
    """
    mlflow.set_tracking_uri(
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    formatted_metric = metric if metric.startswith("metrics.") else f"metrics.{metric}"

    logger.info(f"Searching for best model in '{experiment_name}' ordered by {formatted_metric}...")

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"{formatted_metric} DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    best_run_id = str(runs.iloc[0]["run_id"])
    best_metric = float(runs.iloc[0][formatted_metric])
    best_model_name = runs.iloc[0].get("tags.model_type", "unknown")

    logger.info(
        f"Champion model identified: '{best_model_name}' | "
        f"Run ID: {best_run_id} | {metric}: {best_metric:.4f}"
    )

    # Chargement de l'artefact modèle
    model_uri = f"runs:/{best_run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    return loaded_model, best_run_id