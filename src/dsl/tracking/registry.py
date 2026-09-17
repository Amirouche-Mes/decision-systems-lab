import os
import logging
import mlflow 
from dsl.run_expirement import run_expirement
from dsl.models.classification import DataSplits

logger = logging.getLogger(__name__)

def log_experiment(
        cfg: dict,
        splits: DataSplits,
        schema: dict[str, list[str]],
        tracking_uri: str | None = None,
)-> str:
    """Logs the training experiment in the MLflow registry.

    Args:
        cfg: All modeling and experiment configurations.
        splits: DataSplits instance containing train, val, and optional test sets.
        feature_cols: Cleaned schema dictionary mapping feature types to column names.
        tracking_uri: MLflow db path.
        
    Returns:
        str: The active MLflow run_id.
    """
    mlflow.set_tracking_uri(
            tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        )
    exp_name = cfg["tracking"]["name"]
    logger.info(f"Setting MLflow experiment to: '{exp_name}'")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=cfg["experiment"]["name"]) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started with ID: {run_id}")

        res = run_expirement(
                    name=exp_name,
                    model_name=cfg["model"]["name"],
                    splits=splits,
                    feature_cols=schema,
                    model_params=cfg["model"]["params"],
                    include_test=cfg["experiment"]["include_test"],
                )
        
        mlflow.set_tags(res["tags"])
        mlflow.log_params(res["params"])
        mlflow.log_metrics(res["metrics"])
        mlflow.sklearn.log_model(
            sk_model=res["model"],
            name="model",                          
            serialization_format="cloudpickle",
            input_example=splits.Xva.head(3),  
        )

    logger.info(f"Experiment logged successfully under Run ID: {run_id}")
    return run_id
