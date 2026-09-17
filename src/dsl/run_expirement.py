import logging 
from typing import Any

from dsl.models.classification import DataSplits, BUILDERS
from dsl.evaluation import evaluate_binary_classifier

logger = logging.getLogger(__name__)

def run_expirement(
        name: str,
        model_name: str, 
        splits: DataSplits, 
        feature_cols: dict[str, list[str]],
        model_params: dict[str, Any] | None = None,
        include_test: bool = False,
) -> dict[str, Any]:
    """Runs a single ML training and evaluation experiment.

    Args:
        name: name identifier for the experiment run.
        model_name: Model architecture choice ('lgb' or 'lgr' for now).
        splits: DataSplits object containing training, validation end test splits.
        feature_cols: Dictionary mapping feature tier names to column lists:
            - For 'lgb' and 'lgr': requires 'cat' and 'num' keys 
        model_params: Hyperparameter overrides passed directly to the model trainer.
        include_test: a flag that saying if the dataclass contains test set or not.

    returns:
        Dictionary containing experiment metadata, evaluation metrics, and hyperparams
    
    Raises:
        ValueError: if 'model_name' is unsupported or required feature keys are missing.
    """
    model_params = model_params or {}
    logger.info("Starting experiment: '%s' using model: '%s'", name, model_name)

    builder = BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Allowed: {sorted(BUILDERS)}."
        )

    num_cols = feature_cols.get("num", [])
    cat_cols = feature_cols.get("cat", [])

    total_features = len(cat_cols) + len(num_cols)
    if total_features == 0:
        raise ValueError(
            "At least one numerical ('num') or categorical ('cat') feature must be provided for LightGBM."
        )
    # 1. Model training Dispatch
    model = builder(splits=splits, cat_cols=cat_cols, num_cols=num_cols, hyperparams=model_params)

    # 2. Collect Metrics
    metrics: dict[str, float] = {}

    if hasattr(splits, "Xva") and splits.Xva is not None and len(splits.Xva) > 0:
        val_metrics = evaluate_binary_classifier(
            model=model, X=splits.Xva, y=splits.yva, dataset_name="valid"
        )
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

    if include_test and splits.Xte is not None and len(splits.Xte) > 0:
            test_metrics = evaluate_binary_classifier(
                model=model, X=splits.Xte, y=splits.yte, dataset_name="test"
            )
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # 3. Consolidate Parameters and Tags
    params = {
         **model_params,
         "n_features_num": len(num_cols),
         "n_features_cat": len(cat_cols),
         "total_features": total_features,
    }

    tags = {
         "experiment_name": name,
         "model_type": model_name,
    }

    logger.info("Completed experiment '%s' successfully.", name)
    return {
         "tags": tags,
         "params": params,
         "metrics": metrics,
         "model": model,
    }

    