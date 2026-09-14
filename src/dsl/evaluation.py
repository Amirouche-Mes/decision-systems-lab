# src/dsl/evaluation.py
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_binary_classifier(
        model: Any,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        dataset_name: str = "eval",
) -> dict[str, float]:
    """Evaluates a binary classification model on performance and calibration metrics.

    Args:
        model: Fitted estimator or Pipeline implementing 'predict_proba'.
        X: Feature matrix.
        y: True binary target vector.
        dataset_name: Name of the evaluation partition for logging (e.g., 'valid', 'test').

    Returns:
        Dictionary containing evaluation metrics:
            - real_rate: actual positive class prevalence.
            - p_mean: average predicted probability.
            - calibration_ratio: ratio of average prediction to actual rate (ideal: 1.0).
            - brier: Brier score loss.
            - log_loss: cross-entropy log loss.
            - auc: area under the ROC curve (NaN if only one class present).

    Raises:
        ValueError: if 'X' or 'y' are empty.
        AttributeError: if model does not implement 'predict_proba'.
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Inputs 'X' and 'y' must not be empty.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError("The provided model does not implement 'predict_proba'.")

    probabilities = model.predict_proba(X)[:, 1]
    y_true = np.asarray(y)

    real_rate = float(np.mean(y_true))
    p_mean = float(np.mean(probabilities))
    calibration_ratio = p_mean / real_rate if real_rate > 0 else np.nan

    brier = float(brier_score_loss(y_true, probabilities))
    loss = float(log_loss(y_true, probabilities, labels=[0, 1]))

    # Guard against single-class evaluation target for ROC AUC
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, probabilities))
    else:
        logger.warning(
            "Evaluation set '%s' contains only one class. Setting AUC to NaN.",
            dataset_name,
        )
        auc = np.nan

    metrics = {
        "real_rate": real_rate,
        "p_mean": p_mean,
        "calibration_ratio": calibration_ratio,
        "brier": brier,
        "log_loss": loss,
        "auc": auc,
    }

    logger.info(
        "[%s] AUC: %.4f | Log Loss: %.4f | Calibration Ratio: %.4f",
        dataset_name,
        metrics["auc"],
        metrics["log_loss"],
        metrics["calibration_ratio"],
    )

    return metrics