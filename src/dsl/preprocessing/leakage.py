
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)

def detect_data_leakage(
        df: pd.DataFrame,
        num_cols: list[str],
        target_col: str = "converted",
        auc_threshold: float = 0.40,
        lift_threshold: float = 3.0,
        min_sample_size: int = 30,
    ) -> list[str]:
    """ Identifies candidates features showing signs of target/ data leakage.

    Detection operates via two distinct heuristics:
    1. Null Pattern Leakage: checks if missingness correlates strongly with target.
    2. Univariate AUC Leakage: Checks for extreme predictive power abs(auc - 0.5) >= Threshold AUC

    Args:
        df: Input pandas DataFrame.
        num_cols: List of numeric column names to evaluate via AUC.
        target_col: Name of the binary target column.
        auc_threshold: Minimum absolute deviation from 0.5 to flag an AUC as suspicious.
        lift_threshold: Maximum allowed conversion rate ratio for missing value patterns.
        min_sample_size: Minimum sample count per subset (null vs non-null) to evaluate leakage.

    Returns:
        Stored list of feature column names suspected of data leakage.
    
    Raises:
        KeyError: if 'target_col' is missing from df.
    
    """

    if target_col not in df.columns:
        raise KeyError(f" Target column '{target_col}' is missing from the DF")

    target = df[target_col]
    base_rate = target.mean()

    if base_rate == 0:
        logger.warning("Global target rate is 0. Data leakage detection aborder")
        return []

    suspect_null_cols: set[str] = set()
    suspect_auc_cols: set[str] = set()

    for col in df.columns:
        if col == target_col:
            continue

        # 1. Evaluate missingness patterns for potential leakage
        null_mask = df[col].isna()
        grouped_target = target.groupby(null_mask)
        rates = grouped_target.mean()
        sizes = grouped_target.size()

        if len(rates) == 2 and sizes.min() >= min_sample_size:
            r_null = rates.get(True, 0.0)
            r_not_null = rates.get(False, 0.0)

            # Flag perfect deterministic leakage (0% or 100% convertion)
            is_deterministic_leak = (r_null in (0, 1)) or (r_not_null in (0, 1))

            lift_null = r_null / base_rate
            lift_not_null = r_not_null / base_rate 
            max_lift = max(lift_null, lift_not_null)

            if is_deterministic_leak or max_lift > lift_threshold:
                logger.warning(
                    "Null pattern leakage detected | Feature: %s | Max Lift: %.2f",
                    col,
                    max_lift,
                )
                suspect_null_cols.add(col)

        # 2. Evaluate univariate predictive performance (AUC) for numerical features
        if col in num_cols:
            # Handle Nans determiniscally using average rank placement
            ranked_feature = df[col].rank(method="average", na_option="bottom")
            auc = roc_auc_score(target, ranked_feature)
            auc_delta = abs(auc - 0.5)

            logger.debug("%22s | Univariate AUC: %.3f", col, auc)

            if auc_delta >= auc_threshold:
                logger.warning(
                    "Extreme AUC detected | Feature: %s | AUC: %.3f", col, auc
                )
                suspect_auc_cols.add(col)

    return sorted(list(suspect_null_cols | suspect_auc_cols))

def compute_iv_auc(
        df: pd.DataFrame,
        cat_col: str,
        target_col: str = "converted",
        cv_splits: int = 5,
        random_state: int = 42,
) -> pd.Series:
    """ Computes Information value (IV) and out-of-fold for categorical features

    Args:
        df: input pandas dataframe.
        cat_col: name of the categorical column to evaluate.
        target_col: name of the binary target column.
        cv_splits: number of cross-validation folds for model evaluation.
        random_state: Seed for reproducibility.
    
    Returns:
        Pandas Series containing the computed 'IV' and 'AUC' metrics.
    
    Raises:
        KeyError: if 'cat_col' or 'target_col' are missing from 'df'
            
    """

    for col in (cat_col, target_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is missing from the DF.")

    # 1. Compute information value (IV)
    categorical_series = df[cat_col].astype("category")
    stats = df.groupby(categorical_series, observed=False)[target_col].agg(
        goods="sum",
        total="count"
    )
    stats["bads"] = stats["total"] - stats["goods"]

    total_goods = stats["goods"].sum()
    total_bads = stats["bads"].sum()

    if total_goods == 0 or total_bads == 0:
        logger.warning("Target has only one active class. IV computation aborted")
        return pd.Series({"IV": np.nan, "AUC": 0.5})

    # apply numerical clipping to prevent zero and log(0)
    eps = 1e-6
    prop_goods = np.clip(stats["goods"] / total_goods, eps, 1.0)
    prop_bads = np.clip(stats["bads"] / total_bads, eps, 1.0)

    woe = np.log(prop_goods / prop_bads)
    iv = float(np.sum((prop_goods - prop_bads) * woe))

    # 2. compute cross-validation AUC using native categorical gradient boosting
    model = HistGradientBoostingClassifier(
        max_iter=50,
        categorical_features=[0],
        random_state=random_state,
    )
    feature_matrix = categorical_series.to_frame()

    out_of_fold_probs = cross_val_predict(
        model,
        feature_matrix,
        df[target_col],
        cv=cv_splits,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    auc = float(roc_auc_score(df[target_col], out_of_fold_probs))

    return pd.Series({"IV": iv, "AUC": auc})
