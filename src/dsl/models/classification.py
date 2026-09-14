import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder, FunctionTransformer


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DataSplits:
    """Container holding features matrices and target vectors for training and validation"""

    Xtr: pd.DataFrame
    ytr: pd.Series
    Xva: pd.DataFrame | None = None
    yva: pd.Series | None = None
    Xte: pd.DataFrame | None = None
    yte: pd.Series | None = None
    

def train_lgbm(
        splits: DataSplits,
        cat_cols: list[str],
        num_cols: list[str],
        hyperparams: dict[str, Any] | None = None,
) -> Pipeline:
    """Trains a LightGBM classifier with native categorical feature support.

    Args:
        splits: DataSplits dataclass containing training features and targets.
        cat_cols: List of categorical feature column names.
        num_cols: List of numerical feature column names.
        hyperparams: Optional keyword argument dicrionary overriding default LGBM params.
    
    Returns:
        Fitted Scikit-Learn Pipeline containing feature casting LGBMClassifier.
    
    Raises:
        KeyError: if required features are missing from input DF
    """
    selected_features = cat_cols + num_cols
    for col in selected_features:
        if col not in splits.Xtr.columns:
            raise KeyError(f"Feature '{col}' is missing from training DF")

    default_params: dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "verbose": -1,
        "num_leaves": 31, 
        "min_child_samples": 50,
        "random_state": 0,
        "n_jobs": -1,
    }

    if hyperparams:
        default_params.update(hyperparams)

    # Convert categorical features to Pandas 'category' dtype directly on feature 
    cat_dtypes = {
        c: pd.CategoricalDtype(categories=splits.Xtr[c].dropna().unique())
        for c in cat_cols
    }
    def cast_and_select(df: pd.DataFrame) -> pd.DataFrame:
        out = df[selected_features].copy()
        for c, dt in cat_dtypes.items():
            out[c] = out[c].astype(dt)  # modalité inconnue -> NaN, géré nativement
        return out

    pipeline = Pipeline(steps=[
        ("prep", FunctionTransformer(cast_and_select)),
        ("classifier", lgb.LGBMClassifier(**default_params)),
    ])

    logger.info(
        "Training LightGBM on %d samples with %d features (%d categorical, %d numerical).",
        len(splits.Xtr),
        len(selected_features),
        len(cat_cols),
        len(num_cols)
    )

    # LightGBM handles pandas 'category' natively 
    pipeline.fit(splits.Xtr, splits.ytr)

    return pipeline

def train_lgr(
        splits: DataSplits,
        num_cols: list[str], 
        cat_cols: list[str],
        hyperparams: dict[str, Any] | None = None,
        max_ohe_cardinality: int = 50,
) -> Pipeline:
    """ Trains a logistic Regression Classifier using a Sckit Learn processing pipeline

    Preprocessing includes:
    - Numerical features: standard scaling.
    - Low-cardinality categorical features: One-hot Encoding.
    - High-cardinality categorical features: Target Encoding with out-of-fold CV.

    Args:
        splits: Datasplits dataclass containing training features and targets.
        num_cols: List of numerical feature column names.
        cat_cols: List of categorical feature column names.
        hyperparams: Optional keyword argment dictionary overriding default LGR params.
    
    Returns:
        Fitted Scikit-Learn Pipeline containing ColumnTranformer and LogisticRegression model.
    
    Raises:
        KeyErro: if required features are missing from input DataFrames.
    """
    selected_features = num_cols + cat_cols
    for col in selected_features:
        if col not in splits.Xtr.columns:
            raise KeyError(f"Feature '{col}' is missing from training DF")

    default_params: dict[str, Any] = {
            "C":1.0,
            "max_iter": 1000,
            "random_state": 0,
            "solver": "lbfgs",
            "n_jobs": -1,
        }
    
    if hyperparams:
        default_params.update(hyperparams)

    # cardinality route, measured on training set only
    cardinalities = splits.Xtr[cat_cols].nunique()
    low_card = [c for c in cat_cols if cardinalities[c] <= max_ohe_cardinality]
    high_card = [c for c in cat_cols if cardinalities[c] > max_ohe_cardinality]

    logger.info(
        "Cardinality routing: %d on-hot %s, %d target-encoded %s",
        len(low_card), low_card, len(high_card), high_card,
    )

    # 1. Numerical Features
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if low_card:
        transformers.append(
            ("low_card", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card)
        )
    if high_card:
        transformers.append(
            ("high_card", TargetEncoder(smooth="auto", cv=5, random_state=0), high_card)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**default_params)),
        ]
    )

    logger.info(
        "Training Logistic Regression: %d num, %d low-card, %d high-card",
        len(num_cols),
        len(low_card),
        len(high_card),
    )

    pipeline.fit(splits.Xtr, splits.ytr)

    return pipeline

BUILDERS = {
    "lgbm": train_lgbm,
    "lgr": train_lgr,
}