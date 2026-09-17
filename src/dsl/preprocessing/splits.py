import logging
import pandas as pd
from dsl.models.classification import DataSplits

logger = logging.getLogger(__name__)

def temporal_split(
        df: pd.DataFrame,
        schema: dict,
        split_col: str | None = None,
        low_pct: float = 0.7,
        high_pct: float = 0.85,
    ) -> DataSplits:
    """ Separate a DataFrame in three datasets (train, valid, test) using a temporal base column.

        Args:
            df: source DF to split.
            split_col: Temporal/numeric column name to use in the split
            low_pct: low percentil for the limit train valid sets.
            high_pct: high percentil for the limit valid test sets.

        Returns:
            A tuple (train, valid, test) of dataframes.
        
        Raises:
            ValueError: if the DF is empty or the percentiles are invalid.
            KeyError: if 'split_col' or 'target_col' aren't in the DF.
    
    """
    target_col = schema["target"]
    feature_cols = schema["ids"] + schema["num"] + schema["cat"]
    split_col = split_col or schema["timestamp"]
    if df.empty:
        raise ValueError("The entry DF for the temporal split is empty.")

    if not 0 <= low_pct < high_pct <= 1:
        raise ValueError(
            f" The percentiles must respect : 0 <= low_pct < high_pct <= 1."
            f" Received: low_pct={low_pct}, high_pct={high_pct}"
        )


    for col in (split_col, target_col):
        if col not in df.columns:
            raise KeyError(f"The column {col} is not in the DF.")

    # Get the temporal limites using quantiles
    c1 = df[split_col].quantile(low_pct)
    c2 = df[split_col].quantile(high_pct)

    train = df[df[split_col] <= c1].copy()
    valid = df[(df[split_col] > c1) & (df[split_col] <= c2)].copy()
    test = df[df[split_col] > c2].copy()

    # Logging metrics of each split 
    for name, subset in [("train", train), ("valid", valid), ("test", test)]:
        if len(subset) == 0: 
            logger.warning("The split %s is empty", name)
            continue 

        min_date = pd.to_datetime(subset[split_col].min()).strftime("%Y-%m-%d")
        max_date = pd.to_datetime(subset[split_col].max()).strftime("%Y-%m-%d")
        rate = subset[target_col].mean()

        logger.info(
            "%5s: %6d rows | %s -> %s | rate target: %.4f",
            name, 
            len(subset),
            min_date,
            max_date,
            rate,
        )
    return DataSplits(
        Xtr=train[feature_cols], ytr=train[target_col],
        Xva=valid[feature_cols], yva=valid[target_col],
        Xte=test[feature_cols], yte=test[target_col],
    )