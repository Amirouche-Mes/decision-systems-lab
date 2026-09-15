import logging 
import numpy as np
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)

def add_target_encoding(
        df: pl.DataFrame,
        group_col: str = "partner_id",
        target_col: str = "converted",
        timestamp_col: str = "click_ts",
        prior_weight: float = 20.0,
        out_col_name: str = "partner_rate",
) -> tuple[pl.DataFrame, list[str]]:
    """ Computes a fully expanding out-of-target encoding with bayesian smoothing
    
    Guarantees zera data leakage by computin both the group cumulative mean and the 
    global expanding prior using strictly past rows.

    Args:
        df: Polars Dataframe
        group_col: categorical column to encode.
        target_col: Target binary column.
        timestamp_col: Timestamp column used for explicit chronological sorting.
        prior_weight: smoothing weight (m-estimate parameter).
        out_col_name: output feature column name.
    
    Returns:
        Tupe containing:
            - Enriched Polars DF preserving target_col.
            - List of new numerical column names created.
    """
    # Chrono ordering
    df_sorted = df.sort(timestamp_col)

    # Global expanding prior
    global_prior_expr = (
        (pl.col(target_col).cum_sum() - pl.col(target_col))
        / (pl.col(target_col).cum_count() - 1)
    ).fill_nan(None).fill_null(0.5)

    # Group expanding target encoding
    group_cumsum_past = pl.col(target_col).cum_sum().over(group_col) - pl.col(target_col)
    group_cumcount_past = pl.col(target_col).cum_count().over(group_col) - 1

    target_encding_expr = (
        (group_cumsum_past + (global_prior_expr * prior_weight))
        / (group_cumcount_past + prior_weight)
    ).alias(out_col_name)

    return df_sorted.with_columns(target_encding_expr), [out_col_name]

def add_temporal_delta_features(
        df: pl.DataFrame,
        group_cols: list[str] | None = None,
        timestamp_col: str = "click_ts",
) -> tuple[pl.DataFrame, list[str]]:
    """Calculates time deltas (seconds) since previous events per entity group.

        Args:
            df: Input Polars DataFrame.
            group_cols: List of grouping entity columns.
            timestamp_col: Timestamp column for chronological order.

        Returns:
            Tuple of (Enriched DataFrame, List of created numerical column names).
    """
    df_sorted = df.sort(timestamp_col)
    created_cols = []
    exprs = []

    if group_cols is None:
        group_cols = ["partner_id", "geo"]

    for col in group_cols:
        feature_name = f"time_since_last_click_{col}"
        created_cols.append(feature_name)

        expr = (
            (
                pl.col(timestamp_col)
                - pl.col(timestamp_col).shift(1).over(col)
            )
            .dt.total_seconds()
            .fill_null(-1.0)
            .alias(feature_name)
        )
        exprs.append(expr)

    return df_sorted.with_columns(exprs), created_cols

def add_domain_ratio_and_cyclic_features(
    df: pl.DataFrame,
    price_col: str = "price",
    geo_col: str = "geo",
    timestamp_col: str = "click_ts",
) -> tuple[pd.DataFrame, list[str]]:
    """Computes domain-specific price ratios and cyclic temporal features.

    Args:
        df: Input Polars DataFrame.
        price_col: Numerical price column name.
        geo_col: Geographic entity column.
        timestamp_col: Timestamp column name.

    Returns:
        Tuple of (Enriched DataFrame, List of created numerical column names).
    """
    created_cols = ["price_ratio_geo", "hour_sin", "hour_cos"]

    geo_expanding_mean = (
        (pl.col(price_col).cum_sum().over(geo_col) - pl.col(price_col))
        / (pl.col(price_col).cum_count().over(geo_col) - 1)
    )
    exprs = [
        # Price relative to geographic group average
        (pl.col(price_col) / geo_expanding_mean).fill_nan(None).fill_null(1.0).alias("price_ratio_geo"),
        # Cyclic hour transformation
        (2 * np.pi * pl.col(timestamp_col).dt.hour() / 24.0)
        .sin()
        .alias("hour_sin"),
        (2 * np.pi * pl.col(timestamp_col).dt.hour() / 24.0)
        .cos()
        .alias("hour_cos"),
    ]

    return df.with_columns(exprs), created_cols

def build_booking_features(
    df: pd.DataFrame,
    schema: dict,
    prior_weight: float = 20.0,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Master feature pipeline: dedupes, sorts, enriches, and updates the schema.

        Accepts and returns pandas; uses Polars internally for expanding
        (leakage-safe) feature computations.

        Args:
            df: Raw booking DataFrame (pandas).
            schema: Column-role contract ('num', 'cat', 'ids', 'target', 'timestamp').
            prior_weight: Smoothing weight for target encoding.

        Returns:
            Tuple of (enriched pandas DataFrame, updated schema).
        """
    logger.info("Starting feature engineering execution pipeline...")

    # Explicit baseline sort to satisfy assumptions downstream
    target_col = schema["target"]
    timestamp_col = schema["timestamp"]
    group_col = schema["ids"][0]

    df = pl.from_pandas(df)
    df_current = df.unique(keep="first")  # expanding features require deduped rows
    df_current = df_current.sort(timestamp_col)
    new_num_cols: list[str] = []

    # 1. Target Encoding
    df_current, target_enc_cols = add_target_encoding(
        df=df_current,
        group_col=group_col,
        target_col=target_col,
        timestamp_col=timestamp_col,
        prior_weight=prior_weight,
    )
    new_num_cols.extend(target_enc_cols)

    # 2. Temporal Deltas
    df_current, delta_cols = add_temporal_delta_features(
        df=df_current,
        group_cols=[group_col, "geo"],
        timestamp_col=timestamp_col,
    )
    new_num_cols.extend(delta_cols)

    # 3. Ratios and Cyclic
    df_current, ratio_cols = add_domain_ratio_and_cyclic_features(
        df=df_current,
        geo_col="geo",
        timestamp_col=timestamp_col,
    )
    new_num_cols.extend(ratio_cols)

    schema_update = {**schema, "num": schema["num"] + new_num_cols}

    all_features = schema_update["num"] + schema_update["cat"]

    assert len(all_features) == len(set(all_features)), "Duplicate feature names in schema"

    logger.info("Feature pipeline done. %d new numerical features.", len(new_num_cols))

    return df_current.to_pandas(), schema_update