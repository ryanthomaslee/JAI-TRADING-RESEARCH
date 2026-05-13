"""
Feature pipeline: combines all feature modules into a single DataFrame.

WHY a pipeline module?
  Every downstream consumer (model training, backtest, live scoring) needs
  the exact same feature set in the exact same order.  Without this central
  assembler, it's easy for a training script and a scoring script to compute
  slightly different features and produce silent prediction errors.

  The pipeline also owns the "warm-up period" logic: SMA(200) needs 200 rows
  before it produces a valid value.  We drop the burn-in rows here so that
  downstream code never sees NaN features — it just gets a clean DataFrame.

LEAKAGE RULE: build_features() receives a raw OHLCV DataFrame and returns
  features.  It must NEVER receive labels or anything computed from future
  prices.  Labels are added by a separate module (src/labels/).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.features.technical  import build_technical_features
from src.features.volatility import build_volatility_features
from src.features.volume     import build_volume_features

log = logging.getLogger(__name__)

# The longest rolling window in any feature module.
# SMA(200) needs 200 rows; we need 1 extra for diff/shift(1).
# All rows before this index will have NaNs and are dropped.
WARMUP_PERIOD = 201


def build_features(df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """
    Compute and concatenate all features for a single symbol's OHLCV DataFrame.

    Parameters
    ----------
    df          : validated OHLCV DataFrame from the ingestion layer
    drop_warmup : if True (default), drop the first WARMUP_PERIOD rows
                  that contain NaNs from long rolling windows

    Returns
    -------
    DataFrame with all features.  The original OHLCV columns are retained
    alongside the features so that the labels module can use them.

    WHY keep OHLCV in the output?
      The triple-barrier label module needs access to `high`, `low`, and
      `close` for each forward day.  Keeping everything in one DataFrame
      avoids a separate join and the index-alignment bugs that come with it.
    """
    # Build each feature group
    technical  = build_technical_features(df)
    volatility = build_volatility_features(df)
    volume     = build_volume_features(df)

    # Combine: OHLCV first, then feature groups
    # WHY concat on axis=1?  All DataFrames share the same DatetimeIndex,
    # so pandas aligns them by index automatically — no manual join needed.
    combined = pd.concat([df, technical, volatility, volume], axis=1)

    n_before = len(combined)

    if drop_warmup:
        # Drop rows where ANY feature is NaN (warmup period)
        # WHY dropna rather than just slicing off the first 201 rows?
        #   dropna is robust: if a feature calculation introduces unexpected
        #   NaNs elsewhere, we catch them here rather than propagating silently.
        combined = combined.dropna()

    n_after = len(combined)
    log.debug(
        "build_features: %d rows in, %d rows out (%d dropped in warmup)",
        n_before, n_after, n_before - n_after,
    )

    return combined


def feature_names(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (excludes OHLCV columns).
    Useful for model training to select only the feature columns.
    """
    ohlcv = {"open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in ohlcv]
