"""
Volume features.

WHY volume matters for swing signals?
  Price moves on high volume are more likely to be sustained than moves on
  thin volume.  A breakout with 3× average volume is a very different signal
  from the same price move with 0.3× average volume.  Volume features give
  the model a "conviction" dimension it can't get from price alone.

LEAKAGE RULE: all rolling windows are backward-looking only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Rolling z-score of volume relative to the past `period` days.

    z = (volume_t - mean(volume_{t-period:t})) / std(volume_{t-period:t})

    WHY z-score instead of raw volume?
      Raw volume is non-stationary and varies enormously between stocks
      (AAPL trades ~60M shares/day; JPM trades ~8M).  The z-score is
      unitless and directly answers "how unusual is today's volume vs
      recent history?"  Values > 2 indicate abnormally high volume.

    WHY 20-day window?
      ~1 trading month captures the recent baseline without being
      so long that it smooths over regime changes in liquidity.

    Window: strictly backward-looking rolling mean/std.
    """
    mean_vol = df["volume"].rolling(period).mean()
    std_vol  = df["volume"].rolling(period).std()

    zscore = (df["volume"] - mean_vol) / std_vol.replace(0, np.nan)
    return pd.Series(zscore, index=df.index, name=f"volume_zscore_{period}d")


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV) — running cumulative sum of ±volume.

    OBV_t = OBV_{t-1} + volume_t  if close_t > close_{t-1}
          = OBV_{t-1} - volume_t  if close_t < close_{t-1}
          = OBV_{t-1}              if close_t == close_{t-1}

    WHY OBV?
      OBV tracks whether volume is flowing *into* or *out of* a stock.
      A rising OBV with a flat price signals accumulation (bullish divergence).
      A falling OBV with a rising price signals distribution (bearish divergence).
      These divergences are predictive of medium-term reversals.

    WHY normalise by rolling mean?
      Raw OBV is non-stationary (it grows with the cumulative sum).
      Dividing by the 20-day rolling mean of |OBV| gives a stationary ratio.

    This is strictly backward-looking (cumsum is causal by definition).
    """
    direction = np.sign(df["close"].diff())
    obv_raw   = (direction * df["volume"]).cumsum()

    # Normalise: OBV relative to its recent mean absolute value
    obv_abs_mean = obv_raw.abs().rolling(20).mean().replace(0, np.nan)
    obv_norm     = obv_raw / obv_abs_mean

    return pd.Series(obv_norm, index=df.index, name="obv_norm")


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Today's volume divided by the 20-day rolling average volume.

    Ratio > 1.5 → 50% above average (high conviction day)
    Ratio < 0.5 → 50% below average (low conviction / holiday / drift)

    WHY a ratio instead of z-score for this feature?
      The z-score captures "how many standard deviations above average";
      the ratio captures "what multiple of average".  Both are useful but
      have different shapes: z-score is symmetric; ratio is always positive
      and right-skewed.  XGBoost benefits from having both.

    Window: 20-day rolling mean — strictly backward-looking.
    """
    avg_vol = df["volume"].rolling(period).mean()
    ratio   = df["volume"] / avg_vol.replace(0, np.nan)
    return pd.Series(ratio, index=df.index, name=f"volume_ratio_{period}d")


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all volume features aligned to `df`'s index."""
    return pd.concat(
        [
            volume_zscore(df).to_frame(),
            on_balance_volume(df).to_frame(),
            volume_ratio(df).to_frame(),
        ],
        axis=1,
    )
