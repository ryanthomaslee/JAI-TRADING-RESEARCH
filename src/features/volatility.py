"""
Volatility features.

WHY volatility features for a swing-trading signal?
  Two key reasons:
    1. Regime detection — high-vol regimes have different return dynamics.
       A breakout signal in a low-vol environment is very different from the
       same signal during a VIX spike.
    2. Position sizing input — realized vol feeds into Kelly / fixed-fractional
       sizing in Phase 3; by computing it here we keep sizing logic clean.

LEAKAGE RULE: all windows are strictly backward-looking.
  Rolling std uses the default min_periods and no centered=True.
  ATR uses only the *current and prior* rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling realized volatility as annualised standard deviation of log returns.

    Windows: 10-day and 20-day.

    WHY annualise?
      Multiplying by sqrt(252) converts daily vol to an annualised figure that
      is directly comparable to options-implied vol (e.g. VIX is annualised).
      This makes the feature meaningful beyond backtesting — you can use it in
      position sizing formulas.

    WHY log returns for vol estimation?
      Log returns are approximately normal and additive; simple returns are not.
      Using log returns makes the vol estimate more stable.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))

    vol10  = log_ret.rolling(10).std()  * np.sqrt(252)
    vol20  = log_ret.rolling(20).std()  * np.sqrt(252)

    return pd.DataFrame(
        {
            "realized_vol_10d": vol10,
            "realized_vol_20d": vol20,
        },
        index=df.index,
    )


def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR(14) normalised by close price.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = 14-day Wilder EWM of True Range.

    WHY normalise ATR by close?
      Raw ATR is in $ units.  A $5 ATR on a $10 stock is enormous (50%);
      the same $5 ATR on a $500 stock is trivial (1%).  Dividing by close
      gives a unitless "daily range as % of price" that is comparable across
      all symbols and across time as prices change.

    WHY ATR over simple high-low range?
      ATR accounts for gap opens — if a stock gaps down 5%, the simple
      high-low range misses that overnight volatility entirely.

    Window: 14-day Wilder EWM — strictly backward-looking.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    return pd.Series(atr / close, index=df.index, name=f"atr_{period}_pct")


def high_low_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-bar and rolling high-low range as a fraction of close.

    WHY?
      Intra-day range captures micro-volatility that ATR smooths over.
      A narrow-range day in a high-ATR environment can signal consolidation
      before a continuation move — useful context for the model.

    Windows: 1-bar (today's range) and 5-day rolling mean.
    """
    hl_pct = (df["high"] - df["low"]) / df["close"]

    return pd.DataFrame(
        {
            "hl_range_pct":       hl_pct,
            "hl_range_5d_mean":   hl_pct.rolling(5).mean(),
        },
        index=df.index,
    )


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all volatility features aligned to `df`'s index."""
    return pd.concat(
        [
            realized_volatility(df),
            average_true_range(df).to_frame(),
            high_low_range(df),
        ],
        axis=1,
    )
