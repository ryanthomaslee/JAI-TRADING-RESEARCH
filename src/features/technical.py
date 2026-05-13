"""
Technical indicators — momentum, trend, and mean-reversion signals.

LEAKAGE RULE (critical):
  Every feature at row t must be computable from rows [0 … t] only.
  - Use .rolling(window).mean() with default min_periods.
  - NEVER use centered=True (that peeks at future rows symmetrically).
  - NEVER use .shift(-n) anywhere in this module.
  - NEVER use .expanding() unless it only looks backward (it does, but
    document it clearly so reviewers can verify).

WHY these specific indicators?
  We want signals from three orthogonal "views":
    1. Momentum — is price trending up or down?  (RSI, MACD, log returns)
    2. Mean-reversion — is price stretched relative to history?  (BB, SMA ratios)
    3. Trend confirmation — are we above or below key moving averages?  (SMA/EMA crosses)

  XGBoost can combine these nonlinearly; we just need to supply the raw signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Log returns
# ─────────────────────────────────────────────────────────────────────────────

def log_returns(close: pd.Series) -> pd.DataFrame:
    """
    Log return over 1, 5, 10, and 20 trading days.

    WHY log returns instead of simple returns?
      log(P_t / P_{t-n}) = sum of daily log returns over n days.
      This additive property makes multi-period returns comparable and is
      numerically better-behaved for large price moves.

    Window: each is a simple backward difference — no future data.
    """
    lr = np.log(close / close.shift(1))  # daily log return (1-bar lookback)
    return pd.DataFrame(
        {
            "ret_1d":  lr,
            "ret_5d":  np.log(close / close.shift(5)),
            "ret_10d": np.log(close / close.shift(10)),
            "ret_20d": np.log(close / close.shift(20)),
        },
        index=close.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Simple and Exponential Moving Averages
# ─────────────────────────────────────────────────────────────────────────────

def moving_averages(close: pd.Series) -> pd.DataFrame:
    """
    SMA(20), SMA(50), SMA(200) and EMA(12), EMA(26).

    WHY these windows?
      20 / 50 / 200-day SMAs are the most-watched MAs by institutional traders,
      making them self-fulfilling support/resistance levels — their signals
      have real market impact, not just backtesting noise.

    WHY price-to-MA ratios instead of raw MA values?
      Raw MA values are non-stationary (they drift with price level over 5 years).
      The ratio close/SMA(20) is stationary: it oscillates around 1.0, which
      XGBoost can learn from.

    All windows strictly backward-looking (no centered=True).
    """
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()

    return pd.DataFrame(
        {
            # Ratios: 1.05 means price is 5% above the 20-day MA
            "price_to_sma20":  close / sma20,
            "price_to_sma50":  close / sma50,
            "price_to_sma200": close / sma200,
            # EMA cross: positive = short MA above long MA (uptrend signal)
            "ema_cross":       (ema12 - ema26) / close,
            # Raw MAs kept for Bollinger Bands computation (not used as features)
            "_sma20":          sma20,  # prefixed _ → pipeline strips these
        },
        index=close.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RSI
# ─────────────────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's Relative Strength Index over `period` days.

    WHY RSI(14)?
      14 days is the standard Wilder period and captures ~2-3 weeks of
      momentum — a natural fit for a 1-4 week holding horizon.

    Implementation: uses EWM with alpha=1/period (Wilder smoothing).
    This is backward-looking only.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing = EWM with alpha=1/period, no future data
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return pd.Series(100 - (100 / (1 + rs)), index=close.index, name=f"rsi_{period}")


# ─────────────────────────────────────────────────────────────────────────────
#  MACD
# ─────────────────────────────────────────────────────────────────────────────

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.

    WHY normalise by close price?
      Raw MACD is in price units ($).  NVDA at $800 has a larger raw MACD than
      JPM at $200 for the same *relative* move.  Dividing by close makes the
      feature comparable across assets and over time.

    All EWMs are causal (no future data).
    """
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line

    return pd.DataFrame(
        {
            "macd_line":  macd_line / close,   # normalised
            "macd_signal": sig_line / close,
            "macd_hist":  histogram / close,
        },
        index=close.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

def bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Band position (%B) and bandwidth.

    WHY %B instead of raw upper/lower band levels?
      %B = (price - lower) / (upper - lower) gives a value in [0, 1] that
      is directly comparable across stocks and time periods.
      %B > 1.0 means price is above the upper band (overbought signal).
      %B < 0.0 means price is below the lower band (oversold signal).

    WHY bandwidth?
      Bandwidth = (upper - lower) / middle measures volatility expansion.
      A "Bollinger Squeeze" (low bandwidth) often precedes a large move.

    Window: 20-day rolling — strictly backward-looking.
    """
    sma    = close.rolling(period).mean()
    std    = close.rolling(period).std()
    upper  = sma + n_std * std
    lower  = sma - n_std * std

    pct_b     = (close - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / sma.replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_pct_b":    pct_b,
            "bb_bandwidth": bandwidth,
        },
        index=close.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public builder
# ─────────────────────────────────────────────────────────────────────────────

def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all technical features into one DataFrame aligned to `df`'s index.
    Internal helper columns (prefixed _) are dropped before returning.
    """
    close = df["close"]

    parts = [
        log_returns(close),
        moving_averages(close),
        rsi(close).to_frame(),
        macd(close),
        bollinger_bands(close),
    ]

    result = pd.concat(parts, axis=1)

    # Drop internal helper columns (prefixed with _)
    internal = [c for c in result.columns if c.startswith("_")]
    result = result.drop(columns=internal)

    return result
