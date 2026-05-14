"""
Market screener — six filters that scan the full universe for actionable setups.

Design philosophy:
  These are OBSERVATION tools, not trading signals.  Each function flags
  conditions worth investigating; the user decides whether to act.

  Every function accepts prices_dict so it can be used independently of any
  database or feature pipeline.  Raw OHLCV → screener result in one call.

  Failures are per-symbol: if one symbol's data is malformed, it is skipped
  and a warning is logged.  The rest of the universe is still returned.

Output schema (all six functions return the same columns):
  symbol         : ticker
  current_price  : last close price
  change_pct     : price change over the relevant lookback (%)
  volume_ratio   : today's volume / 20-day avg volume
  rsi            : RSI(14) on last available day
  category_score : 1-10 within this screener (higher = stronger signal)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_OUTPUT_COLS = [
    "symbol", "current_price", "change_pct",
    "volume_ratio", "rsi", "category_score",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder RSI on the last row.  Returns NaN if insufficient data."""
    if len(close) < period + 1:
        return float("nan")
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain.iloc[-1] / (avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else np.nan)
    if np.isnan(rs):
        return 100.0  # all gains, no losses
    return float(100 - (100 / (1 + rs)))


def _volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """Today's volume / rolling 20-day avg volume.  Returns NaN if < period rows."""
    if len(df) < period + 1:
        return float("nan")
    avg = df["volume"].iloc[-(period + 1):-1].mean()   # exclude today from avg
    return float(df["volume"].iloc[-1] / avg) if avg > 0 else float("nan")


def _base_row(sym: str, df: pd.DataFrame, lookback: int = 1) -> dict | None:
    """
    Build the shared output row for a symbol.
    Returns None if the DataFrame is too short to be useful.
    """
    if df is None or len(df) < lookback + 2:
        return None
    close         = df["close"]
    current_price = float(close.iloc[-1])
    prev_price    = float(close.iloc[-(lookback + 1)])
    change_pct    = (current_price - prev_price) / prev_price * 100 if prev_price else float("nan")
    rsi           = _compute_rsi(close)
    vol_ratio     = _volume_ratio(df)

    return {
        "symbol":        sym,
        "current_price": current_price,
        "change_pct":    round(change_pct, 2),
        "volume_ratio":  round(vol_ratio, 2) if not np.isnan(vol_ratio) else float("nan"),
        "rsi":           round(rsi, 1) if not np.isnan(rsi) else float("nan"),
        "category_score": 5,  # default; callers override this
    }


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLS)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Big movers up
# ─────────────────────────────────────────────────────────────────────────────

def big_movers_up(
    prices_dict:   dict[str, pd.DataFrame],
    lookback_days: int   = 1,
    min_pct:       float = 5.0,
) -> pd.DataFrame:
    """
    Symbols up >= min_pct over lookback_days.

    category_score: 1-10 linearly from min_pct to min_pct × 3.
    (A +15% move on a 5% threshold scores 10; a +5% move scores 1.)
    """
    rows = []
    for sym, df in prices_dict.items():
        try:
            row = _base_row(sym, df, lookback=lookback_days)
            if row is None:
                continue
            if row["change_pct"] >= min_pct:
                # Score: 1 at threshold, 10 at 3× threshold
                raw  = (row["change_pct"] - min_pct) / (min_pct * 2)
                row["category_score"] = min(10, max(1, round(1 + raw * 9)))
                rows.append(row)
        except Exception as exc:
            log.warning("[%s] big_movers_up error: %s", sym, exc)

    df_out = pd.DataFrame(rows, columns=_OUTPUT_COLS) if rows else _empty()
    return df_out.sort_values("change_pct", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Big movers down
# ─────────────────────────────────────────────────────────────────────────────

def big_movers_down(
    prices_dict:   dict[str, pd.DataFrame],
    lookback_days: int   = 1,
    min_pct:       float = 5.0,
) -> pd.DataFrame:
    """
    Symbols down >= min_pct (negative change).
    category_score: 1 at threshold, 10 at 3× threshold (more negative = higher score).
    """
    rows = []
    for sym, df in prices_dict.items():
        try:
            row = _base_row(sym, df, lookback=lookback_days)
            if row is None:
                continue
            if row["change_pct"] <= -min_pct:
                raw  = (abs(row["change_pct"]) - min_pct) / (min_pct * 2)
                row["category_score"] = min(10, max(1, round(1 + raw * 9)))
                rows.append(row)
        except Exception as exc:
            log.warning("[%s] big_movers_down error: %s", sym, exc)

    df_out = pd.DataFrame(rows, columns=_OUTPUT_COLS) if rows else _empty()
    return df_out.sort_values("change_pct", ascending=True).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Oversold
# ─────────────────────────────────────────────────────────────────────────────

def oversold(
    prices_dict:       dict[str, pd.DataFrame],
    rsi_threshold:     float = 30.0,
    min_volume_ratio:  float = 0.8,
) -> pd.DataFrame:
    """
    RSI(14) below threshold AND volume not in collapse (volume_ratio >= min_volume_ratio).

    WHY the volume filter?
      RSI can stay oversold for weeks on a dying stock with no buyers.
      Requiring at least 80% of average volume means the move is happening
      on real participation — a rebound needs buyers to show up.

    category_score: 10 at RSI=5, 1 at RSI=threshold−1.
    (Lower RSI = more oversold = higher score.)

    DISCLAIMER: oversold setups historically bounce ~50% of the time.
    This is a setup flag, not a buy signal.
    """
    rows = []
    for sym, df in prices_dict.items():
        try:
            row = _base_row(sym, df)
            if row is None:
                continue
            r = row["rsi"]
            v = row["volume_ratio"]
            if np.isnan(r) or np.isnan(v):
                continue
            if r < rsi_threshold and v >= min_volume_ratio:
                # Score: 10 at RSI 5, 1 at RSI threshold-1
                raw = (rsi_threshold - r) / (rsi_threshold - 5)
                row["category_score"] = min(10, max(1, round(raw * 10)))
                rows.append(row)
        except Exception as exc:
            log.warning("[%s] oversold error: %s", sym, exc)

    df_out = pd.DataFrame(rows, columns=_OUTPUT_COLS) if rows else _empty()
    return df_out.sort_values("rsi", ascending=True).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Overbought
# ─────────────────────────────────────────────────────────────────────────────

def overbought(
    prices_dict:   dict[str, pd.DataFrame],
    rsi_threshold: float = 70.0,
) -> pd.DataFrame:
    """
    RSI(14) above threshold.  Potential exit signal or short setup.
    category_score: 1 at threshold, 10 at RSI 95.
    """
    rows = []
    for sym, df in prices_dict.items():
        try:
            row = _base_row(sym, df)
            if row is None:
                continue
            r = row["rsi"]
            if np.isnan(r):
                continue
            if r > rsi_threshold:
                raw = (r - rsi_threshold) / (95 - rsi_threshold)
                row["category_score"] = min(10, max(1, round(raw * 10)))
                rows.append(row)
        except Exception as exc:
            log.warning("[%s] overbought error: %s", sym, exc)

    df_out = pd.DataFrame(rows, columns=_OUTPUT_COLS) if rows else _empty()
    return df_out.sort_values("rsi", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Volume breakouts
# ─────────────────────────────────────────────────────────────────────────────

def volume_breakouts(
    prices_dict: dict[str, pd.DataFrame],
    multiplier:  float = 3.0,
) -> pd.DataFrame:
    """
    Today's volume >= multiplier × 20-day average.

    Unusual volume often precedes or accompanies significant price moves.
    It does NOT indicate direction — always combine with price context.

    category_score: 1 at multiplier, 10 at 10× avg.
    """
    rows = []
    for sym, df in prices_dict.items():
        try:
            row = _base_row(sym, df)
            if row is None:
                continue
            v = row["volume_ratio"]
            if np.isnan(v):
                continue
            if v >= multiplier:
                raw = (v - multiplier) / (10 - multiplier)
                row["category_score"] = min(10, max(1, round(1 + raw * 9)))
                rows.append(row)
        except Exception as exc:
            log.warning("[%s] volume_breakouts error: %s", sym, exc)

    df_out = pd.DataFrame(rows, columns=_OUTPUT_COLS) if rows else _empty()
    return df_out.sort_values("volume_ratio", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  6. New listings
# ─────────────────────────────────────────────────────────────────────────────

def new_listings(
    prices_dict:      dict[str, pd.DataFrame],
    max_history_days: int = 90,
) -> pd.DataFrame:
    """
    Symbols with fewer than max_history_days of price data.

    New or recently listed instruments lack the history needed for:
      - Reliable RSI / SMA computation (needs 200+ bars)
      - Model scoring (insufficient training data)
      - Backtesting

    These are flagged so the report can warn the user.
    category_score: 1-10 inversely proportional to history length
    (less history = higher score = more caution warranted).
    """
    rows = []
    for sym, df in prices_dict.items():
        try:
            if df is None or df.empty:
                continue
            n_days = len(df)
            if n_days < max_history_days:
                row = _base_row(sym, df, lookback=min(1, n_days - 1))
                if row is None:
                    # Compute even simpler if less than 2 rows
                    rows.append({
                        "symbol":        sym,
                        "current_price": float(df["close"].iloc[-1]) if len(df) else float("nan"),
                        "change_pct":    float("nan"),
                        "volume_ratio":  float("nan"),
                        "rsi":           float("nan"),
                        "category_score": 10,
                    })
                    continue
                raw = 1 - (n_days / max_history_days)
                row["category_score"] = min(10, max(1, round(1 + raw * 9)))
                row["change_pct"] = n_days   # repurpose as "days of history" for display
                rows.append(row)
        except Exception as exc:
            log.warning("[%s] new_listings error: %s", sym, exc)

    df_out = pd.DataFrame(rows, columns=_OUTPUT_COLS) if rows else _empty()
    return df_out.sort_values("change_pct", ascending=True).reset_index(drop=True)
