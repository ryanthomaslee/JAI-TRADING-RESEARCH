"""
Composite 0-10 scoring for daily dashboard signals.

Scoring components:
  With ML model (max 10):
    ML    0-4  calibrated probability of +5% target hit within 20 days
    RSI   0-2  oversold=+2, neutral=+1, overbought=0
    MACD  0-2  bullish histogram cross=+2, above signal=+1, below=0
    Volume 0-2 >1.5× avg=+2, >1.0×=+1, else=0

  Without ML model (max 10, scaled from 6):
    Technical-only (RSI + MACD + Volume = 0-6), scaled: score × 10/6
    These are clearly labelled "technical-only" in the output.

WHY not just use ML probability alone?
  The ML model is trained on 2020-2024 patterns.  In live use, extreme
  RSI / volume conditions that weren't well-represented in training may
  have higher expected returns.  Combining ML with real-time technicals
  makes the score more robust to regime shifts.

WHY 0-10?
  It's instantly interpretable: ≥7 = worth investigating, ≥9 = high conviction.
  Compare that to "probability 0.67" which requires context to evaluate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.dashboard.screener import _compute_rsi, _volume_ratio

log = logging.getLogger(__name__)

# Feature columns used during per_symbol_xgb training (derived once at runtime)
_OHLCV_COLS  = {"open", "high", "low", "close", "volume"}
_LABEL_COLS  = {"label", "barrier_hit", "days_held", "entry_price",
                "exit_price", "realized_return"}
_NON_FEATURE = _OHLCV_COLS | _LABEL_COLS


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE]


def _ml_component(proba: float) -> int:
    """0-4 based on calibrated probability."""
    if proba >= 0.70:
        return 4
    elif proba >= 0.65:
        return 3
    elif proba >= 0.60:
        return 2
    elif proba >= 0.55:
        return 1
    return 0


def _rsi_component(rsi: float) -> int:
    """0-2 based on RSI level."""
    if np.isnan(rsi):
        return 1   # treat unknown as neutral
    if rsi < 30:
        return 2   # oversold
    elif rsi > 70:
        return 0   # overbought
    return 1       # neutral


def _macd_component(df: pd.DataFrame) -> int:
    """
    0-2 based on MACD histogram direction.

    Bullish cross  (+2): histogram positive AND was negative prior bar
    Above signal   (+1): histogram positive
    Below signal   ( 0): histogram negative or zero
    """
    if len(df) < 27:   # need 26-bar EMA
        return 1
    close    = df["close"]
    ema12    = close.ewm(span=12, adjust=False).mean()
    ema26    = close.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    signal   = macd.ewm(span=9, adjust=False).mean()
    hist     = macd - signal

    if len(hist) < 2:
        return 1
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]

    if curr > 0 and prev <= 0:
        return 2   # bullish cross
    elif curr > 0:
        return 1   # above signal
    return 0


def _volume_component(vol_ratio: float) -> int:
    """0-2 based on today's volume vs 20-day average."""
    if np.isnan(vol_ratio):
        return 1
    if vol_ratio >= 1.5:
        return 2
    elif vol_ratio >= 1.0:
        return 1
    return 0


def compute_score(
    symbol:     str,
    prices_df:  pd.DataFrame,
    model=None,   # Model instance or None
) -> dict:
    """
    Compute composite score for a symbol.

    Parameters
    ----------
    symbol    : ticker string
    prices_df : OHLCV DataFrame (UTC DatetimeIndex)
    model     : loaded production Model instance, or None

    Returns
    -------
    dict with keys:
      score         : 0-10 composite score (float, 1 decimal)
      ml_proba      : calibrated probability (None if no model)
      has_ml        : bool
      rsi           : current RSI(14)
      volume_ratio  : today / 20-day avg
      reasoning     : list of plain-English strings explaining each component
    """
    if prices_df is None or len(prices_df) < 15:
        return {
            "symbol": symbol, "score": 0.0, "ml_proba": None,
            "has_ml": False, "rsi": float("nan"),
            "volume_ratio": float("nan"), "reasoning": ["Insufficient price data"],
        }

    rsi       = _compute_rsi(prices_df["close"])
    vol_ratio = _volume_ratio(prices_df)

    rsi_pts    = _rsi_component(rsi)
    macd_pts   = _macd_component(prices_df)
    vol_pts    = _volume_component(vol_ratio)

    reasoning: list[str] = []

    # ── RSI reasoning ─────────────────────────────────────────────────────
    if not np.isnan(rsi):
        if rsi < 30:
            reasoning.append(f"RSI {rsi:.0f} — oversold, historical bounce zone.")
        elif rsi < 45:
            reasoning.append(f"RSI {rsi:.0f} — below midline, pullback territory.")
        elif rsi <= 55:
            reasoning.append(f"RSI {rsi:.0f} — neutral, no momentum signal.")
        elif rsi <= 70:
            reasoning.append(f"RSI {rsi:.0f} — above midline, momentum present.")
        else:
            reasoning.append(f"RSI {rsi:.0f} — overbought, momentum extended, watch for reversal.")

    # ── MACD reasoning ────────────────────────────────────────────────────
    if macd_pts == 2:
        reasoning.append("MACD bullish cross — histogram turned positive today.")
    elif macd_pts == 1:
        reasoning.append("MACD above signal line — upward momentum.")
    else:
        reasoning.append("MACD below signal line — no upward momentum confirmation.")

    # ── Volume reasoning ──────────────────────────────────────────────────
    if not np.isnan(vol_ratio):
        if vol_ratio >= 1.5:
            reasoning.append(f"Volume {vol_ratio:.1f}× average — strong participation.")
        elif vol_ratio >= 1.0:
            reasoning.append(f"Volume {vol_ratio:.1f}× average — normal participation.")
        else:
            reasoning.append(f"Volume {vol_ratio:.1f}× average — below average.")

    # ── ML component ──────────────────────────────────────────────────────
    ml_proba: float | None = None
    ml_pts    = 0
    has_ml    = False

    if model is not None:
        try:
            from src.features.pipeline import build_features
            feat_df  = build_features(prices_df, drop_warmup=True)
            feat_cols = _feature_cols(feat_df)
            if feat_cols and len(feat_df) > 0:
                X       = feat_df[feat_cols].iloc[-1:].values
                ml_proba = float(model.predict_proba(X)[0])
                ml_pts   = _ml_component(ml_proba)
                has_ml   = True
                reasoning.append(
                    f"ML model: {ml_proba:.0%} probability of +5% target (calibrated)."
                )
            else:
                reasoning.append("ML model: insufficient feature data, skipped.")
        except Exception as exc:
            log.warning("[%s] ML scoring error: %s", symbol, exc)
            reasoning.append(f"ML scoring failed: {exc}")

    # ── Composite score ───────────────────────────────────────────────────
    tech_pts = rsi_pts + macd_pts + vol_pts   # 0-6

    if has_ml:
        raw_score = ml_pts + tech_pts          # 0-10
        score     = float(raw_score)
    else:
        # Scale 0-6 technical score to 0-10
        score = round(tech_pts * 10 / 6, 1)
        reasoning.append("(Technical-only score — no ML model for this symbol.)")

    return {
        "symbol":       symbol,
        "score":        score,
        "ml_proba":     ml_proba,
        "has_ml":       has_ml,
        "rsi":          round(rsi, 1) if not np.isnan(rsi) else float("nan"),
        "volume_ratio": round(vol_ratio, 2) if not np.isnan(vol_ratio) else float("nan"),
        "reasoning":    reasoning,
    }


def score_all(
    prices_dict: dict[str, pd.DataFrame],
    models:      dict[str, object],   # {symbol: Model}
) -> list[dict]:
    """
    Score every symbol in prices_dict.
    Returns list of score dicts sorted by score descending.
    """
    results = []
    for sym, df in prices_dict.items():
        model = models.get(sym)
        try:
            result = compute_score(sym, df, model=model)
            result["symbol"] = sym
            results.append(result)
        except Exception as exc:
            log.warning("[%s] scoring failed: %s", sym, exc)

    return sorted(results, key=lambda r: r["score"], reverse=True)
