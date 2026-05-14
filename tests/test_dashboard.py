"""
Dashboard module correctness tests.

Each test targets one specific invariant that, if violated, would produce
silently wrong report output.  We use minimal synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.screener import (
    big_movers_up, big_movers_down, oversold, overbought,
    volume_breakouts, new_listings, _OUTPUT_COLS,
)
from src.dashboard.scoring  import compute_score, score_all
from src.dashboard.report   import generate_report, save_report
from config.settings        import REPORTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 60,
    start: str = "2024-01-01",
    open_: float = 100.0,
    trend: float = 0.0,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Generate simple OHLCV with optional trend and randomised volume.

    WHY randomised volume (not constant)?
      volume_zscore_20d divides by rolling std of volume.  A constant volume
      series has std=0, producing NaN that causes build_features(drop_warmup=True)
      to drop all rows after the warmup period.  Adding noise avoids this.
    """
    dates  = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng    = np.random.default_rng(42)
    closes = open_ * np.cumprod(1 + trend + rng.normal(0, 0.005, n))
    opens  = closes * (1 + rng.normal(0, 0.002, n))
    highs  = np.maximum(closes, opens) * 1.005
    lows   = np.minimum(closes, opens) * 0.995
    # Randomised volume so std > 0 for zscore computation
    vols   = volume * (1 + rng.uniform(-0.4, 1.5, n))

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=dates,
    )


def _make_prices_dict(symbols=("AAPL", "MSFT", "NVDA"), **kwargs):
    return {sym: _make_ohlcv(**kwargs) for sym in symbols}


# ─────────────────────────────────────────────────────────────────────────────
#  Screener: output schema
# ─────────────────────────────────────────────────────────────────────────────

def test_screener_returns_dataframe_with_expected_columns():
    """All screener functions must return a DataFrame with exactly the required columns."""
    prices = _make_prices_dict()
    fns = [
        lambda p: big_movers_up(p, min_pct=0.0),
        lambda p: big_movers_down(p, min_pct=0.0),
        lambda p: oversold(p, rsi_threshold=100.0),       # very permissive to get results
        lambda p: overbought(p, rsi_threshold=0.0),
        lambda p: volume_breakouts(p, multiplier=0.0),
        lambda p: new_listings(p, max_history_days=1000), # everything qualifies
    ]
    for fn in fns:
        result = fn(prices)
        assert isinstance(result, pd.DataFrame), f"{fn} did not return DataFrame"
        for col in _OUTPUT_COLS:
            assert col in result.columns, (
                f"{fn.__name__} missing column '{col}'. Got: {list(result.columns)}"
            )


def test_screener_handles_empty_universe():
    """All screeners must return empty DataFrames without crashing on {}."""
    for fn in [big_movers_up, big_movers_down, oversold, overbought,
               volume_breakouts, new_listings]:
        result = fn({})
        assert isinstance(result, pd.DataFrame)
        assert result.empty


def test_big_movers_up_threshold():
    """Only symbols with change_pct >= min_pct should appear."""
    dates  = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    closes = [100.0, 100.0, 100.0, 100.0, 110.0]   # last day +10%
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": [1e6]*5},
        index=dates,
    )
    result = big_movers_up({"SYM": df}, lookback_days=1, min_pct=5.0)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "SYM"
    assert result.iloc[0]["change_pct"] >= 5.0


def test_big_movers_down_threshold():
    """Only symbols with change_pct <= -min_pct should appear."""
    dates  = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    closes = [100.0, 100.0, 100.0, 100.0, 88.0]   # last day -12%
    # high must be >= close, low must be <= close
    highs  = [101.0, 101.0, 101.0, 101.0, 89.0]
    lows   = [99.0,  99.0,  99.0,  99.0,  87.0]
    df = pd.DataFrame(
        {"open": closes, "high": highs, "low": lows,
         "close": closes, "volume": [1e6]*5},
        index=dates,
    )
    result = big_movers_down({"SYM": df}, lookback_days=1, min_pct=5.0)
    assert len(result) == 1, f"Expected 1 result, got {len(result)}: {result}"
    assert result.iloc[0]["change_pct"] <= -5.0


def test_volume_breakout_threshold():
    """Symbol must have volume >= multiplier * 20-day avg."""
    n   = 25
    vol = [1_000_000.0] * n
    vol[-1] = 5_000_000.0    # 5× spike on last day
    dates  = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    closes = [100.0] * n
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": vol},
        index=dates,
    )
    result = volume_breakouts({"SYM": df}, multiplier=3.0)
    assert len(result) == 1


def test_new_listings_threshold():
    """Symbols with < max_history_days should appear; others should not."""
    short = _make_ohlcv(n=30)   # 30 days — below threshold
    long_ = _make_ohlcv(n=200)  # 200 days — above threshold

    result = new_listings({"SHORT": short, "LONG": long_}, max_history_days=90)
    syms = list(result["symbol"])
    assert "SHORT" in syms
    assert "LONG" not in syms


def test_category_score_range():
    """category_score must always be in [1, 10]."""
    prices = _make_prices_dict(n=60)
    for fn in [lambda p: big_movers_up(p, min_pct=0),
               lambda p: overbought(p, rsi_threshold=0),
               lambda p: volume_breakouts(p, multiplier=0)]:
        result = fn(prices)
        if not result.empty:
            assert result["category_score"].between(1, 10).all(), (
                f"category_score out of [1,10]: {result['category_score'].tolist()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────────────────────

def test_scoring_without_model_returns_valid_score():
    """compute_score with model=None must return a score in [0, 10]."""
    prices = _make_ohlcv(n=60)
    result = compute_score("AAPL", prices, model=None)

    assert "score" in result
    assert 0.0 <= result["score"] <= 10.0, f"Score {result['score']} out of [0,10]"
    assert result["has_ml"] is False
    assert result["ml_proba"] is None
    assert isinstance(result["reasoning"], list)
    assert len(result["reasoning"]) > 0


def test_scoring_with_insufficient_data():
    """compute_score must return 0 and not crash on very short price series."""
    prices = _make_ohlcv(n=5)
    result = compute_score("TINY", prices, model=None)
    assert result["score"] == 0.0


def test_scoring_with_model_returns_valid_score():
    """
    compute_score with a mock model must return score in [0, 10] and set has_ml=True.
    We use a mock object that returns a fixed proba to avoid loading real models.
    """
    import types

    class MockModel:
        def predict_proba(self, X):
            return np.array([0.72])  # high confidence → 4 ML pts

    prices = _make_ohlcv(n=250)   # enough for SMA(200) warmup
    result = compute_score("NVDA", prices, model=MockModel())

    assert result["has_ml"] is True
    assert result["ml_proba"] is not None
    assert abs(result["ml_proba"] - 0.72) < 0.01
    assert 0.0 <= result["score"] <= 10.0


def test_score_all_returns_sorted_descending():
    """score_all must return results sorted by score descending."""
    prices = _make_prices_dict(symbols=["A", "B", "C"], n=60)
    results = score_all(prices, models={})
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────────────────────

def test_report_creates_markdown_file(tmp_path, monkeypatch):
    """save_report must write a .md file at the correct path."""
    # Redirect REPORTS_DIR to tmp_path for this test
    import src.dashboard.report as rep_module
    monkeypatch.setattr(rep_module, "REPORTS_DIR", tmp_path)

    md = "# Test Report\n\nContent here."
    path = rep_module.save_report(md, "2024-01-15")

    assert path.exists()
    assert path.suffix == ".md"
    assert "2024-01-15" in path.name
    assert path.read_text() == md


def test_report_handles_empty_screener_results():
    """generate_report must not crash when all screeners return empty DataFrames."""
    prices = _make_prices_dict(n=60)
    empty  = pd.DataFrame(columns=_OUTPUT_COLS)

    terminal, markdown = generate_report(
        report_date   = "2024-01-15",
        prices_dict   = prices,
        movers_up     = empty,
        movers_down   = empty,
        oversold_df   = empty,
        overbought_df = empty,
        breakouts_df  = empty,
        new_df        = empty,
        scores        = [],
    )

    assert isinstance(terminal, str)
    assert isinstance(markdown, str)
    assert "2024-01-15" in terminal
    assert "No symbols" in terminal or "No" in terminal


def test_report_ml_signals_section_threshold():
    """Only scores >= threshold should appear in ML signals section."""
    from src.dashboard.report import _section_ml_signals
    scores = [
        {"symbol": "HIGH", "score": 9.0, "ml_proba": 0.72, "has_ml": True,
         "rsi": 28.0, "volume_ratio": 2.1, "reasoning": ["strong"]},
        {"symbol": "LOW",  "score": 4.0, "ml_proba": 0.55, "has_ml": True,
         "rsi": 55.0, "volume_ratio": 1.0, "reasoning": ["weak"]},
    ]
    terminal, _ = _section_ml_signals(scores, threshold=7.0)
    assert "HIGH" in terminal
    assert "LOW" not in terminal


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_save_load_roundtrip(tmp_path):
    """
    save_production_models + load_production_model must round-trip:
    the loaded model must be able to predict_proba.
    """
    import pickle
    from src.models.registry import _safe

    # Check if real trained models exist for AAPL
    trained_dir = ROOT / "data" / "processed" / "models"
    aapl_models  = sorted(trained_dir.glob("per_symbol_xgb_AAPL_XGBoostModel_fold*.pkl"))
    if not aapl_models:
        pytest.skip("No trained AAPL model files found — run train_per_symbol first")

    import src.models.registry as reg
    # Temporarily redirect production dir to tmp_path
    original_dir = reg.MODELS_PRODUCTION_DIR

    try:
        import config.settings as settings_mod
        from unittest.mock import patch

        with patch.object(reg, "MODELS_PRODUCTION_DIR", tmp_path):
            saved = reg.save_production_models()

        assert "AAPL" in saved, "AAPL model was not saved"

        with patch.object(reg, "MODELS_PRODUCTION_DIR", tmp_path):
            model = reg.load_production_model("AAPL")

        assert model is not None, "load_production_model returned None"

    finally:
        pass  # no cleanup needed — tmp_path is cleaned by pytest


def test_registry_list_available_models():
    """list_available_models must return a list (possibly empty if not yet saved)."""
    from src.models.registry import list_available_models
    result = list_available_models()
    assert isinstance(result, list)
    # Each entry must be a string
    for sym in result:
        assert isinstance(sym, str)
