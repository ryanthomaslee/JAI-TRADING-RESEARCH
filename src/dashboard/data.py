"""
Data orchestration layer for the dashboard.

Extracts the pipeline logic that was embedded in scripts/daily_report.py
into a reusable function so both the terminal script and the Streamlit app
can call it without duplication.

Usage:
    from src.dashboard.data import build_dashboard_data
    data = build_dashboard_data(refresh=False)  # uses cached prices
    data = build_dashboard_data(refresh=True)   # pulls fresh from APIs first
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    RAW_STOCKS_DIR, RAW_CRYPTO_DIR, PARQUET_ENGINE,
    MODELS_PRODUCTION_DIR,
    SCREENER_MOVER_PCT, SCREENER_MOVER_LOOKBACK,
    SCREENER_RSI_OVERSOLD, SCREENER_RSI_OVERBOUGHT,
    SCREENER_MIN_VOLUME_RATIO, SCREENER_VOL_BREAKOUT_MULT,
    SCREENER_NEW_LISTING_DAYS,
)
from src.models.registry import load_all_production_models, save_production_models
from src.dashboard.screener import (
    big_movers_up, big_movers_down, oversold, overbought,
    volume_breakouts, new_listings,
)
from src.dashboard.scoring import score_all

log = logging.getLogger(__name__)


@dataclass
class DashboardData:
    date: str                          # ISO date string "YYYY-MM-DD"
    universe_size: int
    movers_up: pd.DataFrame
    movers_down: pd.DataFrame
    oversold: pd.DataFrame
    overbought: pd.DataFrame
    breakouts: pd.DataFrame
    new_listings: pd.DataFrame
    scores_list: list[dict]            # all symbols, sorted by score desc, with reasoning
    high_conviction: list[dict]        # scores where score >= threshold
    all_scores: pd.DataFrame           # flat DataFrame of scores (no reasoning column)
    prices_dict: dict[str, pd.DataFrame]


def _load_universe() -> tuple[list[str], list[str]]:
    cfg_path = ROOT / "config" / "universe.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    stocks = [s["symbol"]  for s in cfg.get("stocks", [])]
    crypto = [s["ccxt_id"] for s in cfg.get("crypto", [])]
    return stocks, crypto


def _load_prices(
    stock_symbols: list[str],
    crypto_symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Load all cached OHLCV parquet files. Skips missing files silently."""
    prices: dict[str, pd.DataFrame] = {}

    for sym in stock_symbols:
        path = RAW_STOCKS_DIR / f"{sym}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path, engine=PARQUET_ENGINE)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = pd.to_datetime(df.index).normalize()
                prices[sym] = df
            except Exception as exc:
                log.warning("[%s] Failed to load price file: %s", sym, exc)

    for sym in crypto_symbols:
        safe = sym.replace("/", "_")
        path = RAW_CRYPTO_DIR / f"{safe}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path, engine=PARQUET_ENGINE)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = pd.to_datetime(df.index).normalize()
                prices[sym] = df
            except Exception as exc:
                log.warning("[%s] Failed to load price file: %s", sym, exc)

    return prices


def _pull_fresh(full: bool = False) -> None:
    """Invoke scripts/pull_data as a subprocess (recent-only by default)."""
    flag  = [] if full else ["--recent-only"]
    cmd   = [sys.executable, "-m", "scripts.pull_data"] + flag
    log.info("Pulling fresh data: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        log.warning("Data pull exited with code %d — continuing with cached data",
                    result.returncode)


def build_dashboard_data(
    refresh: bool = False,
    full_refresh: bool = False,
    ml_threshold: float = 7.0,
    symbols: list[str] | None = None,
) -> DashboardData:
    """
    Build all screener + scoring results needed for the dashboard.

    Parameters
    ----------
    refresh       : Pull fresh prices (--recent-only) before building
    full_refresh  : Pull full history from 2020 (slow, use weekly)
    ml_threshold  : Minimum composite score for high_conviction list
    symbols       : Limit to specific symbols (None = full universe)

    Returns
    -------
    DashboardData dataclass with all screener results and scores
    """
    if refresh or full_refresh:
        _pull_fresh(full=full_refresh)

    stock_syms, crypto_syms = _load_universe()
    prices = _load_prices(stock_syms, crypto_syms)

    if symbols:
        prices = {sym: df for sym, df in prices.items() if sym in symbols}

    if not prices:
        raise RuntimeError("No price data found. Run pull_data.py first.")

    log.info("Loaded %d symbols", len(prices))

    # Ensure production models exist
    if not any(MODELS_PRODUCTION_DIR.glob("*.pkl")):
        log.info("No production models found — running registry save…")
        save_production_models()

    models = load_all_production_models()
    log.info("Loaded %d production ML models", len(models))

    # Run screeners
    movers_up_df   = big_movers_up(prices,
                                    lookback_days=SCREENER_MOVER_LOOKBACK,
                                    min_pct=SCREENER_MOVER_PCT)
    movers_down_df = big_movers_down(prices,
                                      lookback_days=SCREENER_MOVER_LOOKBACK,
                                      min_pct=SCREENER_MOVER_PCT)
    oversold_df    = oversold(prices,
                               rsi_threshold=SCREENER_RSI_OVERSOLD,
                               min_volume_ratio=SCREENER_MIN_VOLUME_RATIO)
    overbought_df  = overbought(prices, rsi_threshold=SCREENER_RSI_OVERBOUGHT)
    breakouts_df   = volume_breakouts(prices, multiplier=SCREENER_VOL_BREAKOUT_MULT)
    new_df         = new_listings(prices, max_history_days=SCREENER_NEW_LISTING_DAYS)

    log.info("Screeners: %d up, %d down, %d oversold, %d overbought, %d breakouts, %d new",
             len(movers_up_df), len(movers_down_df), len(oversold_df),
             len(overbought_df), len(breakouts_df), len(new_df))

    # Score all symbols
    scores_list = score_all(prices, models)
    high_conviction = [s for s in scores_list if s.get("score", 0) >= ml_threshold]

    all_scores_df = pd.DataFrame([
        {
            "symbol":       s["symbol"],
            "score":        s["score"],
            "rsi":          s.get("rsi"),
            "volume_ratio": s.get("volume_ratio"),
            "has_ml":       s.get("has_ml", False),
            "ml_proba":     s.get("ml_proba"),
        }
        for s in scores_list
    ])

    return DashboardData(
        date            = date.today().isoformat(),
        universe_size   = len(prices),
        movers_up       = movers_up_df,
        movers_down     = movers_down_df,
        oversold        = oversold_df,
        overbought      = overbought_df,
        breakouts       = breakouts_df,
        new_listings    = new_df,
        scores_list     = scores_list,
        high_conviction = high_conviction,
        all_scores      = all_scores_df,
        prices_dict     = prices,
    )
