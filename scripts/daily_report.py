"""
Daily market intelligence report — main entry point.

Usage:
    uv run python -m scripts.daily_report                 # refresh + report
    uv run python -m scripts.daily_report --no-refresh    # skip data pull
    uv run python -m scripts.daily_report --full-refresh  # re-pull from 2020
    uv run python -m scripts.daily_report --symbols AAPL,MSFT,BTC/USDT
    uv run python -m scripts.daily_report --save-only     # no terminal print

Workflow:
    1. Refresh data (--recent-only by default for speed)
    2. Load all available price data
    3. Run six screeners across full universe
    4. Load production ML models + compute composite scores
    5. Generate report (terminal + markdown)
    6. Print save path
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

from config.settings import (
    RAW_STOCKS_DIR, RAW_CRYPTO_DIR, PARQUET_ENGINE,
    SCREENER_MOVER_PCT, SCREENER_MOVER_LOOKBACK,
    SCREENER_RSI_OVERSOLD, SCREENER_RSI_OVERBOUGHT,
    SCREENER_MIN_VOLUME_RATIO, SCREENER_VOL_BREAKOUT_MULT,
    SCREENER_NEW_LISTING_DAYS,
)
from src.models.registry  import load_all_production_models, save_production_models
from src.dashboard.screener import (
    big_movers_up, big_movers_down, oversold, overbought,
    volume_breakouts, new_listings,
)
from src.dashboard.scoring import score_all
from src.dashboard.report  import generate_report, save_report


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_universe() -> tuple[list[str], list[str]]:
    cfg_path = ROOT / "config" / "universe.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    stocks = [s["symbol"]  for s in cfg.get("stocks", [])]
    crypto = [s["ccxt_id"] for s in cfg.get("crypto", [])]
    return stocks, crypto


def load_prices(
    stock_symbols: list[str],
    crypto_symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Load all cached OHLCV files. Skips missing files silently."""
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


# ─────────────────────────────────────────────────────────────────────────────
#  Data refresh
# ─────────────────────────────────────────────────────────────────────────────

def refresh_data(full: bool = False) -> None:
    """
    Pull fresh market data via the existing pull_data script.
    Uses --recent-only (last 400 days) by default for speed.
    Full refresh re-pulls from 2020.
    """
    flag = [] if full else ["--recent-only"]
    cmd  = [sys.executable, "-m", "scripts.pull_data"] + flag
    log.info("Refreshing data: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        log.warning("Data refresh exited with code %d — continuing with cached data",
                    result.returncode)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily market intelligence report.")
    parser.add_argument("--no-refresh",   action="store_true",
                        help="Skip data pull (use cached prices)")
    parser.add_argument("--full-refresh", action="store_true",
                        help="Re-pull full history from 2020 (slow)")
    parser.add_argument("--symbols",      default=None,
                        help="Comma-separated symbol list (e.g. AAPL,MSFT,BTC/USDT)")
    parser.add_argument("--save-only",    action="store_true",
                        help="Save markdown file but do not print to terminal")
    parser.add_argument("--ml-threshold", type=float, default=7.0,
                        help="Minimum composite score for ML signals section (default: 7)")
    args = parser.parse_args()

    today = date.today().isoformat()
    log.info("Daily report — %s", today)

    # ── Step 1: Refresh data ──────────────────────────────────────────────────
    if not args.no_refresh:
        refresh_data(full=args.full_refresh)
    else:
        log.info("Skipping data refresh (--no-refresh)")

    # ── Step 2: Load prices ───────────────────────────────────────────────────
    stock_syms, crypto_syms = load_universe()
    prices = load_prices(stock_syms, crypto_syms)

    # Filter to requested symbols if --symbols was passed
    if args.symbols:
        requested = [s.strip() for s in args.symbols.split(",")]
        prices    = {sym: df for sym, df in prices.items() if sym in requested}

    log.info("Loaded %d symbols with price data", len(prices))
    if not prices:
        log.error("No price data found. Run with --full-refresh to fetch data.")
        sys.exit(1)

    # ── Step 3: Ensure production models exist ────────────────────────────────
    from config.settings import MODELS_PRODUCTION_DIR
    if not any(MODELS_PRODUCTION_DIR.glob("*.pkl")):
        log.info("No production models found — running registry save…")
        save_production_models()

    # ── Step 4: Load models ───────────────────────────────────────────────────
    models = load_all_production_models()
    log.info("Loaded %d production ML models", len(models))

    # ── Step 5: Run screeners ─────────────────────────────────────────────────
    log.info("Running screeners across %d symbols…", len(prices))
    movers_up_df    = big_movers_up(prices,
                                     lookback_days=SCREENER_MOVER_LOOKBACK,
                                     min_pct=SCREENER_MOVER_PCT)
    movers_down_df  = big_movers_down(prices,
                                       lookback_days=SCREENER_MOVER_LOOKBACK,
                                       min_pct=SCREENER_MOVER_PCT)
    oversold_df     = oversold(prices,
                                rsi_threshold=SCREENER_RSI_OVERSOLD,
                                min_volume_ratio=SCREENER_MIN_VOLUME_RATIO)
    overbought_df   = overbought(prices, rsi_threshold=SCREENER_RSI_OVERBOUGHT)
    breakouts_df    = volume_breakouts(prices, multiplier=SCREENER_VOL_BREAKOUT_MULT)
    new_df          = new_listings(prices, max_history_days=SCREENER_NEW_LISTING_DAYS)

    log.info(
        "Screeners: %d up, %d down, %d oversold, %d overbought, %d breakouts, %d new",
        len(movers_up_df), len(movers_down_df), len(oversold_df),
        len(overbought_df), len(breakouts_df), len(new_df),
    )

    # ── Step 6: ML + composite scoring ───────────────────────────────────────
    log.info("Computing composite scores…")
    scores = score_all(prices, models)

    # ── Step 7: Generate report ───────────────────────────────────────────────
    terminal_str, markdown_str = generate_report(
        report_date   = today,
        prices_dict   = prices,
        movers_up     = movers_up_df,
        movers_down   = movers_down_df,
        oversold_df   = oversold_df,
        overbought_df = overbought_df,
        breakouts_df  = breakouts_df,
        new_df        = new_df,
        scores        = scores,
        ml_threshold  = args.ml_threshold,
    )

    # ── Step 8: Save + print ──────────────────────────────────────────────────
    report_path = save_report(markdown_str, today)

    if not args.save_only:
        print(terminal_str)

    print(f"\n  Report saved → {report_path}")


if __name__ == "__main__":
    main()
