"""
CLI script to pull all data for the configured universe.

Usage:
    uv run python -m scripts.pull_data               # use defaults from settings
    uv run python -m scripts.pull_data --refresh     # force re-download everything
    uv run python -m scripts.pull_data --stocks-only
    uv run python -m scripts.pull_data --crypto-only

WHY a CLI script instead of a notebook?
  Scripts are reproducible, scriptable, and can be scheduled (cron, GitHub
  Actions).  Notebooks are great for exploration but bad for automation.
  The --refresh flag lets you re-pull without editing any code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# ── Make sure the project root is on sys.path ─────────────────────────────────
# WHY?  When you run `python -m scripts.pull_data`, Python adds the project
# root to sys.path automatically because of the -m flag.  This explicit insert
# is a safety net for when someone calls the file directly.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import HIST_START, HIST_END
from src.ingestion.stocks import pull_stocks
from src.ingestion.crypto import pull_crypto

# Recent-only pulls this many days back (enough for SMA(200) warmup + buffer)
_RECENT_DAYS = 400

# ── Logging setup ─────────────────────────────────────────────────────────────
# Log to stdout so it's readable in a terminal and capturable by CI logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def load_universe() -> tuple[list[str], list[str]]:
    """Parse config/universe.yaml and return (stock_symbols, crypto_symbols)."""
    cfg_path = ROOT / "config" / "universe.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    stocks = [s["symbol"] for s in cfg.get("stocks", [])]
    crypto = [s["ccxt_id"] for s in cfg.get("crypto", [])]
    return stocks, crypto


def print_summary(label: str, data: dict) -> None:
    """Pretty-print row counts and date ranges after a pull."""
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    if not data:
        print("  (no data)")
        return
    for sym, df in data.items():
        if df is not None and not df.empty:
            print(
                f"  {sym:<14}  {len(df):>5} rows  "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
        else:
            print(f"  {sym:<14}  FAILED")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull market data for the configured universe.")
    parser.add_argument("--refresh",      action="store_true", help="Force re-download (ignore cache)")
    parser.add_argument("--stocks-only",  action="store_true", help="Pull stocks only")
    parser.add_argument("--crypto-only",  action="store_true", help="Pull crypto only")
    parser.add_argument("--start",        default=HIST_START,  help=f"Start date (default: {HIST_START})")
    parser.add_argument("--end",          default=HIST_END,    help="End date (default: today)")
    parser.add_argument(
        "--recent-only",
        action="store_true",
        help=(
            f"Only fetch last {_RECENT_DAYS} days (faster daily refresh). "
            "Sufficient for all screener indicators and ML scoring."
        ),
    )
    args = parser.parse_args()

    if args.recent_only:
        recent_start = (datetime.today() - timedelta(days=_RECENT_DAYS)).strftime("%Y-%m-%d")
        args.start   = recent_start
        args.refresh = True   # must re-fetch to update cache
        log.info("--recent-only: fetching from %s (last %d days)", recent_start, _RECENT_DAYS)

    stock_symbols, crypto_symbols = load_universe()

    log.info("Universe: %d stocks, %d crypto pairs", len(stock_symbols), len(crypto_symbols))
    log.info("Date range: %s → %s", args.start, args.end or "today")
    if args.refresh:
        log.info("Force-refresh enabled — ignoring cache")

    stock_data: dict = {}
    crypto_data: dict = {}

    # ── Stocks ─────────────────────────────────────────────────────────────────
    if not args.crypto_only:
        log.info("Pulling stocks…")
        stock_data = pull_stocks(
            stock_symbols,
            start=args.start,
            end=args.end,
            force_refresh=args.refresh,
        )

    # ── Crypto ─────────────────────────────────────────────────────────────────
    if not args.stocks_only:
        log.info("Pulling crypto…")
        crypto_data = pull_crypto(
            crypto_symbols,
            start=args.start,
            end=args.end,
            force_refresh=args.refresh,
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary("STOCKS", stock_data)
    print_summary("CRYPTO", crypto_data)

    # Exit 1 if everything failed, so CI can catch it
    if not stock_data and not crypto_data:
        sys.exit(1)


if __name__ == "__main__":
    main()
