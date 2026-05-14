"""
Daily market intelligence report — main entry point.

Usage:
    uv run python -m scripts.daily_report                 # refresh + report
    uv run python -m scripts.daily_report --no-refresh    # skip data pull
    uv run python -m scripts.daily_report --full-refresh  # re-pull from 2020
    uv run python -m scripts.daily_report --symbols AAPL,MSFT,BTC/USDT
    uv run python -m scripts.daily_report --save-only     # no terminal print

The data orchestration lives in src/dashboard/data.py and is shared with
the Streamlit web app (app.py).  This script formats and prints the result.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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

from src.dashboard.data   import build_dashboard_data
from src.dashboard.report import generate_report, save_report


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
                        help="Minimum composite score for high-conviction section (default: 7)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    if args.no_refresh:
        log.info("Skipping data refresh (--no-refresh)")

    data = build_dashboard_data(
        refresh      = not args.no_refresh and not args.full_refresh,
        full_refresh = args.full_refresh,
        ml_threshold = args.ml_threshold,
        symbols      = symbols,
    )

    log.info("Daily report — %s", data.date)
    log.info("Loaded %d symbols with price data", data.universe_size)
    log.info("Screeners: %d up, %d down, %d oversold, %d overbought, %d breakouts, %d new",
             len(data.movers_up), len(data.movers_down), len(data.oversold),
             len(data.overbought), len(data.breakouts), len(data.new_listings))

    terminal_str, markdown_str = generate_report(
        report_date   = data.date,
        prices_dict   = data.prices_dict,
        movers_up     = data.movers_up,
        movers_down   = data.movers_down,
        oversold_df   = data.oversold,
        overbought_df = data.overbought,
        breakouts_df  = data.breakouts,
        new_df        = data.new_listings,
        scores        = data.scores_list,
        ml_threshold  = args.ml_threshold,
    )

    report_path = save_report(markdown_str, data.date)

    if not args.save_only:
        print(terminal_str)

    print(f"\n  Report saved → {report_path}")


if __name__ == "__main__":
    main()
