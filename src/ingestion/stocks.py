"""
yfinance stock data source.

Design notes
────────────
WHY auto_adjust=True?
  yfinance returns *adjusted* prices (split- and dividend-corrected) when
  auto_adjust=True.  This is essential for backtesting — without it, every
  stock split looks like a massive overnight price drop, and your model will
  learn a spurious "crash" pattern.

WHY retries with exponential backoff?
  yfinance hits Yahoo's undocumented API which has rate limits and occasional
  transient failures.  A simple retry loop with doubling waits (2s, 4s, 8s)
  handles 99% of transient errors without hammering the server.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import MAX_RETRIES, RETRY_BACKOFF_S, RAW_STOCKS_DIR
from src.ingestion.base import DataSource

log = logging.getLogger(__name__)


class StockDataSource(DataSource):
    """Downloads daily OHLCV bars for equities via yfinance."""

    def __init__(self, cache_dir: Path = RAW_STOCKS_DIR) -> None:
        super().__init__(cache_dir)

    def _fetch_raw(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        """
        Pull daily bars with exponential-backoff retries.
        """
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,   # split/dividend adjusted — see module docstring
                    actions=False,      # we don't need dividends/splits columns
                )

                if df.empty:
                    raise ValueError(f"yfinance returned empty DataFrame for {symbol}")

                # yfinance returns columns like "Open", "High" etc — base class
                # validate_ohlcv will lowercase them
                return df

            except Exception as exc:
                last_exc = exc
                wait = RETRY_BACKOFF_S * (2 ** (attempt - 1))
                log.warning(
                    "[%s] Attempt %d/%d failed: %s. Retrying in %ds…",
                    symbol, attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[{symbol}] All {MAX_RETRIES} attempts failed. Last error: {last_exc}"
        )


def pull_stocks(
    symbols: list[str],
    start: str,
    end: str | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Convenience function: pull multiple symbols, return {symbol: DataFrame}.
    Failed symbols are logged and skipped (we don't abort the whole run for
    one bad symbol).
    """
    source = StockDataSource()
    results: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            results[sym] = source.load(sym, start, end, force_refresh=force_refresh)
            log.info("[%s] OK — %d rows", sym, len(results[sym]))
        except Exception as exc:
            log.error("[%s] FAILED: %s", sym, exc)

    return results
