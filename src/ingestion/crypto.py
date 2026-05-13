"""
ccxt / Binance crypto data source.

Design notes
────────────
WHY ccxt instead of a direct Binance API call?
  ccxt is a unified library that supports 100+ exchanges with the same
  interface.  If you want to add Coinbase or Kraken later, you change one
  string — not the entire ingestion layer.

WHY pagination?
  Binance's OHLCV endpoint returns max 1000 candles per call.  5 years of
  daily bars = ~1825 rows, so we need at least 2 calls.  The pagination loop
  handles this automatically regardless of the date range.

WHY rate limiting?
  Binance enforces a 1200 request/minute weight limit.  Hitting it returns
  HTTP 429 and a temporary IP ban.  A small sleep between calls keeps us
  well below the limit.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import ccxt
import pandas as pd

from config.settings import RATE_LIMIT_MS, RAW_CRYPTO_DIR, MAX_RETRIES, RETRY_BACKOFF_S
from src.ingestion.base import DataSource

log = logging.getLogger(__name__)

# Binance returns candles in this order
_CCXT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Max candles per API call (Binance limit)
_PAGE_SIZE = 1000


class CryptoDataSource(DataSource):
    """Downloads daily OHLCV bars for crypto pairs via ccxt / Binance."""

    def __init__(self, cache_dir: Path = RAW_CRYPTO_DIR) -> None:
        super().__init__(cache_dir)
        # WHY sandbox=False + no API key?
        #   Public OHLCV endpoints on Binance don't require authentication.
        #   We only need keys for trading, which this system never does.
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,   # ccxt's built-in polite throttle
                "options": {"defaultType": "spot"},
            }
        )

    def _fetch_raw(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        """
        Paginated fetch: loop until we have all bars from `start` to `end`.
        """
        # Convert start/end strings to millisecond timestamps (ccxt standard)
        since_ms = self.exchange.parse8601(f"{start}T00:00:00Z")
        end_ms   = (
            self.exchange.parse8601(f"{end}T00:00:00Z")
            if end
            else int(time.time() * 1000)
        )

        all_candles: list[list] = []
        fetch_since = since_ms

        while True:
            log.debug(
                "[%s] Fetching page from %s",
                symbol,
                pd.Timestamp(fetch_since, unit="ms", tz="UTC"),
            )

            candles = self._fetch_page(symbol, fetch_since)

            if not candles:
                break

            all_candles.extend(candles)

            last_ts = candles[-1][0]  # timestamp of last candle in this page

            # Stop if we've passed end_ms or got a partial page (end of data)
            if last_ts >= end_ms or len(candles) < _PAGE_SIZE:
                break

            # Advance to one millisecond after the last candle
            fetch_since = last_ts + 1
            time.sleep(RATE_LIMIT_MS / 1000)  # polite pause

        if not all_candles:
            raise ValueError(f"ccxt returned no candles for {symbol}")

        df = pd.DataFrame(all_candles, columns=_CCXT_COLUMNS)

        # Convert ms timestamps to UTC DatetimeIndex
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        # Filter to requested date range (pagination may slightly overshoot)
        df = df[df.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]

        return df

    def _fetch_page(self, symbol: str, since_ms: int) -> list[list]:
        """Single paginated request with retry logic."""
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe="1d",
                    since=since_ms,
                    limit=_PAGE_SIZE,
                )
            except Exception as exc:
                last_exc = exc
                wait = RETRY_BACKOFF_S * (2 ** (attempt - 1))
                log.warning(
                    "[%s] Page fetch attempt %d/%d failed: %s. Retrying in %ds…",
                    symbol, attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[{symbol}] Page fetch failed after {MAX_RETRIES} attempts: {last_exc}"
        )


def pull_crypto(
    symbols: list[str],
    start: str,
    end: str | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Convenience function: pull multiple symbols, return {symbol: DataFrame}.
    Uses a single exchange instance (connection reuse) for efficiency.
    """
    source = CryptoDataSource()
    results: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            results[sym] = source.load(sym, start, end, force_refresh=force_refresh)
            log.info("[%s] OK — %d rows", sym, len(results[sym]))
        except Exception as exc:
            log.error("[%s] FAILED: %s", sym, exc)

    return results
