"""
Abstract base class for all data sources.

WHY an abstract base class?
  Stocks come from yfinance, crypto from ccxt, and in the future maybe
  alternative data from other APIs.  Without a contract, every downstream
  module (features, labels, backtest) would need to know which source produced
  a given DataFrame and handle each format differently.

  The ABC enforces:
    1.  A canonical OHLCV column schema — every consumer can assume the same
        column names and dtypes.
    2.  UTC timestamps — mixing timezone-aware and naive datetimes is one of
        the most common sources of silent data bugs in quant systems.
    3.  Parquet caching — so we don't re-download on every run.
    4.  Validation — catches truncated pulls, NaN floods, or bad dtypes early.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import PARQUET_COMPRESSION, PARQUET_ENGINE

log = logging.getLogger(__name__)

# ── Canonical schema ──────────────────────────────────────────────────────────
# All ingestion modules MUST return a DataFrame that conforms to this schema.
# Column order matters: downstream code may rely on position for speed.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

OHLCV_DTYPES: dict[str, type] = {
    "open":   float,
    "high":   float,
    "low":    float,
    "close":  float,
    "volume": float,
}


def validate_ohlcv(df: pd.DataFrame, symbol: str, min_rows: int = 100) -> pd.DataFrame:
    """
    Validate and normalise a raw OHLCV DataFrame.

    Raises ValueError with a descriptive message on any violation so the
    caller knows exactly what went wrong rather than propagating silent NaNs.
    """
    if df is None or df.empty:
        raise ValueError(f"[{symbol}] DataFrame is empty")

    # ── Normalise column names to lowercase ───────────────────────────────────
    df.columns = [c.lower() for c in df.columns]

    # ── Check required columns ────────────────────────────────────────────────
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{symbol}] Missing columns: {missing}")

    # ── Enforce UTC-aware DatetimeIndex ───────────────────────────────────────
    # WHY UTC?  yfinance returns US/Eastern, ccxt returns UTC.  Normalising
    # everything to UTC means "2023-03-12 00:00 UTC" always means the same
    # wall-clock instant regardless of DST or source timezone.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{symbol}] Index must be DatetimeIndex, got {type(df.index)}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"

    # ── Cast dtypes ───────────────────────────────────────────────────────────
    for col, dtype in OHLCV_DTYPES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # ── Drop fully-NaN rows ───────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=OHLCV_COLUMNS)
    dropped = before - len(df)
    if dropped:
        log.warning("[%s] Dropped %d NaN rows", symbol, dropped)

    # ── Minimum row count ─────────────────────────────────────────────────────
    if len(df) < min_rows:
        raise ValueError(
            f"[{symbol}] Only {len(df)} rows after cleaning (min={min_rows}). "
            "Check your date range or symbol name."
        )

    # ── Sanity checks ─────────────────────────────────────────────────────────
    # High must be >= Low on every row; if not, the data is corrupt.
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        log.warning("[%s] %d rows where high < low — likely split/dividend artifact", symbol, bad_hl)

    # ── Sort ascending by time ────────────────────────────────────────────────
    df = df.sort_index()

    # Return only the canonical columns (extras like "dividends" are dropped)
    return df[OHLCV_COLUMNS]


class DataSource(ABC):
    """
    Abstract base class all ingestion modules must subclass.

    Subclasses implement `_fetch_raw()` and get caching + validation for free.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        # Replace "/" so "BTC/USDT" becomes "BTC_USDT.parquet" — safe filename
        safe = symbol.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}.parquet"

    def load(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Public entry point.  Returns a validated, cached DataFrame.

        WHY separate `load()` from `_fetch_raw()`?
          The caching and validation logic is identical for every source.
          Subclasses only need to know *how* to talk to their API; the
          infrastructure lives here in the base class.
        """
        cache = self._cache_path(symbol)

        if cache.exists() and not force_refresh:
            log.info("[%s] Loading from cache: %s", symbol, cache)
            df = pd.read_parquet(cache, engine=PARQUET_ENGINE)
            # Parquet stores timestamps; restore tz if stripped
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df

        log.info("[%s] Fetching from API (start=%s, end=%s)", symbol, start, end)
        raw = self._fetch_raw(symbol, start, end)
        df  = validate_ohlcv(raw, symbol)

        # ── Merge with existing cache ─────────────────────────────────────────
        # WHY merge instead of replace?
        #   --recent-only fetches a short window for speed but must not destroy
        #   the full historical cache built by the initial setup pull.  We merge
        #   new rows into the existing file so history is always preserved and
        #   today's prices are always fresh.
        if cache.exists():
            try:
                old = pd.read_parquet(cache, engine=PARQUET_ENGINE)
                if old.index.tz is None:
                    old.index = old.index.tz_localize("UTC")
                df = (
                    pd.concat([old, df])
                    .sort_index()
                    # Keep the newer fetch when the same timestamp appears in both
                    .groupby(level=0).last()
                    [OHLCV_COLUMNS]
                )
            except Exception as exc:
                log.warning("[%s] Could not merge with existing cache (%s) — replacing", symbol, exc)

        # ── Write to Parquet ──────────────────────────────────────────────────
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(
            table,
            cache,
            compression=PARQUET_COMPRESSION,
        )
        log.info("[%s] Cached %d rows → %s", symbol, len(df), cache)
        return df

    @abstractmethod
    def _fetch_raw(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        """
        Download raw data from the source and return a DataFrame with at least
        the OHLCV columns.  Index must be datetime (tz may be anything —
        validate_ohlcv will normalise it).
        """
        ...
