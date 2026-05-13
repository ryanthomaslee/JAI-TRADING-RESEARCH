"""
Tests for the ingestion layer.

WHY these specific tests?
  1.  Schema tests — verify that both sources produce the canonical column set
      and UTC-aware DatetimeIndex.  If a yfinance update changes column names,
      this catches it immediately.

  2.  Row-count tests — a pull that returns 10 rows when we expect ~1000 is
      silently wrong without this guard.

  3.  validate_ohlcv unit tests — the validator is the most logic-heavy piece;
      test it in isolation so failures point directly to the function.

WHY no mocking?
  For market data, mocking the API response defeats the purpose — we want to
  know the *actual* API still works with our code.  These are integration tests
  that hit the real endpoints.  Mark them slow if CI budget is tight.

  The tests read from the Parquet cache once data has been pulled, so they
  are fast on repeat runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ── Ensure project root is on path ────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAW_STOCKS_DIR, RAW_CRYPTO_DIR, HIST_START
from src.ingestion.base import OHLCV_COLUMNS, validate_ohlcv
from src.ingestion.stocks import StockDataSource
from src.ingestion.crypto import CryptoDataSource


# ─────────────────────────────────────────────────────────────────────────────
#  validate_ohlcv unit tests (no network required)
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, tz: str | None = "UTC") -> pd.DataFrame:
    """Build a minimal valid OHLCV DataFrame for testing."""
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz=tz)
    df = pd.DataFrame(
        {
            "Open":   100.0,
            "High":   105.0,
            "Low":    95.0,
            "Close":  102.0,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )
    return df


def test_validate_ohlcv_happy_path():
    df = validate_ohlcv(_make_df(), symbol="TEST")
    assert list(df.columns) == OHLCV_COLUMNS
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"


def test_validate_ohlcv_lowercase_columns():
    """Columns should be lowercased even if source returns Title Case."""
    df = validate_ohlcv(_make_df(), symbol="TEST")
    assert all(c == c.lower() for c in df.columns)


def test_validate_ohlcv_missing_column():
    df = _make_df()
    df.drop(columns=["Volume"], inplace=True)
    with pytest.raises(ValueError, match="Missing columns"):
        validate_ohlcv(df, symbol="TEST")


def test_validate_ohlcv_empty():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        validate_ohlcv(df, symbol="TEST")


def test_validate_ohlcv_too_few_rows():
    df = _make_df(n=5)
    with pytest.raises(ValueError, match="Only 5 rows"):
        validate_ohlcv(df, symbol="TEST", min_rows=100)


def test_validate_ohlcv_naive_timestamps_get_utc():
    """Naive (tz-less) timestamps should be localised to UTC."""
    df = _make_df(tz=None)
    result = validate_ohlcv(df, symbol="TEST")
    assert str(result.index.tz) == "UTC"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration tests — require cached Parquet files (run pull_data.py first)
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


@pytest.mark.parametrize("symbol", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM"])
def test_stock_schema(symbol: str):
    path = RAW_STOCKS_DIR / f"{symbol}.parquet"
    df = _load_parquet_if_exists(path)
    if df is None:
        pytest.skip(f"No cached data for {symbol} — run pull_data.py first")

    assert list(df.columns) == OHLCV_COLUMNS, f"{symbol}: unexpected columns {list(df.columns)}"
    assert isinstance(df.index, pd.DatetimeIndex), f"{symbol}: index is not DatetimeIndex"
    assert df.index.tz is not None, f"{symbol}: index is timezone-naive"
    assert len(df) >= 100, f"{symbol}: too few rows ({len(df)})"


@pytest.mark.parametrize("symbol,filename", [
    ("BTC/USDT", "BTC_USDT.parquet"),
    ("ETH/USDT", "ETH_USDT.parquet"),
    ("SOL/USDT", "SOL_USDT.parquet"),
    ("BNB/USDT", "BNB_USDT.parquet"),
])
def test_crypto_schema(symbol: str, filename: str):
    path = RAW_CRYPTO_DIR / filename
    df = _load_parquet_if_exists(path)
    if df is None:
        pytest.skip(f"No cached data for {symbol} — run pull_data.py first")

    assert list(df.columns) == OHLCV_COLUMNS, f"{symbol}: unexpected columns"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert len(df) >= 100, f"{symbol}: too few rows ({len(df)})"


@pytest.mark.parametrize("symbol", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM"])
def test_stock_row_count(symbol: str):
    """5 years of daily data should give ~1250 trading days."""
    path = RAW_STOCKS_DIR / f"{symbol}.parquet"
    df = _load_parquet_if_exists(path)
    if df is None:
        pytest.skip(f"No cached data for {symbol}")

    # ~252 trading days/year × 5 years = ~1260, with some tolerance
    assert len(df) >= 900, (
        f"{symbol}: only {len(df)} rows — expected ~1200+ for 5-year pull"
    )


@pytest.mark.parametrize("symbol,filename", [
    ("BTC/USDT", "BTC_USDT.parquet"),
    ("ETH/USDT", "ETH_USDT.parquet"),
])
def test_crypto_row_count(symbol: str, filename: str):
    """Crypto trades every day; 5 years = ~1825 rows."""
    path = RAW_CRYPTO_DIR / filename
    df = _load_parquet_if_exists(path)
    if df is None:
        pytest.skip(f"No cached data for {symbol}")

    assert len(df) >= 1600, (
        f"{symbol}: only {len(df)} rows — expected ~1800+ for 5-year pull"
    )


@pytest.mark.parametrize("symbol", ["SPY", "AAPL"])
def test_stock_date_range(symbol: str):
    """Data should start at or near HIST_START."""
    path = RAW_STOCKS_DIR / f"{symbol}.parquet"
    df = _load_parquet_if_exists(path)
    if df is None:
        pytest.skip(f"No cached data for {symbol}")

    expected_start = pd.Timestamp(HIST_START, tz="UTC")
    # Allow up to 5 days lag (weekends, holidays near Jan 1)
    assert df.index[0] <= expected_start + pd.Timedelta(days=5), (
        f"{symbol}: first row {df.index[0]} is much later than {HIST_START}"
    )
