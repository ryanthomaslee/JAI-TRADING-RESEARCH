"""
Central configuration for market-intel.

WHY a single settings file?
  Every part of the pipeline (ingestion, features, backtest) needs to agree on
  paths and date ranges. Hardcoding them in multiple files leads to subtle bugs
  where one module uses a different start date than another.  A single source
  of truth fixes that.
"""

from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
# __file__ is config/settings.py → .parent is config/ → .parent is project root
ROOT = Path(__file__).parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_STOCKS_DIR  = DATA_DIR / "raw" / "stocks"
RAW_CRYPTO_DIR  = DATA_DIR / "raw" / "crypto"
PROCESSED_DIR   = DATA_DIR / "processed"
FEATURES_DIR    = DATA_DIR / "features"

# Ensure directories exist at import time (idempotent)
for _d in [RAW_STOCKS_DIR, RAW_CRYPTO_DIR, PROCESSED_DIR, FEATURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Historical window ─────────────────────────────────────────────────────────
# WHY 5 years?  Walk-forward CV for swing trading needs at least 3–4 years of
# out-of-sample folds to be statistically meaningful.  5 years gives us ~60
# monthly folds or ~10 6-month folds with room to spare.
HIST_START = "2020-01-01"
HIST_END   = None          # None → "today" at pull time

# ── Interval ─────────────────────────────────────────────────────────────────
# Daily bars for a 1-4 week signal horizon.  Intra-day adds noise and storage
# cost without improving medium-swing accuracy.
INTERVAL = "1d"

# ── Parquet settings ─────────────────────────────────────────────────────────
# Parquet is columnar + compressed → 5-10× smaller than CSV, and pandas reads
# it ~10× faster.  We use snappy compression (fast) not gzip (smaller).
PARQUET_ENGINE     = "pyarrow"
PARQUET_COMPRESSION = "snappy"

# ── Retry / rate-limit defaults ───────────────────────────────────────────────
MAX_RETRIES      = 3
RETRY_BACKOFF_S  = 2   # seconds between retries (doubles each attempt)
RATE_LIMIT_MS    = 200  # ms between crypto API calls

# ── Dashboard screener thresholds ────────────────────────────────────────────
# All defaults are tunable here without touching screener code.
SCREENER_MOVER_PCT         = 5.0   # % move to qualify as a "big mover"
SCREENER_MOVER_LOOKBACK    = 1     # days for big-mover lookback
SCREENER_RSI_OVERSOLD      = 30    # RSI below this = oversold
SCREENER_RSI_OVERBOUGHT    = 70    # RSI above this = overbought
SCREENER_MIN_VOLUME_RATIO  = 0.8   # min volume/avg for oversold filter
SCREENER_VOL_BREAKOUT_MULT = 3.0   # volume multiplier for breakout flag
SCREENER_NEW_LISTING_DAYS  = 90    # history days below this = new listing

# Maximum rows shown in terminal per section (remainder in .md file only)
REPORT_TERMINAL_MAX_ROWS   = 15

# Production model directory (registry output)
MODELS_PRODUCTION_DIR = DATA_DIR / "models" / "production"
MODELS_PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

# Reports directory
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
