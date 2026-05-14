"""
Production model registry.

WHY a registry?
  Phase 3 trains N folds × M symbols = many model files.  The daily scoring
  pipeline needs exactly one model per symbol: the last (most data-rich) fold.
  The registry abstracts that selection and provides a clean load API so that
  scoring code never has to know about fold numbering or file naming.

  It also provides a stable location (data/models/production/) that the rest
  of the pipeline can depend on, separate from the training artefacts in
  data/processed/models/.

Model selection policy: LAST fold (highest fold number).
  Walk-forward folds use an expanding training window; the last fold trains
  on the most historical data and is therefore the best-calibrated model for
  live prediction.  Earlier folds are kept for cross-validation analysis only.

Supported strategy: per_symbol_xgb
  We use XGBoost per-symbol models because Phase 3 showed they outperform the
  global model on individual stocks (JPM PR-AUC 0.52 vs 0.42 global), and
  Phase 4.5 confirmed the per-symbol WINNERS universe is the tradeable subset.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from config.settings import MODELS_PRODUCTION_DIR

log = logging.getLogger(__name__)

# Source models live here (Phase 3 training output)
_TRAINED_MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "models"

# Naming convention: {strategy}_{safe_symbol}_{ModelClass}_fold{n}.pkl
_STRATEGY        = "per_symbol_xgb"
_MODEL_CLASS     = "XGBoostModel"

# The 10 symbols that have trained models (original Phase 3 universe)
_CORE_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM",
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
]


def _safe(symbol: str) -> str:
    """Convert symbol to filesystem-safe name (matches trainer.py convention)."""
    return symbol.replace("/", "_").replace(":", "_")


def _find_last_fold(symbol: str) -> Path | None:
    """
    Find the highest-numbered fold model file for a symbol.
    Returns the Path or None if no model exists.
    """
    safe = _safe(symbol)
    pattern = f"{_STRATEGY}_{safe}_{_MODEL_CLASS}_fold*.pkl"
    matches  = sorted(_TRAINED_MODELS_DIR.glob(pattern))
    if not matches:
        log.warning("No model files found for %s (pattern: %s)", symbol, pattern)
        return None
    # Last in sorted order = highest fold number
    return matches[-1]


def save_production_models(strategy: str = _STRATEGY) -> dict[str, Path]:
    """
    Copy the last-fold model for each core symbol to the production directory.

    Returns a dict {symbol: destination_path} for the models that were saved.
    Skips symbols where no model file exists (warns but doesn't crash).
    """
    saved: dict[str, Path] = {}
    MODELS_PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    for sym in _CORE_SYMBOLS:
        src = _find_last_fold(sym)
        if src is None:
            continue
        dest = MODELS_PRODUCTION_DIR / f"{_safe(sym)}.pkl"
        shutil.copy2(src, dest)
        log.info("  Saved production model: %s → %s  (source: %s)",
                 sym, dest.name, src.name)
        saved[sym] = dest

    log.info("Registry: %d/%d production models saved to %s",
             len(saved), len(_CORE_SYMBOLS), MODELS_PRODUCTION_DIR)
    return saved


def load_production_model(symbol: str):
    """
    Load the production model for a symbol.

    Returns the Model instance or None if no production model exists.
    """
    # Lazy import to avoid circular deps
    from src.models.base import Model

    dest = MODELS_PRODUCTION_DIR / f"{_safe(symbol)}.pkl"
    if not dest.exists():
        log.debug("No production model for %s at %s", symbol, dest)
        return None
    try:
        from src.models.xgboost_model import XGBoostModel
        model = XGBoostModel.load(dest)
        log.debug("Loaded production model for %s", symbol)
        return model
    except Exception as exc:
        log.warning("Failed to load production model for %s: %s", symbol, exc)
        return None


def load_all_production_models() -> dict[str, object]:
    """
    Load all available production models.
    Returns {symbol: Model} for models that loaded successfully.
    """
    models: dict[str, object] = {}
    for sym in _CORE_SYMBOLS:
        m = load_production_model(sym)
        if m is not None:
            models[sym] = m
    log.info("Registry: loaded %d production models", len(models))
    return models


def list_available_models() -> list[str]:
    """Return list of symbols for which a production model exists."""
    available = []
    for sym in _CORE_SYMBOLS:
        path = MODELS_PRODUCTION_DIR / f"{_safe(sym)}.pkl"
        if path.exists():
            available.append(sym)
    return available
