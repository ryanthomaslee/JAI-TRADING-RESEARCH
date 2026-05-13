"""
Walk-forward training engine.

Two strategies, one interface:

  train_global():    Pool all symbols → one model per fold
  train_per_symbol(): One model per symbol per fold

WHY compare global vs per-symbol?
─────────────────────────────────
Global model (pooled):
  PRO: Sees patterns that are universal across assets (e.g. "high RSI
       everywhere tends to revert"), more data per fold (~14k rows vs 1.4k).
  CON: The model must also learn what makes NVDA different from SPY.
       Symbol dummies help, but the model may average-out asset-specific behaviour.

Per-symbol model:
  PRO: Captures idiosyncratic patterns (NVDA has much wilder momentum
       than SPY; BTC has 3× the volatility of any stock).
  CON: Only ~1400 rows per fold. XGBoost can work with this, but it's
       less stable than the pooled ~14k rows.

In practice, the answer often depends on the asset: liquid indexes (SPY,
QQQ) tend to behave like the market and benefit from global pooling;
individual stocks and crypto tend to have idiosyncratic regimes that
benefit from per-symbol models.  The comparison script will show us which.

Calibration data split:
  Inside each fold, the training data (everything before the embargo gap)
  is further split 80/20:
    - 80%: model fitting (XGBoost/LGBM trees)
    - 20%: calibration (isotonic regression mapping)
  The split is chronological (first 80%, last 20%) — never random.
  WHY chronological?  Randomly shuffling would cause the model to be
  calibrated on data that predates its own training — leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from config.settings import FEATURES_DIR, PARQUET_ENGINE, PARQUET_COMPRESSION
from src.models.base import Model
from src.models.splits import PurgedWalkForwardSplit

log = logging.getLogger(__name__)

# Columns to use as features (everything except OHLCV and label metadata)
_OHLCV_COLS    = {"open", "high", "low", "close", "volume"}
_LABEL_COLS    = {"label", "barrier_hit", "days_held", "entry_price",
                  "exit_price", "realized_return"}
_NON_FEATURE   = _OHLCV_COLS | _LABEL_COLS


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE]


def _load_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _binary_target(df: pd.DataFrame) -> np.ndarray:
    """
    Convert triple-barrier label {-1, 0, 1} to binary {0, 1}.

    Binary target = 1 iff label == +1 (target hit).
    Labels -1 (stop) and 0 (time) are both treated as "not a buy signal."

    WHY binary instead of multiclass?
      We want a BUY PROBABILITY.  The distinction between "stopped out"
      and "time barrier" matters for risk management (Phase 4) but not
      for the entry signal.  Multiclass would force the model to jointly
      predict three outcomes — a much harder problem with less signal per class.
    """
    return (df["label"] == 1).astype(int).values


def _split_train_cal(
    X: np.ndarray,
    y: np.ndarray,
    cal_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological 80/20 split of training data into fit + calibration sets.
    The LAST cal_fraction rows become the calibration set.

    WHY last 20% as calibration (not first 20%)?
      The calibration set should be as close as possible to the test set
      in time — it's learning to correct the score-to-probability mapping
      in the recent regime, which is more relevant than the old regime.
    """
    n_cal   = max(50, int(len(X) * cal_fraction))
    n_fit   = len(X) - n_cal
    return X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:]


def _save_predictions(preds_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(preds_df, preserve_index=True)
    pq.write_table(table, out_path, compression=PARQUET_COMPRESSION)


# ─────────────────────────────────────────────────────────────────────────────
#  Global training (all symbols pooled)
# ─────────────────────────────────────────────────────────────────────────────

def train_global(
    model_cls:   Type[Model],
    splitter:    PurgedWalkForwardSplit,
    strategy_name: str = "global",
    features_dir:  Path = FEATURES_DIR,
    save_models:   bool = True,
    **model_kwargs,
) -> list[tuple[int, Model, pd.DataFrame]]:
    """
    Train one model per fold on the pooled multi-symbol dataset.

    Returns list of (fold_idx, fitted_model, predictions_df) where
    predictions_df has columns [y_true, proba_raw, proba_cal, symbol, fold].

    The predictions_df for all folds concatenated = out-of-fold (OOF)
    predictions — the honest estimate of live performance.
    """
    log.info("=== train_global | strategy=%s | model=%s ===",
             strategy_name, model_cls.__name__)

    # ── Load and pool all symbols ──────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    symbols: list[str] = []

    cfg_path = Path(__file__).parent.parent.parent / "config" / "universe.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    stock_syms  = [s["symbol"]   for s in cfg.get("stocks", [])]
    crypto_syms = [s["ccxt_id"]  for s in cfg.get("crypto", [])]
    all_syms    = stock_syms + crypto_syms

    for sym in all_syms:
        safe = sym.replace("/", "_").replace(":", "_")
        path = features_dir / f"{safe}.parquet"
        if not path.exists():
            log.warning("Skipping %s — %s not found", sym, path)
            continue
        df = _load_symbol(path)
        df["_symbol"] = sym
        frames.append(df)
        symbols.append(sym)
        log.info("  Loaded %s: %d rows", sym, len(df))

    if not frames:
        raise RuntimeError("No feature files found in " + str(features_dir))

    # Concatenate and sort by timestamp (cross-asset chronological order)
    pool = pd.concat(frames).sort_index()
    log.info("Pooled dataset: %d rows across %d symbols", len(pool), len(symbols))

    # ── One-hot encode symbol ──────────────────────────────────────────────
    # WHY one-hot instead of label encoding?
    #   Label encoding (SPY=0, QQQ=1, …) implies ordinal relationship.
    #   One-hot lets the model learn independent behaviour per symbol without
    #   the implicit ordering.  10 symbols → 10 dummy columns.
    symbol_dummies = pd.get_dummies(pool["_symbol"], prefix="sym", dtype=float)
    pool = pd.concat([pool.drop(columns=["_symbol"]), symbol_dummies], axis=1)

    feat_cols = _feature_cols(pool) + [c for c in pool.columns if c.startswith("sym_")]
    X_all = pool[feat_cols].values
    y_all = _binary_target(pool)

    results: list[tuple[int, Model, pd.DataFrame]] = []
    all_fold_preds: list[pd.DataFrame] = []

    for fold, (tr_idx, te_idx) in enumerate(splitter.split(pool)):
        log.info("--- Fold %d | train=%d  test=%d ---", fold, len(tr_idx), len(te_idx))

        X_tr_full, y_tr_full = X_all[tr_idx], y_all[tr_idx]
        X_te,      y_te      = X_all[te_idx], y_all[te_idx]

        # Chronological 80/20 split of training data for fit + calibration
        X_fit, y_fit, X_cal, y_cal = _split_train_cal(X_tr_full, y_tr_full)

        # Use cal set as both validation (early stopping) and calibration
        model = model_cls(**model_kwargs)
        model.fit(X_fit, y_fit, X_val=X_cal, y_val=y_cal, X_cal=X_cal, y_cal=y_cal)

        # Predict on test set
        raw_proba = model._raw_proba(X_te)
        cal_proba = model.predict_proba(X_te)

        test_rows = pool.iloc[te_idx]
        fold_preds = pd.DataFrame(
            {
                "y_true":    y_te,
                "proba_raw": raw_proba,
                "proba_cal": cal_proba,
                "symbol":    test_rows["_symbol"].values if "_symbol" in test_rows.columns
                             else pool.iloc[te_idx].get("_symbol",
                             pd.Series(["unknown"] * len(te_idx))).values,
                "fold":      fold,
            },
            index=test_rows.index,
        )

        # Recover symbol from the original pool before dummy encoding
        sym_cols = [c for c in pool.columns if c.startswith("sym_")]
        if sym_cols:
            sym_matrix = pool.iloc[te_idx][sym_cols]
            sym_labels = sym_matrix.idxmax(axis=1).str.replace("sym_", "", regex=False)
            fold_preds["symbol"] = sym_labels.values

        all_fold_preds.append(fold_preds)
        results.append((fold, model, fold_preds))

        if save_models:
            model_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "models"
            model.save(model_dir / f"{strategy_name}_{model_cls.__name__}_fold{fold}.pkl")

        pos_rate = y_te.mean()
        prec_60  = (y_te[cal_proba >= 0.60].mean()
                    if (cal_proba >= 0.60).sum() > 0 else float("nan"))
        log.info(
            "  Fold %d: base_rate=%.3f  prec@0.6=%.3f  coverage@0.6=%.3f",
            fold, pos_rate, prec_60, (cal_proba >= 0.60).mean(),
        )

    # ── Save pooled OOF predictions per symbol ─────────────────────────────
    oof_all = pd.concat(all_fold_preds)
    pred_base = (Path(__file__).parent.parent.parent
                 / "data" / "processed" / "predictions" / strategy_name)

    for sym in oof_all["symbol"].unique():
        sym_preds = oof_all[oof_all["symbol"] == sym]
        safe = str(sym).replace("/", "_").replace(":", "_")
        _save_predictions(sym_preds, pred_base / f"{safe}.parquet")
        log.info("Saved OOF predictions for %s: %d rows → %s",
                 sym, len(sym_preds), pred_base / f"{safe}.parquet")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Per-symbol training
# ─────────────────────────────────────────────────────────────────────────────

def train_per_symbol(
    model_cls:    Type[Model],
    splitter:     PurgedWalkForwardSplit,
    strategy_name: str = "per_symbol",
    features_dir:  Path = FEATURES_DIR,
    save_models:   bool = True,
    **model_kwargs,
) -> dict[str, list[tuple[int, Model, pd.DataFrame]]]:
    """
    Train one model per symbol per fold.

    Returns {symbol: [(fold_idx, fitted_model, predictions_df), …]}
    """
    log.info("=== train_per_symbol | strategy=%s | model=%s ===",
             strategy_name, model_cls.__name__)

    cfg_path = Path(__file__).parent.parent.parent / "config" / "universe.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    stock_syms  = [s["symbol"]   for s in cfg.get("stocks", [])]
    crypto_syms = [s["ccxt_id"]  for s in cfg.get("crypto", [])]
    all_syms    = stock_syms + crypto_syms

    all_results: dict[str, list] = {}

    for sym in all_syms:
        safe = sym.replace("/", "_").replace(":", "_")
        path = features_dir / f"{safe}.parquet"
        if not path.exists():
            log.warning("Skipping %s — file not found", sym)
            continue

        df = _load_symbol(path)
        log.info("--- Symbol: %s | %d rows ---", sym, len(df))

        feat_cols = _feature_cols(df)
        X_all     = df[feat_cols].values
        y_all     = _binary_target(df)

        sym_results:    list = []
        sym_fold_preds: list = []

        for fold, (tr_idx, te_idx) in enumerate(splitter.split(df)):
            log.info("  Fold %d | train=%d  test=%d", fold, len(tr_idx), len(te_idx))

            X_tr_full, y_tr_full = X_all[tr_idx], y_all[tr_idx]
            X_te,      y_te      = X_all[te_idx], y_all[te_idx]

            X_fit, y_fit, X_cal, y_cal = _split_train_cal(X_tr_full, y_tr_full)

            model = model_cls(**model_kwargs)
            model.fit(X_fit, y_fit, X_val=X_cal, y_val=y_cal, X_cal=X_cal, y_cal=y_cal)

            raw_proba = model._raw_proba(X_te)
            cal_proba = model.predict_proba(X_te)

            fold_preds = pd.DataFrame(
                {
                    "y_true":    y_te,
                    "proba_raw": raw_proba,
                    "proba_cal": cal_proba,
                    "symbol":    sym,
                    "fold":      fold,
                },
                index=df.index[te_idx],
            )
            sym_fold_preds.append(fold_preds)
            sym_results.append((fold, model, fold_preds))

            if save_models:
                model_dir = (Path(__file__).parent.parent.parent
                             / "data" / "processed" / "models")
                model.save(
                    model_dir / f"{strategy_name}_{safe}_{model_cls.__name__}_fold{fold}.pkl"
                )

            pos_rate = y_te.mean()
            prec_60  = (y_te[cal_proba >= 0.60].mean()
                        if (cal_proba >= 0.60).sum() > 0 else float("nan"))
            log.info(
                "    base_rate=%.3f  prec@0.6=%.3f  cov@0.6=%.3f",
                pos_rate, prec_60, (cal_proba >= 0.60).mean(),
            )

        # Save OOF predictions for this symbol
        if sym_fold_preds:
            oof_sym = pd.concat(sym_fold_preds)
            pred_base = (Path(__file__).parent.parent.parent
                         / "data" / "processed" / "predictions" / strategy_name)
            _save_predictions(oof_sym, pred_base / f"{safe}.parquet")

        all_results[sym] = sym_results

    return all_results
