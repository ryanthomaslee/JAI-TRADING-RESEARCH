"""
Aggregate OOF predictions from all four strategies, compute metrics,
produce comparison table and diagnostic plots.

Four strategies compared:
  global_xgb      global_lgbm
  per_symbol_xgb  per_symbol_lgbm

Usage:
    uv run python -m scripts.compare_models
"""
from __future__ import annotations
import logging, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import PARQUET_ENGINE
from src.models.evaluation import (
    compute_metrics,
    plot_pr_curve,
    plot_calibration_curve,
    plot_threshold_sweep,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

PRED_BASE  = ROOT / "data" / "processed" / "predictions"
PLOT_DIR   = ROOT / "data" / "processed" / "plots"
METRICS_OUT = ROOT / "data" / "processed" / "comparison_metrics.csv"

STRATEGIES = ["global_xgb", "global_lgbm", "per_symbol_xgb", "per_symbol_lgbm"]


def load_oof(strategy: str) -> pd.DataFrame | None:
    """Concatenate all per-symbol OOF parquets for a strategy."""
    base = PRED_BASE / strategy
    if not base.exists():
        log.warning("No predictions found for %s", strategy)
        return None

    frames = []
    for f in sorted(base.glob("*.parquet")):
        df = pd.read_parquet(f, engine=PARQUET_ENGINE)
        df["strategy"] = strategy
        frames.append(df)

    if not frames:
        log.warning("Empty predictions directory: %s", base)
        return None

    return pd.concat(frames)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all OOF predictions ──────────────────────────────────────────
    all_preds: dict[str, pd.DataFrame] = {}
    for strat in STRATEGIES:
        df = load_oof(strat)
        if df is not None:
            all_preds[strat] = df
            log.info("Loaded %s: %d rows", strat, len(df))

    if not all_preds:
        print("No OOF predictions found. Run train_global.py and train_per_symbol.py first.")
        sys.exit(1)

    # ── Compute metrics per (strategy, symbol) ────────────────────────────
    rows = []
    for strat, df in all_preds.items():
        for sym in sorted(df["symbol"].unique()):
            sdf = df[df["symbol"] == sym]
            if len(sdf) < 20 or sdf["y_true"].sum() < 5:
                log.warning("Skipping %s/%s — too few samples", strat, sym)
                continue
            m = compute_metrics(sdf["y_true"].values, sdf["proba_cal"].values)
            m["strategy"] = strat
            m["symbol"]   = sym
            rows.append(m)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_OUT, index=False)
    log.info("Saved metrics → %s", METRICS_OUT)

    # ── Print comparison table ────────────────────────────────────────────
    pivot = metrics_df.pivot_table(
        index="symbol",
        columns="strategy",
        values="pr_auc",
        aggfunc="mean",
    ).round(4)

    print("\n" + "═" * 75)
    print("  PR-AUC COMPARISON (higher = better, base rate varies by symbol)")
    print("═" * 75)
    print(pivot.to_string())
    print()

    # Supplementary table: precision at top decile
    pivot2 = metrics_df.pivot_table(
        index="symbol",
        columns="strategy",
        values="prec_top10",
        aggfunc="mean",
    ).round(4)
    print("  PRECISION @ TOP DECILE (signals you'd actually trade)")
    print("─" * 75)
    print(pivot2.to_string())
    print()

    # Brier score (lower = better)
    pivot3 = metrics_df.pivot_table(
        index="symbol",
        columns="strategy",
        values="brier",
        aggfunc="mean",
    ).round(4)
    print("  BRIER SCORE (lower = better, random = 0.25)")
    print("─" * 75)
    print(pivot3.to_string())
    print("═" * 75 + "\n")

    # ── Generate diagnostic plots per strategy × symbol ───────────────────
    for strat, df in all_preds.items():
        for sym in sorted(df["symbol"].unique()):
            sdf = df[df["symbol"] == sym]
            if len(sdf) < 20 or sdf["y_true"].sum() < 5:
                continue
            safe_sym = str(sym).replace("/", "_")
            y   = sdf["y_true"].values
            p   = sdf["proba_cal"].values
            lbl = f"{strat} | {sym}"

            plot_pr_curve(
                y, p,
                PLOT_DIR / strat / f"{safe_sym}_pr_curve.png",
                label=lbl,
            )
            plot_calibration_curve(
                y, p,
                PLOT_DIR / strat / f"{safe_sym}_calibration.png",
                label=lbl,
            )
            plot_threshold_sweep(
                y, p,
                PLOT_DIR / strat / f"{safe_sym}_threshold_sweep.png",
                label=lbl,
            )

    log.info("Plots saved to %s", PLOT_DIR)
    print(f"Metrics CSV:   {METRICS_OUT}")
    print(f"Plots saved:   {PLOT_DIR}/")


if __name__ == "__main__":
    main()
