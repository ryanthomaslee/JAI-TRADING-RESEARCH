"""
Build the full feature + label dataset for all symbols.

This script:
  1. Loads each symbol's raw Parquet from data/raw/
  2. Runs build_features() → 22 technical/volatility/volume features
  3. Runs triple_barrier_labels() → labels, barrier metadata
  4. Joins features + labels on the shared index
  5. Saves the joined DataFrame to data/features/{symbol}.parquet

WHY join features and labels AFTER computing them separately?
  Keeping them separate until this step enforces the module boundary:
  feature code can NEVER accidentally reference label columns, and label
  code can NEVER accidentally reference feature columns.  The join here
  is the one legitimate place where they meet.

WHY inner join?
  Features lose the first ~200 rows (warmup) and labels lose the last 20
  rows.  Inner join keeps only rows that have BOTH valid features AND a
  valid label — the intersection is the clean training dataset.

Usage:
    uv run python -m scripts.build_dataset
    uv run python -m scripts.build_dataset --symbol SPY  # single symbol
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    RAW_STOCKS_DIR, RAW_CRYPTO_DIR, FEATURES_DIR,
    PARQUET_ENGINE, PARQUET_COMPRESSION,
)
from src.features.pipeline import build_features, feature_names
from src.labels.triple_barrier import triple_barrier_labels

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def load_raw(symbol: str, is_crypto: bool) -> pd.DataFrame | None:
    """Load a symbol's raw Parquet; return None if not found."""
    safe = symbol.replace("/", "_").replace(":", "_")
    directory = RAW_CRYPTO_DIR if is_crypto else RAW_STOCKS_DIR
    path = directory / f"{safe}.parquet"

    if not path.exists():
        log.warning("[%s] Raw file not found: %s", symbol, path)
        return None

    df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def build_and_save(symbol: str, df: pd.DataFrame) -> dict:
    """
    Run the full feature + label pipeline for one symbol.
    Returns a summary dict for the final report.
    """
    log.info("[%s] Building features…", symbol)
    feat_df = build_features(df)

    log.info("[%s] Computing labels…", symbol)
    label_df = triple_barrier_labels(df)

    # ── Inner join: only rows with BOTH valid features AND a label ────────────
    # WHY inner join?
    #   - Feature warmup drops first ~200 rows (no SMA-200 yet)
    #   - Label module drops last 20 rows (incomplete forward window)
    #   - Inner join is the mathematically correct intersection
    combined = feat_df.join(label_df, how="inner")

    if combined.empty:
        log.error("[%s] Inner join produced empty DataFrame!", symbol)
        return {"symbol": symbol, "rows": 0, "error": "empty after join"}

    # ── Validate no NaNs in feature columns ───────────────────────────────────
    feats = feature_names(feat_df)
    nan_count = combined[feats].isna().sum().sum()
    if nan_count > 0:
        log.warning("[%s] %d NaN values in features after join", symbol, nan_count)

    # ── Save to data/features/ ────────────────────────────────────────────────
    safe = symbol.replace("/", "_").replace(":", "_")
    out_path = FEATURES_DIR / f"{safe}.parquet"

    table = pa.Table.from_pandas(combined, preserve_index=True)
    pq.write_table(table, out_path, compression=PARQUET_COMPRESSION)
    log.info("[%s] Saved %d rows, %d features → %s", symbol, len(combined), len(feats), out_path)

    # ── Label distribution ────────────────────────────────────────────────────
    label_counts = combined["label"].value_counts().sort_index()
    return {
        "symbol":   symbol,
        "rows":     len(combined),
        "features": len(feats),
        "label_-1": label_counts.get(-1, 0),
        "label_0":  label_counts.get(0, 0),
        "label_+1": label_counts.get(1, 0),
        "start":    combined.index[0].date(),
        "end":      combined.index[-1].date(),
    }


def print_summary(results: list[dict]) -> None:
    header = (
        f"\n{'Symbol':<14} {'Rows':>6} {'Feats':>6} "
        f"{'Stop(-1)':>10} {'Time(0)':>9} {'Target(+1)':>11} "
        f"{'Start':<12} {'End':<12}"
    )
    print("\n" + "═" * 85)
    print("  PHASE 2 DATASET SUMMARY")
    print("═" * 85)
    print(header)
    print("─" * 85)
    for r in results:
        if "error" in r:
            print(f"  {r['symbol']:<12}  ERROR: {r['error']}")
            continue
        total = r["label_-1"] + r["label_0"] + r["label_+1"]
        print(
            f"  {r['symbol']:<12} {r['rows']:>6} {r['features']:>6} "
            f"  {r['label_-1']:>6} ({r['label_-1']/total*100:>4.0f}%) "
            f"  {r['label_0']:>6} ({r['label_0']/total*100:>4.0f}%) "
            f"  {r['label_+1']:>6} ({r['label_+1']/total*100:>4.0f}%)  "
            f"  {str(r['start']):<12} {str(r['end']):<12}"
        )
    print("═" * 85 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature+label dataset.")
    parser.add_argument("--symbol", help="Process a single symbol only")
    args = parser.parse_args()

    cfg_path = ROOT / "config" / "universe.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    universe = [
        (s["symbol"],   False) for s in cfg.get("stocks", [])
    ] + [
        (s["ccxt_id"],  True)  for s in cfg.get("crypto", [])
    ]

    if args.symbol:
        universe = [(sym, is_c) for sym, is_c in universe if sym == args.symbol]
        if not universe:
            print(f"Symbol '{args.symbol}' not found in universe.yaml")
            sys.exit(1)

    results = []
    for symbol, is_crypto in universe:
        df = load_raw(symbol, is_crypto)
        if df is None:
            results.append({"symbol": symbol, "rows": 0, "error": "raw file missing"})
            continue
        try:
            summary = build_and_save(symbol, df)
            results.append(summary)
        except Exception as exc:
            log.exception("[%s] Failed: %s", symbol, exc)
            results.append({"symbol": symbol, "rows": 0, "error": str(exc)})

    print_summary(results)


if __name__ == "__main__":
    main()
