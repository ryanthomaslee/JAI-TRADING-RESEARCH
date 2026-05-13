"""
Full Phase 4 backtest orchestration.

Uses per_symbol_xgb OOF predictions (best strategy from Phase 3).
Runs: threshold sweep → sizing comparison → cost impact → breakdowns → plots.

Usage:
    uv run python -m scripts.run_backtest
    uv run python -m scripts.run_backtest --strategy global_xgb
    uv run python -m scripts.run_backtest --threshold 0.60  # skip sweep, use fixed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

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

from config.settings import RAW_STOCKS_DIR, RAW_CRYPTO_DIR, PARQUET_ENGINE
from src.backtest.costs    import TransactionCostModel, ZeroCostModel
from src.backtest.engine   import run_backtest
from src.backtest.metrics  import compute_metrics
from src.backtest.portfolio import Portfolio
from src.backtest.analysis import (
    threshold_sweep, sizing_comparison, cost_impact,
    per_symbol_breakdown, per_fold_attribution,
)
from src.backtest.plots import (
    plot_equity_curve, plot_drawdown_underwater,
    plot_returns_distribution, plot_per_symbol_pnl,
)

PRED_BASE   = ROOT / "data" / "processed" / "predictions"
RESULTS_DIR = ROOT / "data" / "processed" / "backtest_results"
PLOT_DIR    = ROOT / "data" / "processed" / "backtest_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"}


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(strategy: str) -> pd.DataFrame:
    base = PRED_BASE / strategy
    frames = []
    for f in sorted(base.glob("*.parquet")):
        df = pd.read_parquet(f, engine=PARQUET_ENGINE)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No prediction files in {base}")
    combined = pd.concat(frames)
    combined.index = pd.to_datetime(combined.index).normalize()
    return combined.sort_index()


def load_prices() -> dict[str, pd.DataFrame]:
    prices = {}
    cfg_path = ROOT / "config" / "universe.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for s in cfg.get("stocks", []):
        sym  = s["symbol"]
        path = RAW_STOCKS_DIR / f"{sym}.parquet"
        if path.exists():
            df = pd.read_parquet(path, engine=PARQUET_ENGINE)
            if df.index.tz is None: df.index = df.index.tz_localize("UTC")
            df.index = df.index.normalize()
            prices[sym] = df

    for s in cfg.get("crypto", []):
        sym  = s["ccxt_id"]
        safe = sym.replace("/", "_")
        path = RAW_CRYPTO_DIR / f"{safe}.parquet"
        if path.exists():
            df = pd.read_parquet(path, engine=PARQUET_ENGINE)
            if df.index.tz is None: df.index = df.index.tz_localize("UTC")
            df.index = df.index.normalize()
            prices[sym] = df

    return prices


# ─────────────────────────────────────────────────────────────────────────────
#  Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, pct=False, dollar=False):
    if val is None or (isinstance(val, float) and (val != val)):
        return "  N/A"
    if pct:
        return f"{val:+7.2f}%"
    if dollar:
        return f"${val:>8,.0f}"
    if isinstance(val, bool):
        return "  YES" if val else "   no"
    if isinstance(val, float):
        return f"{val:>7.3f}"
    return str(val)


def print_headline_table(label: str, m: dict) -> None:
    """Print the single-row headline metrics dict in a readable format."""
    warn_ss  = " ⚠ SMALL SAMPLE (<2yr)" if m.get("small_sample_warning") else ""
    warn_dd  = " ⚠ UNTRADEABLE DD>50%" if m.get("untradeable_drawdown") else ""
    warn_neg = " ⚠ NEGATIVE SHARPE" if m.get("sharpe", 0) < 0 else ""

    print(f"\n  {label}{warn_ss}{warn_dd}{warn_neg}")
    print(f"  {'─'*60}")
    print(f"  Net Return    : {_fmt(m.get('total_return_pct'), pct=True)}")
    print(f"  CAGR          : {_fmt(m.get('cagr_pct'), pct=True)}")
    print(f"  Sharpe        : {_fmt(m.get('sharpe'))}")
    print(f"  Sortino       : {_fmt(m.get('sortino'))}")
    print(f"  Calmar        : {_fmt(m.get('calmar'))}")
    print(f"  Max Drawdown  : {_fmt(m.get('max_drawdown_pct'), pct=True)}")
    print(f"  Max DD Days   : {m.get('max_dd_duration_days', 'N/A')}")
    print(f"  Win Rate      : {_fmt(m.get('win_rate_pct'), pct=True)}")
    print(f"  Profit Factor : {_fmt(m.get('profit_factor'))}")
    print(f"  # Trades      : {m.get('num_trades', 0)}")
    print(f"  Avg Hold Days : {_fmt(m.get('avg_holding_days'))}")
    print(f"  Exposure      : {_fmt(m.get('exposure_pct'), pct=True)}")
    print(f"  OOS Days      : {m.get('n_days_oos', 'N/A')}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",  default="per_symbol_xgb")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Skip sweep and use this fixed threshold")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    args = parser.parse_args()

    log.info("Loading predictions for strategy: %s", args.strategy)
    preds  = load_predictions(args.strategy)
    prices = load_prices()
    log.info("Loaded %d prediction rows, %d symbols in price dict",
             len(preds), len(prices))

    cost_model = TransactionCostModel()

    print("\n" + "═"*65)
    print(f"  PHASE 4 BACKTEST  |  strategy={args.strategy}  |  cash=${args.cash:,.0f}")
    print("═"*65)

    # ── A) Threshold sweep ────────────────────────────────────────────────────
    if args.threshold is None:
        log.info("Running threshold sweep…")
        thresholds = [0.50, 0.55, 0.60, 0.65]
        sweep_df = threshold_sweep(preds, prices, thresholds=thresholds,
                                   cost_model=cost_model, initial_cash=args.cash)
        print("\n  THRESHOLD SWEEP (realistic costs, fixed_fractional sizing)")
        print("  " + "─"*63)
        print(f"  {'Thresh':>7}  {'Sharpe':>7}  {'CAGR%':>7}  {'MaxDD%':>8}  "
              f"{'WinRate%':>9}  {'PF':>6}  {'Trades':>7}  {'Exp%':>6}")
        print("  " + "─"*63)
        for _, row in sweep_df.iterrows():
            warn = " ⚠" if row.get("small_sample_warning") else ""
            print(f"  {row['threshold']:>7.2f}  {row.get('sharpe', 0):>7.3f}  "
                  f"{row.get('cagr_pct', 0):>+7.2f}  {row.get('max_drawdown_pct', 0):>+8.2f}  "
                  f"{row.get('win_rate_pct', 0):>9.1f}  {row.get('profit_factor', 0):>6.3f}  "
                  f"{row.get('num_trades', 0):>7d}  {row.get('exposure_pct', 0):>6.1f}{warn}")

        # Pick threshold with best Sharpe
        best_row    = sweep_df.loc[sweep_df["sharpe"].idxmax()]
        op_threshold = float(best_row["threshold"])
        print(f"\n  → Operating threshold: {op_threshold:.2f}  "
              f"(Sharpe={best_row['sharpe']:.3f})")

        # Warn if best Sharpe is negative
        if float(best_row["sharpe"]) < 0:
            print("\n  ⚠  WARNING: ALL THRESHOLDS PRODUCE NEGATIVE SHARPE AT REALISTIC COSTS.")
            print("     The strategy does not appear to be profitable after transaction costs.")
    else:
        op_threshold = args.threshold
        sweep_df     = pd.DataFrame()
        print(f"\n  Using fixed threshold: {op_threshold:.2f}")

    # ── B) Sizing comparison ──────────────────────────────────────────────────
    log.info("Running sizing comparison at threshold=%.2f…", op_threshold)
    sizing_df = sizing_comparison(preds, prices, threshold=op_threshold,
                                  cost_model=cost_model, initial_cash=args.cash)
    print(f"\n  SIZING COMPARISON (threshold={op_threshold:.2f}, realistic costs)")
    print("  " + "─"*63)
    print(f"  {'Mode':<25}  {'Sharpe':>7}  {'CAGR%':>7}  {'MaxDD%':>8}  {'Trades':>7}")
    print("  " + "─"*63)
    for _, row in sizing_df.iterrows():
        print(f"  {row['sizing_mode']:<25}  {row.get('sharpe', 0):>7.3f}  "
              f"{row.get('cagr_pct', 0):>+7.2f}  {row.get('max_drawdown_pct', 0):>+8.2f}  "
              f"{row.get('num_trades', 0):>7d}")

    best_sizing = sizing_df.loc[sizing_df["sharpe"].idxmax(), "sizing_mode"]
    print(f"\n  → Best sizing mode: {best_sizing}")

    # ── C) Cost impact ────────────────────────────────────────────────────────
    log.info("Running cost impact comparison…")
    cost_df = cost_impact(preds, prices, threshold=op_threshold,
                          sizing_mode=best_sizing, initial_cash=args.cash)
    print(f"\n  COST IMPACT (threshold={op_threshold:.2f}, {best_sizing} sizing)")
    print("  " + "─"*63)
    print(f"  {'Cost Model':<25}  {'Sharpe':>7}  {'CAGR%':>7}  {'MaxDD%':>8}  {'PF':>6}")
    print("  " + "─"*63)
    for _, row in cost_df.iterrows():
        print(f"  {row['cost_model']:<25}  {row.get('sharpe', 0):>7.3f}  "
              f"{row.get('cagr_pct', 0):>+7.2f}  {row.get('max_drawdown_pct', 0):>+8.2f}  "
              f"{row.get('profit_factor', 0):>6.3f}")

    # Explicit warning if strategy only works at zero cost
    zero_sharpe = float(cost_df[cost_df["cost_model"] == "ZeroCostModel"]["sharpe"].iloc[0])
    real_sharpe = float(cost_df[cost_df["cost_model"] == "TransactionCostModel"]["sharpe"].iloc[0])
    if zero_sharpe > 0.5 and real_sharpe < 0:
        print("\n  ⚠  WARNING: Strategy only works at zero cost.")
        print("     Zero-cost Sharpe is positive but realistic-cost Sharpe is negative.")
        print("     DO NOT TRADE THIS STRATEGY without reducing turnover or costs.")

    # ── D) Full run at operating threshold + best sizing ─────────────────────
    log.info("Running full backtest at operating settings…")
    portfolio_real = Portfolio(initial_cash=args.cash, sizing_mode=best_sizing,
                               entry_threshold=op_threshold)
    trades_real, equity_real = run_backtest(
        predictions_df=preds, prices_dict=prices,
        cost_model=TransactionCostModel(),
        portfolio=portfolio_real, threshold=op_threshold,
    )
    metrics_real = compute_metrics(equity_real, trades_real)

    portfolio_zero = Portfolio(initial_cash=args.cash, sizing_mode=best_sizing,
                               entry_threshold=op_threshold)
    trades_zero, equity_zero = run_backtest(
        predictions_df=preds, prices_dict=prices,
        cost_model=ZeroCostModel(),
        portfolio=portfolio_zero, threshold=op_threshold,
    )
    metrics_zero = compute_metrics(equity_zero, trades_zero)

    print("\n" + "═"*65)
    print("  HEADLINE RESULTS")
    print("═"*65)
    print_headline_table("Zero Cost", metrics_zero)
    print_headline_table("Realistic Cost", metrics_real)

    # ── E) Per-symbol breakdown ───────────────────────────────────────────────
    sym_df = per_symbol_breakdown(trades_real)
    if not sym_df.empty:
        print("\n  PER-SYMBOL BREAKDOWN (realistic costs)")
        print("  " + "─"*70)
        print(f"  {'Symbol':<14}  {'Trades':>7}  {'WinRate%':>9}  {'AvgRet%':>8}  "
              f"{'TotalPnL':>10}  {'PF':>6}  {'AvgDays':>8}")
        print("  " + "─"*70)
        for _, row in sym_df.iterrows():
            pnl_str = f"${row['total_pnl']:>+8.0f}"
            print(f"  {row['symbol']:<14}  {int(row['num_trades']):>7d}  "
                  f"{row['win_rate_pct']:>9.1f}  {row['avg_return_pct']:>+8.2f}  "
                  f"{pnl_str:>10}  {row['profit_factor']:>6.3f}  "
                  f"{row['avg_holding_days']:>8.1f}")

    # ── F) Per-fold attribution ───────────────────────────────────────────────
    fold_df = per_fold_attribution(trades_real, equity_real)
    if not fold_df.empty:
        print("\n  PER-FOLD ATTRIBUTION (realistic costs)")
        print("  " + "─"*55)
        print(f"  {'Fold':>5}  {'Trades':>7}  {'WinRate%':>9}  {'AvgRet%':>8}  "
              f"{'TotalPnL':>10}  {'PF':>6}")
        print("  " + "─"*55)
        for _, row in fold_df.iterrows():
            pnl_str = f"${row['total_pnl']:>+8.0f}"
            print(f"  {int(row['fold']):>5d}  {int(row['num_trades']):>7d}  "
                  f"{row['win_rate_pct']:>9.1f}  {row['avg_return_pct']:>+8.2f}  "
                  f"{pnl_str:>10}  {row['profit_factor']:>6.3f}")

    # ── G) Plots ──────────────────────────────────────────────────────────────
    log.info("Generating plots…")
    # Load SPY for buy-and-hold baseline
    spy_baseline = None
    if "SPY" in prices:
        spy_prices = prices["SPY"]
        mask = spy_prices.index.isin(equity_real.index)
        if mask.any():
            spy_baseline = spy_prices.loc[mask, "close"]

    if not equity_real.empty:
        plot_equity_curve(equity_real, PLOT_DIR / "equity_curve.png",
                          baseline_equity=spy_baseline,
                          title=f"Strategy Equity Curve — {args.strategy} @ {op_threshold:.2f}")
        plot_drawdown_underwater(equity_real, PLOT_DIR / "drawdown.png")

    if not trades_real.empty:
        plot_returns_distribution(trades_real, PLOT_DIR / "returns_dist.png")
        plot_per_symbol_pnl(trades_real, PLOT_DIR / "per_symbol_pnl.png")

    # ── H) Save results ───────────────────────────────────────────────────────
    summary = {
        "strategy":       args.strategy,
        "threshold":      op_threshold,
        "sizing_mode":    best_sizing,
        "realistic_cost": metrics_real,
        "zero_cost":      metrics_zero,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if not trades_real.empty:
        table = pa.Table.from_pandas(trades_real)
        pq.write_table(table, RESULTS_DIR / "trades.parquet")

    if not equity_real.empty:
        table = pa.Table.from_pandas(equity_real, preserve_index=True)
        pq.write_table(table, RESULTS_DIR / "equity_curve.parquet")

    print(f"\n  Results → {RESULTS_DIR}/")
    print(f"  Plots   → {PLOT_DIR}/")
    print()

    # Final cost-only verdict
    if metrics_real.get("sharpe", 0) < 0:
        print("  ⚠  FINAL VERDICT: Negative Sharpe at realistic costs.")
        print("     Review per-symbol breakdown to find which names drag performance.")
    elif metrics_real.get("sharpe", 0) < 0.5:
        print("  ⚠  FINAL VERDICT: Positive but weak Sharpe (<0.5) at realistic costs.")
        print("     Strategy needs improvement before live deployment.")
    else:
        print("  ✓  FINAL VERDICT: Positive Sharpe at realistic costs.")


if __name__ == "__main__":
    main()
