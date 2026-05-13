"""
Phase 4.5 orchestration — universe filtering, cooldown sweep, buy-and-hold comparison.

NO model retraining.  Uses the same per_symbol_xgb OOF predictions as Phase 4.

Rounds:
  1. Universe filter  — WINNERS-only (JPM, AAPL, NVDA) vs all-10 baseline
  2. Threshold × cooldown grid search — find best (threshold, cooldown) combo
  3. Buy-and-hold comparison — did the ML system beat passive holding?
  4. Final operating-point verdict

Usage:
    uv run python -m scripts.run_phase4_5
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
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
from src.backtest.costs     import TransactionCostModel, ZeroCostModel
from src.backtest.engine    import run_backtest
from src.backtest.metrics   import compute_metrics
from src.backtest.portfolio import Portfolio
from src.backtest.benchmark import equal_weight_buy_hold
from src.backtest.analysis  import per_symbol_breakdown
from src.backtest.plots     import plot_equity_curve

PRED_BASE   = ROOT / "data" / "processed" / "predictions"
RESULTS_DIR = ROOT / "data" / "processed" / "phase4_5_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────
# Phase 4 showed JPM (PF=7.5), AAPL (PF=1.75), NVDA (PF=1.02) as positive.
# SOL/USDT (PF=0.69) and BNB/USDT (PF=0.00) dragged returns.
# WINNERS = symbols with PF > 1.0 in Phase 4 per-symbol breakdown.
WINNERS = ["JPM", "AAPL", "NVDA"]
LOSERS  = ["SPY", "QQQ", "MSFT", "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

STRATEGY = "per_symbol_xgb"


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders (same as run_backtest.py)
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
#  Run helper
# ─────────────────────────────────────────────────────────────────────────────

def _run(
    preds:         pd.DataFrame,
    prices:        dict[str, pd.DataFrame],
    threshold:     float,
    cooldown_days: int   = 0,
    cost_model=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run one backtest and return (trades, equity, metrics)."""
    if cost_model is None:
        cost_model = TransactionCostModel()
    p = Portfolio(
        initial_cash=10_000,
        max_open_positions=5,
        sizing_mode="fixed_fractional",
        entry_threshold=threshold,
    )
    trades, equity = run_backtest(
        predictions_df=preds,
        prices_dict=prices,
        cost_model=cost_model,
        portfolio=p,
        threshold=threshold,
        cooldown_days=cooldown_days,
    )
    m = compute_metrics(equity, trades)
    return trades, equity, m


def _fmt(val, pct=False):
    if val is None or (isinstance(val, float) and val != val):
        return "    N/A"
    if pct:
        return f"{val:+7.2f}%"
    if isinstance(val, float):
        return f"{val:>7.3f}"
    return str(val)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved plot → %s", path.name)


def plot_sharpe_heatmap(
    grid: dict[tuple[float, int], float],
    thresholds: list[float],
    cooldowns:  list[int],
    savepath: Path,
) -> None:
    """Heatmap of Sharpe: x = threshold, y = cooldown."""
    data = np.array(
        [[grid.get((t, c), float("nan")) for t in thresholds] for c in cooldowns]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=max(-1.0, float(np.nanmin(data))),
                   vmax=max(0.1,  float(np.nanmax(data))))

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_yticks(range(len(cooldowns)))
    ax.set_yticklabels([str(c) for c in cooldowns])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cooldown (days)")
    ax.set_title("Sharpe — threshold × cooldown grid (WINNERS universe)")

    for (j, i), val in np.ndenumerate(data):
        if not np.isnan(val):
            ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black")

    fig.colorbar(im, ax=ax, label="Sharpe")
    _savefig(fig, savepath)


def plot_equity_overlay(
    ml_equity:  pd.DataFrame,
    bh_equity:  pd.DataFrame,
    savepath:   Path,
    ml_label:   str = "ML strategy",
    bh_label:   str = "Buy & Hold (WINNERS)",
) -> None:
    """Equity curve overlay: ML strategy vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(11, 5))

    eq_ml = ml_equity["equity"]
    eq_bh = bh_equity["equity"]

    # Normalise to same start value for fair visual comparison
    scale = eq_ml.iloc[0] / eq_bh.iloc[0]

    ax.plot(eq_ml.index, eq_ml.values, lw=2, color="steelblue", label=ml_label)
    ax.plot(eq_bh.index, eq_bh.values * scale, lw=1.5, ls="--",
            color="darkorange", alpha=0.85, label=bh_label)

    ax.set_title("ML Strategy vs Buy-and-Hold (WINNERS: JPM, AAPL, NVDA)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, savepath)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 65)
    print("  PHASE 4.5  |  Universe filtering + cooldown + buy-and-hold")
    print("═" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading predictions and prices…")
    all_preds = load_predictions(STRATEGY)
    prices    = load_prices()

    # Phase 4 baseline metrics (all 10 symbols, threshold=0.65, no cooldown)
    PHASE4_SHARPE = 0.069
    PHASE4_CAGR   = 0.04

    # ── Filter predictions to WINNERS only ───────────────────────────────────
    winners_preds = all_preds[all_preds["symbol"].isin(WINNERS)]
    winners_prices = {sym: prices[sym] for sym in WINNERS if sym in prices}

    log.info(
        "WINNERS universe: %d symbols, %d prediction rows",
        len(WINNERS), len(winners_preds),
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  ROUND 1 — Universe filter: WINNERS-only vs Phase 4 baseline
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  ROUND 1 — Universe Filter (threshold=0.65, no cooldown)")
    print("─" * 65)

    trades_w, equity_w, m_w = _run(winners_preds, winners_prices, threshold=0.65)
    sym_breakdown = per_symbol_breakdown(trades_w)

    print(f"\n  {'Strategy':<25}  {'Sharpe':>7}  {'CAGR':>7}  {'MaxDD':>7}  "
          f"{'WinRate':>8}  {'Trades':>7}  {'PF':>6}")
    print("  " + "─" * 68)
    print(f"  {'Phase 4 (all 10 symbols)':<25}  "
          f"{PHASE4_SHARPE:>7.3f}  {PHASE4_CAGR:>+7.2f}%  "
          f"{'  -1.17%':>7}  {'41.9%':>8}  {'167':>7}  {'1.065':>6}")
    print(f"  {'WINNERS only (JPM/AAPL/NVDA)':<25}  "
          f"{m_w.get('sharpe', 0):>7.3f}  "
          f"{m_w.get('cagr_pct', 0):>+7.2f}%  "
          f"{m_w.get('max_drawdown_pct', 0):>+7.2f}%  "
          f"{m_w.get('win_rate_pct', 0):>7.1f}%  "
          f"{m_w.get('num_trades', 0):>7d}  "
          f"{m_w.get('profit_factor', 0):>6.3f}")

    print("\n  Per-symbol (WINNERS, threshold=0.65):")
    print(f"  {'Symbol':<10}  {'Trades':>7}  {'WinRate':>8}  {'PF':>6}  {'TotalPnL':>10}")
    print("  " + "─" * 46)
    for _, row in sym_breakdown.iterrows():
        print(f"  {row['symbol']:<10}  {int(row['num_trades']):>7d}  "
              f"{row['win_rate_pct']:>7.1f}%  {row['profit_factor']:>6.3f}  "
              f"${row['total_pnl']:>+8.0f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  ROUND 2 — Threshold × cooldown grid search
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  ROUND 2 — Threshold × Cooldown Grid Search (WINNERS universe)")
    print("─" * 65)

    thresholds = [0.60, 0.65, 0.70, 0.75]
    cooldowns  = [0, 3, 5, 10]

    grid:    dict[tuple[float, int], float] = {}
    metrics_grid: dict[tuple[float, int], dict] = {}

    total = len(thresholds) * len(cooldowns)
    done  = 0
    for t in thresholds:
        for c in cooldowns:
            done += 1
            log.info("  Grid [%2d/%d] threshold=%.2f cooldown=%d", done, total, t, c)
            _, _, m = _run(winners_preds, winners_prices, threshold=t, cooldown_days=c)
            grid[(t, c)] = m.get("sharpe", float("nan"))
            metrics_grid[(t, c)] = m

    # Print grid table
    print(f"\n  Sharpe heatmap (row=cooldown, col=threshold):")
    header = f"  {'cooldown':>10}" + "".join(f"  t={t:.2f}" for t in thresholds)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for c in cooldowns:
        row_str = f"  {c:>9}d" + "".join(
            f"  {grid.get((t, c), float('nan')):>6.3f}" for t in thresholds
        )
        print(row_str)

    # Best combo by Sharpe
    best_key   = max(grid, key=lambda k: (grid[k] if not np.isnan(grid[k]) else -999))
    best_sharpe = grid[best_key]
    best_t, best_c = best_key
    best_m = metrics_grid[best_key]

    print(f"\n  → Best combo: threshold={best_t:.2f}, cooldown={best_c}d  "
          f"(Sharpe={best_sharpe:.3f})")

    print(f"\n  Full metrics at best operating point:")
    print(f"    Sharpe  : {best_m.get('sharpe', 'N/A'):.3f}")
    print(f"    CAGR    : {best_m.get('cagr_pct', 0):+.2f}%")
    print(f"    Max DD  : {best_m.get('max_drawdown_pct', 0):+.2f}%")
    print(f"    Win Rate: {best_m.get('win_rate_pct', 0):.1f}%")
    print(f"    PF      : {best_m.get('profit_factor', 0):.3f}")
    print(f"    Trades  : {best_m.get('num_trades', 0)}")
    print(f"    Exposure: {best_m.get('exposure_pct', 0):.1f}%")

    # Save heatmap
    plot_sharpe_heatmap(
        grid, thresholds, cooldowns,
        savepath=RESULTS_DIR / "sharpe_heatmap.png",
    )

    # Re-run best to get equity curve for plotting
    trades_best, equity_best, _ = _run(
        winners_preds, winners_prices,
        threshold=best_t, cooldown_days=best_c,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  ROUND 3 — Buy-and-hold comparison (SAME date range as OOF predictions)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  ROUND 3 — Buy-and-Hold Comparison (same OOF date range)")
    print("─" * 65)

    oof_start = winners_preds.index.min()
    oof_end   = winners_preds.index.max()
    log.info("OOF date range: %s → %s", oof_start.date(), oof_end.date())

    bh_trades, bh_equity, bh_m = equal_weight_buy_hold(
        prices_dict=winners_prices,
        symbols=WINNERS,
        start_date=oof_start,
        end_date=oof_end,
        initial_cash=10_000.0,
        cost_model=TransactionCostModel(),
    )

    print(f"\n  {'Strategy':<25}  {'Sharpe':>7}  {'CAGR':>8}  {'MaxDD':>7}  "
          f"{'WinRate':>8}  {'Trades':>7}")
    print("  " + "─" * 67)
    print(f"  {'ML system (best combo)':<25}  "
          f"{best_m.get('sharpe', 0):>7.3f}  "
          f"{best_m.get('cagr_pct', 0):>+8.2f}%  "
          f"{best_m.get('max_drawdown_pct', 0):>+7.2f}%  "
          f"{best_m.get('win_rate_pct', 0):>7.1f}%  "
          f"{best_m.get('num_trades', 0):>7d}")
    print(f"  {'Buy-and-Hold (WINNERS)':<25}  "
          f"{bh_m.get('sharpe', 0):>7.3f}  "
          f"{bh_m.get('cagr_pct', 0):>+8.2f}%  "
          f"{bh_m.get('max_drawdown_pct', 0):>+7.2f}%  "
          f"{'N/A':>8}  "
          f"{len(bh_trades):>7d}")

    sharpe_delta = best_m.get("sharpe", 0) - bh_m.get("sharpe", 0)
    cagr_delta   = best_m.get("cagr_pct", 0) - bh_m.get("cagr_pct", 0)
    print(f"\n  Δ Sharpe (ML - B&H): {sharpe_delta:+.3f}")
    print(f"  Δ CAGR   (ML - B&H): {cagr_delta:+.2f}%")

    # Buy-and-hold per-symbol breakdown
    print("\n  Buy-and-hold per-symbol returns:")
    print(f"  {'Symbol':<10}  {'EntryPrice':>10}  {'ExitPrice':>10}  {'Return':>8}  {'PnL':>10}")
    print("  " + "─" * 50)
    for _, row in bh_trades.iterrows():
        print(f"  {row['symbol']:<10}  {row['entry_price']:>10.2f}  "
              f"{row['exit_price']:>10.2f}  "
              f"{row['return_pct'] * 100:>+7.1f}%  "
              f"${row['dollar_pnl']:>+8.0f}")

    # Equity curve overlay
    # Align buy-and-hold equity to ML equity index for clean overlay
    plot_equity_overlay(
        ml_equity=equity_best,
        bh_equity=bh_equity,
        savepath=RESULTS_DIR / "equity_overlay.png",
        ml_label=f"ML strategy (t={best_t:.2f}, cooldown={best_c}d)",
        bh_label="Buy & Hold equal-weight (JPM+AAPL+NVDA)",
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  ROUND 4 — Final operating-point verdict
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("  ROUND 4 — Final Operating-Point Verdict")
    print("─" * 65)

    ml_sharpe  = best_m.get("sharpe", 0)
    bh_sharpe  = bh_m.get("sharpe", 0)
    beats_bh   = ml_sharpe > bh_sharpe

    print(f"\n  ML best Sharpe : {ml_sharpe:.3f}  "
          f"(threshold={best_t:.2f}, cooldown={best_c}d, WINNERS only)")
    print(f"  B&H Sharpe     : {bh_sharpe:.3f}  (equal-weight JPM+AAPL+NVDA)")
    print(f"  Beats B&H      : {'YES' if beats_bh else 'NO'}")

    print()
    if ml_sharpe > 0.8 and beats_bh:
        verdict = "TRADEABLE BASELINE"
        detail  = (
            "Sharpe > 0.8 AND beats passive buy-and-hold.\n"
            "  Strategy has risk-adjusted edge over its own universe."
        )
    elif ml_sharpe > 0.5 and not beats_bh:
        verdict = "MARGINAL — strategy works but underperforms passive holding of same symbols"
        detail  = (
            f"Sharpe={ml_sharpe:.3f} is respectable but buy-and-hold Sharpe={bh_sharpe:.3f} "
            "is higher.\n"
            "  The ML system adds selection complexity without adding risk-adjusted return."
        )
    elif ml_sharpe > 0.5 and beats_bh:
        verdict = "MARGINAL — Sharpe > 0.5 and beats buy-and-hold, but below 0.8 threshold"
        detail  = (
            f"Sharpe={ml_sharpe:.3f} beats buy-and-hold ({bh_sharpe:.3f}) but is below 0.8.\n"
            "  Promising signal — proceed to Phase 5 feature improvements."
        )
    else:
        verdict = "NOT YET TRADEABLE — proceed to Phase 5 improvements"
        detail  = (
            f"Sharpe={ml_sharpe:.3f} is below the 0.5 minimum.\n"
            "  Strategy does not justify the complexity over passive holding."
        )

    print(f"  ► VERDICT: {verdict}")
    print(f"  {detail}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    summary = {
        "strategy":        STRATEGY,
        "winners_universe": WINNERS,
        "best_threshold":  best_t,
        "best_cooldown":   best_c,
        "ml_metrics":      {k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in best_m.items()},
        "bh_metrics":      {k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in bh_m.items()},
        "beats_buy_hold":  beats_bh,
        "verdict":         verdict,
    }
    import json
    with open(RESULTS_DIR / "phase4_5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    trades_best.to_parquet(RESULTS_DIR / "trades_best.parquet")
    equity_best.to_parquet(RESULTS_DIR / "equity_best.parquet")

    print(f"\n  Results → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
