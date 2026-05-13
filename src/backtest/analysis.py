"""
Backtest analysis: breakdowns, sweeps, and comparisons.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Literal

import pandas as pd
import numpy as np

from src.backtest.costs import TransactionCostModel, ZeroCostModel
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.portfolio import Portfolio

log = logging.getLogger(__name__)


def per_symbol_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute win rate, profit factor, avg return, and trade count per symbol.
    Sorted by profit factor descending (best contributors first).
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for sym, grp in trades_df.groupby("symbol"):
        rets   = grp["return_pct"].values
        wins   = rets[rets > 0]
        losses = rets[rets <= 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")
        rows.append({
            "symbol":          sym,
            "num_trades":      len(rets),
            "win_rate_pct":    round(len(wins) / len(rets) * 100, 1),
            "avg_return_pct":  round(float(rets.mean() * 100), 2),
            "total_pnl":       round(float(grp["dollar_pnl"].sum()), 2),
            "profit_factor":   round(float(pf), 3),
            "avg_holding_days": round(float(grp["holding_days"].mean()), 1),
        })

    return pd.DataFrame(rows).sort_values("profit_factor", ascending=False).reset_index(drop=True)


def per_fold_attribution(
    trades_df:       pd.DataFrame,
    equity_curve_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Break down performance by CV fold.
    Shows whether certain folds drive all the return (regime dependence).
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for fold, grp in trades_df.groupby("fold"):
        rets     = grp["return_pct"].values
        wins     = rets[rets > 0]
        losses   = rets[rets <= 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")
        rows.append({
            "fold":            int(fold),
            "num_trades":      len(rets),
            "win_rate_pct":    round(len(wins) / len(rets) * 100, 1) if len(rets) else 0,
            "avg_return_pct":  round(float(rets.mean() * 100), 2) if len(rets) else 0,
            "total_pnl":       round(float(grp["dollar_pnl"].sum()), 2),
            "profit_factor":   round(float(pf), 3),
        })

    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def _run_single(
    predictions_df: pd.DataFrame,
    prices_dict:    dict,
    cost_model:     TransactionCostModel,
    threshold:      float,
    sizing_mode:    str,
    initial_cash:   float = 10_000.0,
) -> dict:
    """Helper: run one backtest and return its metrics dict."""
    portfolio = Portfolio(
        initial_cash=initial_cash,
        sizing_mode=sizing_mode,
        entry_threshold=threshold,
    )
    trades, equity = run_backtest(
        predictions_df=predictions_df,
        prices_dict=prices_dict,
        cost_model=cost_model,
        portfolio=portfolio,
        threshold=threshold,
    )
    m = compute_metrics(equity, trades)
    m["threshold"] = threshold
    m["sizing_mode"] = sizing_mode
    m["cost_model"] = type(cost_model).__name__
    return m


def threshold_sweep(
    predictions_df: pd.DataFrame,
    prices_dict:    dict,
    thresholds:     list[float] = [0.50, 0.55, 0.60, 0.65],
    cost_model:     TransactionCostModel | None = None,
    initial_cash:   float = 10_000.0,
) -> pd.DataFrame:
    """
    Run backtest at each threshold with realistic costs, fixed_fractional sizing.
    Returns DataFrame comparing key metrics at each threshold.

    WHY sweep thresholds?
      There is a fundamental trade-off:
      Low threshold (0.50) → many signals → higher exposure → more diversified,
        but lower signal quality → lower precision → more stop-outs.
      High threshold (0.65) → few signals → low exposure → concentrated,
        but higher signal quality → better precision → fewer stop-outs.
      The sweep finds where Sharpe is maximised along this frontier.
    """
    if cost_model is None:
        from src.backtest.costs import TransactionCostModel
        cost_model = TransactionCostModel()

    rows = []
    for t in thresholds:
        log.info("  threshold_sweep: t=%.2f", t)
        m = _run_single(predictions_df, prices_dict, cost_model, t, "fixed_fractional", initial_cash)
        rows.append(m)

    cols = ["threshold", "sharpe", "cagr_pct", "total_return_pct",
            "max_drawdown_pct", "win_rate_pct", "profit_factor",
            "num_trades", "exposure_pct", "small_sample_warning"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def sizing_comparison(
    predictions_df: pd.DataFrame,
    prices_dict:    dict,
    threshold:      float,
    cost_model:     TransactionCostModel | None = None,
    initial_cash:   float = 10_000.0,
) -> pd.DataFrame:
    """Compare the three sizing modes at a fixed threshold."""
    if cost_model is None:
        from src.backtest.costs import TransactionCostModel
        cost_model = TransactionCostModel()

    rows = []
    for mode in ["fixed_fractional", "confidence_weighted", "equal_weight"]:
        log.info("  sizing_comparison: mode=%s", mode)
        m = _run_single(predictions_df, prices_dict, cost_model, threshold, mode, initial_cash)
        rows.append(m)

    cols = ["sizing_mode", "sharpe", "cagr_pct", "total_return_pct",
            "max_drawdown_pct", "win_rate_pct", "profit_factor", "num_trades"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def cost_impact(
    predictions_df: pd.DataFrame,
    prices_dict:    dict,
    threshold:      float,
    sizing_mode:    str = "fixed_fractional",
    initial_cash:   float = 10_000.0,
) -> pd.DataFrame:
    """
    Compare zero-cost vs realistic-cost backtest at the same threshold.

    WHY this matters:
      If zero-cost Sharpe = 1.2 and realistic-cost Sharpe = -0.1, the
      strategy only works on paper.  We report both and flag the gap.
    """
    rows = []
    for cm in [ZeroCostModel(), TransactionCostModel()]:
        log.info("  cost_impact: cost_model=%s", type(cm).__name__)
        m = _run_single(predictions_df, prices_dict, cm, threshold, sizing_mode, initial_cash)
        rows.append(m)

    cols = ["cost_model", "sharpe", "cagr_pct", "total_return_pct",
            "max_drawdown_pct", "win_rate_pct", "profit_factor", "num_trades"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]
