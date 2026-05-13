"""
Backtest diagnostic plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def _savefig(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve(
    equity_curve:       pd.DataFrame,
    savepath:           str | Path,
    baseline_equity:    pd.Series | None = None,
    title:              str = "Equity Curve",
) -> None:
    """
    Plot portfolio equity over time.  Optionally overlay a buy-and-hold baseline.

    WHY show baseline?
      Beating the market in absolute return is not enough — if SPY returned
      25% and you returned 22% with twice the volatility, you underperformed.
      The equity curve comparison makes this visible immediately.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    eq = equity_curve["equity"]
    ax.plot(eq.index, eq.values, lw=2, color="steelblue", label="Strategy")

    if baseline_equity is not None:
        # Normalise to same starting value
        scale  = eq.iloc[0] / baseline_equity.iloc[0]
        scaled = baseline_equity * scale
        ax.plot(scaled.index, scaled.values, lw=1.5, ls="--",
                color="gray", alpha=0.8, label="Buy & Hold (SPY)")

    ax.set_title(title)
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, savepath)


def plot_drawdown_underwater(
    equity_curve: pd.DataFrame,
    savepath:     str | Path,
    title:        str = "Drawdown (Underwater Plot)",
) -> None:
    """
    Underwater plot: area below zero showing drawdown depth over time.
    Helps visualise recovery periods and prolonged stress periods.
    """
    eq          = equity_curve["equity"].values
    running_max = np.maximum.accumulate(eq)
    drawdown    = (eq - running_max) / running_max * 100   # in %

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(equity_curve.index, drawdown, 0,
                    color="crimson", alpha=0.6, label="Drawdown")
    ax.axhline(0, color="black", lw=0.8)

    max_dd = drawdown.min()
    ax.axhline(max_dd, color="darkred", ls="--", lw=1,
               label=f"Max DD: {max_dd:.1f}%")

    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, savepath)


def plot_returns_distribution(
    trades_df: pd.DataFrame,
    savepath:  str | Path,
    title:     str = "Trade Returns Distribution",
) -> None:
    """
    Histogram of per-trade returns with win/loss split.
    Shows the return distribution shape — fat tails, skewness, etc.
    """
    if trades_df.empty:
        return

    rets  = trades_df["return_pct"].values * 100
    wins  = rets[rets > 0]
    losses = rets[rets <= 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(rets.min() - 1, rets.max() + 1, 40)

    ax.hist(wins,   bins=bins, color="seagreen",  alpha=0.7, label=f"Wins  n={len(wins)}")
    ax.hist(losses, bins=bins, color="crimson",   alpha=0.7, label=f"Losses n={len(losses)}")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(rets.mean(), color="steelblue", ls="--", lw=1.5,
               label=f"Mean {rets.mean():.2f}%")

    ax.set_title(title)
    ax.set_xlabel("Return per Trade (%)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, savepath)


def plot_per_symbol_pnl(
    trades_df: pd.DataFrame,
    savepath:  str | Path,
    title:     str = "Cumulative P&L per Symbol",
) -> None:
    """
    Bar chart of total dollar P&L per symbol.
    Identifies which symbols drive returns and which are drags.
    """
    if trades_df.empty:
        return

    pnl = (trades_df.groupby("symbol")["dollar_pnl"]
           .sum()
           .sort_values(ascending=True))

    colors = ["seagreen" if v >= 0 else "crimson" for v in pnl.values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(pnl.index, pnl.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)

    for bar, val in zip(bars, pnl.values):
        ax.text(val + (50 if val >= 0 else -50), bar.get_y() + bar.get_height() / 2,
                f"${val:+.0f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Cumulative Dollar P&L ($)")
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, savepath)
