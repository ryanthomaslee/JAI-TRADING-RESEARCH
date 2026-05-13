"""
Backtest performance metrics.

WHY these specific metrics and not just "return"?
  Total return tells you nothing without context.  A 30% return over 5 years
  is mediocre; over 6 months it's excellent.  The metrics below answer the
  questions a risk-adjusted investor actually cares about:

  CAGR:        "What is the annualized growth rate?"
  Sharpe:      "How much return per unit of volatility?"  (target: > 1.0)
  Sortino:     "How much return per unit of DOWNSIDE volatility?"
               Sharpe penalises positive volatility (upside moves) equally
               with negative.  Sortino only penalises downside — more
               appropriate for trend-following strategies.
  Calmar:      "How much return per unit of maximum drawdown?"
               Useful for understanding drawdown-relative performance.
  Max DD:      "What's the worst loss from peak to trough?"  (> 50% = untradeable)
  Win rate:    "What fraction of trades are profitable?"
               Without profit_factor context, win rate is misleading.
               A system with 70% wins but 3:1 loss/win ratio destroys capital.
  Profit factor: gross_wins / gross_losses — must be > 1.0 to be profitable.
  Exposure:    "How much time is the portfolio actually in the market?"
               High exposure + low Sharpe = the strategy is basically buy-and-hold.

Annualisation: 252 trading days
  Even though we have crypto (365 days), our 10-symbol universe is majority
  stocks.  252 is the conservative choice and the industry standard.
  We flag small-sample periods explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades_df:    pd.DataFrame,
) -> dict:
    """
    Compute all backtest performance metrics.

    Parameters
    ----------
    equity_curve : DataFrame with DatetimeIndex and 'equity' column
    trades_df    : DataFrame from engine.run_backtest()

    Returns
    -------
    Flat dict of metrics.  All rates/ratios are floats.
    All "pct" values are already in percent (e.g. 15.3 means 15.3%).
    """
    metrics: dict = {}

    if equity_curve.empty or len(equity_curve) < 2:
        return {"error": "insufficient equity curve data"}

    equity = equity_curve["equity"].values
    dates  = pd.to_datetime(equity_curve.index)

    # ── Return metrics ────────────────────────────────────────────────────────
    initial  = equity[0]
    final    = equity[-1]
    total_return = (final / initial) - 1.0

    n_days = (dates[-1] - dates[0]).days
    n_years = max(n_days / 365.0, 1 / 365.0)

    cagr = (final / initial) ** (1.0 / n_years) - 1.0

    metrics["total_return_pct"] = round(total_return * 100, 2)
    metrics["cagr_pct"]         = round(cagr * 100, 2)
    metrics["n_days_oos"]       = int(n_days)
    metrics["small_sample_warning"] = n_years < 2.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    running_max  = np.maximum.accumulate(equity)
    drawdown     = (equity - running_max) / running_max
    max_dd       = float(drawdown.min())
    metrics["max_drawdown_pct"] = round(max_dd * 100, 2)

    # Max drawdown duration: longest continuous period below a previous peak
    underwater = drawdown < 0
    dd_dur = 0
    cur_dur = 0
    for u in underwater:
        cur_dur = cur_dur + 1 if u else 0
        dd_dur  = max(dd_dur, cur_dur)
    metrics["max_dd_duration_days"] = int(dd_dur)

    # Flag untradeable drawdown
    metrics["untradeable_drawdown"] = max_dd < -0.50

    # ── Daily returns for Sharpe / Sortino ────────────────────────────────────
    daily_ret = pd.Series(equity).pct_change().dropna().values

    ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)

    if daily_ret.std() > 1e-10:
        sharpe = (daily_ret.mean() / daily_ret.std()) * ann_factor
    else:
        sharpe = 0.0

    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 1e-10:
        sortino = (daily_ret.mean() / downside.std()) * ann_factor
    else:
        sortino = 0.0 if daily_ret.mean() <= 0 else float("inf")

    calmar = cagr / abs(max_dd) if max_dd < -1e-6 else 0.0

    metrics["sharpe"]  = round(float(sharpe), 3)
    metrics["sortino"] = round(float(sortino), 3)
    metrics["calmar"]  = round(float(calmar), 3)

    # ── Trade statistics ──────────────────────────────────────────────────────
    if trades_df.empty:
        metrics.update({
            "num_trades": 0, "win_rate_pct": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "profit_factor": 0.0, "avg_holding_days": 0.0,
            "exposure_pct": 0.0,
        })
        return metrics

    rets   = trades_df["return_pct"].values
    wins   = rets[rets > 0]
    losses = rets[rets <= 0]

    win_rate = len(wins) / len(rets) if len(rets) > 0 else 0.0

    gross_wins   = wins.sum()  if len(wins) > 0   else 0.0
    gross_losses = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_wins / gross_losses

    metrics["num_trades"]      = int(len(rets))
    metrics["win_rate_pct"]    = round(win_rate * 100, 1)
    metrics["avg_win_pct"]     = round(float(wins.mean() * 100) if len(wins) > 0 else 0.0, 2)
    metrics["avg_loss_pct"]    = round(float(losses.mean() * 100) if len(losses) > 0 else 0.0, 2)
    metrics["profit_factor"]   = round(float(profit_factor), 3)
    metrics["avg_holding_days"]= round(float(trades_df["holding_days"].mean()), 1)

    # ── Exposure ──────────────────────────────────────────────────────────────
    # Fraction of simulation days with at least one open position
    if "n_positions" in equity_curve.columns:
        in_market = (equity_curve["n_positions"] > 0).mean()
        metrics["exposure_pct"] = round(float(in_market) * 100, 1)
    else:
        metrics["exposure_pct"] = float("nan")

    return metrics
