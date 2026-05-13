"""
Buy-and-hold benchmark for backtest comparisons.

WHY does this module exist?
  A strategy that beats random noise but underperforms passive index holding
  is economically worthless — you'd be better off buying SPY and going to the
  beach.  This benchmark answers: "did the ML strategy add value over the
  simplest possible alternative?"

Design:
  equal_weight_buy_hold invests initial_cash equally across all symbols at
  start_date's open and holds to end_date with zero rebalancing.  This is the
  minimal bar any active strategy must clear.

  The equity curve is computed daily using close prices, matching exactly the
  format produced by run_backtest so that plot overlays and metrics comparisons
  are apples-to-apples.

  Cost model:
    Entry cost only (no exit).  A true buy-and-hold never sells, so there is
    no round-trip.  We apply entry cost to be conservative.  If cost_model is
    None we use ZeroCostModel (no friction).
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd
import numpy as np

from src.backtest.costs import TransactionCostModel, ZeroCostModel
from src.backtest.metrics import compute_metrics

log = logging.getLogger(__name__)


def equal_weight_buy_hold(
    prices_dict:  dict[str, pd.DataFrame],
    symbols:      list[str],
    start_date:   pd.Timestamp | str,
    end_date:     pd.Timestamp | str,
    initial_cash: float = 10_000.0,
    cost_model:   TransactionCostModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Simulate an equal-weight buy-and-hold portfolio.

    Parameters
    ----------
    prices_dict  : {symbol: OHLCV DataFrame with UTC DatetimeIndex}
    symbols      : list of symbols to hold (must be keys in prices_dict)
    start_date   : date of initial purchase (at open price)
    end_date     : last date of the equity curve
    initial_cash : starting capital ($)
    cost_model   : entry cost only; defaults to ZeroCostModel

    Returns
    -------
    trades_df    : DataFrame with one row per symbol (the initial buy).
                   Columns match run_backtest output for test compatibility.
    equity_df    : DatetimeIndex DataFrame with 'equity', 'cash', 'n_positions'
    metrics      : dict from compute_metrics(equity_df, trades_df)
    """
    if cost_model is None:
        cost_model = ZeroCostModel()

    start_date = pd.Timestamp(start_date).normalize()
    end_date   = pd.Timestamp(end_date).normalize()

    # ── Normalise price indices ────────────────────────────────────────────────
    price_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = prices_dict.get(sym)
        if df is None:
            log.warning("Symbol %s not found in prices_dict, skipping", sym)
            continue
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        price_data[sym] = df2

    active_symbols = list(price_data.keys())
    if not active_symbols:
        raise ValueError("No valid symbols found in prices_dict")

    # ── Determine union of dates in [start_date, end_date] ────────────────────
    all_dates_set: set[pd.Timestamp] = set()
    for df in price_data.values():
        mask = (df.index >= start_date) & (df.index <= end_date)
        all_dates_set.update(df.index[mask].tolist())
    all_dates = sorted(all_dates_set)

    if not all_dates:
        raise ValueError(
            f"No price data found between {start_date.date()} and {end_date.date()}"
        )

    # Find the first date >= start_date that has data for at least one symbol
    actual_start = all_dates[0]

    # ── Initial buy: equal allocation across symbols at actual_start open ─────
    alloc_per_sym = initial_cash / len(active_symbols)
    positions: dict[str, dict] = {}   # sym -> {shares, entry_price, asset_class}
    remaining_cash = initial_cash

    _CRYPTO_SYMS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"}

    initial_trades = []
    for sym in active_symbols:
        df = price_data[sym]
        if actual_start not in df.index:
            log.warning("No open price for %s on %s, skipping", sym, actual_start.date())
            continue

        raw_open   = float(df.loc[actual_start, "open"])
        ac         = "crypto" if sym in _CRYPTO_SYMS else "stock"
        eff_entry  = cost_model.apply_entry_cost(raw_open, ac)

        if ac == "crypto":
            import math
            shares = alloc_per_sym / eff_entry
        else:
            import math
            shares = math.floor(alloc_per_sym / eff_entry)
            if shares < 1:
                log.warning("Insufficient cash to buy 1 share of %s at %.2f", sym, eff_entry)
                continue

        cost = shares * eff_entry
        remaining_cash -= cost

        positions[sym] = {
            "shares":      shares,
            "entry_price": eff_entry,
            "asset_class": ac,
        }

        initial_trades.append({
            "symbol":       sym,
            "entry_date":   actual_start,
            "exit_date":    end_date,          # notional — held to end
            "entry_price":  eff_entry,
            "exit_price":   float("nan"),      # not yet closed
            "shares":       shares,
            "return_pct":   float("nan"),
            "dollar_pnl":   float("nan"),
            "holding_days": (end_date - actual_start).days,
            "exit_reason":  "buy_and_hold",
            "asset_class":  ac,
            "proba":        1.0,
            "fold":         -1,
        })

    log.info(
        "Buy-and-hold: %d symbols bought on %s, remaining cash=%.2f",
        len(positions), actual_start.date(), remaining_cash,
    )

    # ── Build daily equity curve ───────────────────────────────────────────────
    equity_rows = []
    for date in all_dates:
        pos_value = 0.0
        for sym, pos in positions.items():
            df = price_data[sym]
            if date in df.index:
                pos_value += pos["shares"] * float(df.loc[date, "close"])
            else:
                # Missing data day: carry last known value
                pos_value += pos["shares"] * pos["entry_price"]

        equity_rows.append({
            "date":        date,
            "equity":      remaining_cash + pos_value,
            "cash":        remaining_cash,
            "n_positions": len(positions),
        })

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    equity_df.index = pd.to_datetime(equity_df.index)

    # ── Fill in final exit prices in trades so metrics are complete ────────────
    last_date = all_dates[-1]
    for trade in initial_trades:
        sym = trade["symbol"]
        pos = positions.get(sym)
        if pos is None:
            continue
        df = price_data[sym]
        last_close_date = max(d for d in df.index if d <= last_date)
        last_close = float(df.loc[last_close_date, "close"])
        eff_exit   = cost_model.apply_exit_cost(last_close, pos["asset_class"])
        ret        = (eff_exit / pos["entry_price"]) - 1.0
        pnl        = pos["shares"] * (eff_exit - pos["entry_price"])

        trade["exit_price"]  = eff_exit
        trade["return_pct"]  = ret
        trade["dollar_pnl"]  = pnl

    trades_df = pd.DataFrame(initial_trades)
    if not trades_df.empty:
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])

    metrics = compute_metrics(equity_df, trades_df)

    return trades_df, equity_df, metrics
