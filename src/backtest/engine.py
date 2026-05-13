"""
Walk-forward backtest engine.

THE MOST IMPORTANT RULE IN THIS FILE:
  Entries execute at t+1 OPEN, never at t's close.

  WHY is this the most common backtest leakage bug?
    Suppose the model emits a signal at the close of day t.  In a real system
    you receive the signal after the market closes and submit a market-open
    order for t+1.  The order fills at t+1's opening price.

    Backtests that fill at t's close are using a price that wasn't available
    when the signal was generated — the close is only known AFTER the close.
    This overstates returns by capturing the late-session momentum that already
    happened, sometimes by 0.3-0.8% per trade on volatile stocks.

    We enforce this by: on day t, record the signal but DON'T trade.
    On day t+1, look up t+1's open price and execute.

Exit priority when both barriers touched same day:
  If today's HIGH >= target AND today's LOW <= stop, which wins?
  We assume STOP wins (worst case). Reason:
    - Intra-day order is unobservable on daily bars
    - The conservative assumption penalises the strategy honestly
    - A real system using stop-market orders would often get the stop
      triggered first on a gap-down day

Structure:
  1. Build a time-ordered list of all dates across all symbols
  2. For each date:
     a) Mark-to-market open positions with today's close
     b) CHECK EXITS: iterate open positions, check high/low/expiry
     c) CHECK ENTRIES: for any pending signal from yesterday, execute at today's open
     d) QUEUE SIGNALS: record today's predictions above threshold as pending
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta

import pandas as pd
import numpy as np

from src.backtest.costs import TransactionCostModel, ZeroCostModel
from src.backtest.portfolio import Portfolio

log = logging.getLogger(__name__)

# Symbols that are crypto (need fractional shares + higher costs)
_CRYPTO_SYMBOLS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"}


def _asset_class(symbol: str) -> str:
    return "crypto" if symbol in _CRYPTO_SYMBOLS else "stock"


def run_backtest(
    predictions_df: pd.DataFrame,
    prices_dict:    dict[str, pd.DataFrame],
    cost_model:     TransactionCostModel,
    portfolio:      Portfolio,
    threshold:      float = 0.55,
    upper_pct:      float = 0.05,
    lower_pct:      float = 0.03,
    max_days:       int   = 20,
    cooldown_days:  int   = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate trading on OOF predictions.

    Parameters
    ----------
    predictions_df : combined predictions with columns [symbol, proba_cal, fold]
                     and DatetimeIndex of signal dates (day t)
    prices_dict    : {symbol: OHLCV DataFrame} — raw prices for execution
    cost_model     : TransactionCostModel (realistic or zero)
    portfolio      : Portfolio instance (fresh for each backtest run)
    threshold      : minimum proba_cal to generate a signal
    upper_pct      : profit target fraction
    lower_pct      : stop-loss fraction
    max_days       : maximum holding period (calendar days × 1.5 conversion used
                     in Portfolio.enter)
    cooldown_days  : after a position in symbol X exits on date D, block new
                     entries in X until D + cooldown_days have elapsed.
                     Cooldown is per-symbol — blocking JPM does not affect AAPL.
                     0 = no cooldown (default, matches Phase 4 behaviour).

    Returns
    -------
    trades_df     : DataFrame of all closed trades
    equity_curve  : DataFrame of daily equity values
    """
    # ── Normalize index to date-only for alignment ────────────────────────────
    # Predictions have UTC timestamps (e.g. 2021-11-17 05:00:00+00:00).
    # Prices also have UTC timestamps.  We normalize both to date objects
    # for lookup, avoiding timezone-comparison issues.
    pred = predictions_df.copy()
    pred.index = pd.to_datetime(pred.index).normalize()

    # Build price lookup: {symbol: {date -> row}}
    price_data: dict[str, pd.DataFrame] = {}
    for sym, df in prices_dict.items():
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        price_data[sym] = df2

    # ── Build sorted list of all dates that appear in predictions ────────────
    # WHY from predictions only (not full price history)?
    #   The OOF predictions only exist for test-fold dates.  Iterating over
    #   full price history would include training-fold dates where we have no
    #   predictions — leading to phantom "no signal" days that inflate the
    #   denominator of exposure metrics.
    all_dates = sorted(pred.index.unique())
    log.info("Backtest: %d prediction dates, %d symbols, threshold=%.2f",
             len(all_dates), pred["symbol"].nunique(), threshold)

    # pending_entries[date] = list of (symbol, proba, fold) to execute NEXT day
    pending_entries: dict[pd.Timestamp, list[tuple]] = defaultdict(list)

    # cooldown tracking: last date a position in each symbol was closed
    # (keyed by symbol; only populated when cooldown_days > 0)
    last_exit_date: dict[str, pd.Timestamp] = {}

    # Queue signals from the day BEFORE the first prediction date
    # so the first day's signals get executed on the second day
    for dt in all_dates:
        day_preds = pred.loc[pred.index == dt]
        for _, row in day_preds.iterrows():
            sym    = row["symbol"]
            proba  = float(row["proba_cal"])
            fold   = int(row["fold"])
            if proba >= threshold:
                # Schedule entry for NEXT available date
                pending_entries[dt].append((sym, proba, fold))

    # ── Main simulation loop ──────────────────────────────────────────────────
    for i, date in enumerate(all_dates):

        # ── a) Get today's prices ─────────────────────────────────────────────
        close_prices:  dict[str, float] = {}
        open_prices:   dict[str, float] = {}
        high_prices:   dict[str, float] = {}
        low_prices:    dict[str, float] = {}

        all_active_syms = set(portfolio.positions.keys()) | {
            sym for sym, _, _ in pending_entries.get(date, [])
        }

        for sym in all_active_syms | set(predictions_df["symbol"].unique()):
            sym_prices = price_data.get(sym)
            if sym_prices is None:
                continue
            if date in sym_prices.index:
                row = sym_prices.loc[date]
                close_prices[sym]  = float(row["close"])
                open_prices[sym]   = float(row["open"])
                high_prices[sym]   = float(row["high"])
                low_prices[sym]    = float(row["low"])

        # ── b) Mark-to-market with today's close ──────────────────────────────
        portfolio.mark_to_market(close_prices, date)

        # ── c) Execute entries from YESTERDAY'S signals ───────────────────────
        # These are the signals queued from date[i-1]; we use today's OPEN price.
        # THIS IS THE KEY ANTI-LEAKAGE STEP.
        if i > 0:
            prev_date = all_dates[i - 1]
            for sym, proba, fold in pending_entries.get(prev_date, []):
                if not portfolio.can_enter(sym):
                    continue
                # Cooldown gate: refuse re-entry if last exit was too recent
                if cooldown_days > 0 and sym in last_exit_date:
                    earliest_reentry = last_exit_date[sym] + timedelta(days=cooldown_days)
                    if date < earliest_reentry:
                        log.debug(
                            "COOLDOWN %s: last exit %s, earliest reentry %s, skipping %s",
                            sym, last_exit_date[sym].date(), earliest_reentry.date(), date.date(),
                        )
                        continue
                entry_price = open_prices.get(sym)
                if entry_price is None or entry_price <= 0:
                    log.debug("No open price for %s on %s, skipping entry", sym, date.date())
                    continue

                ac = _asset_class(sym)
                portfolio.enter(
                    symbol=sym,
                    date=date,
                    price=entry_price,
                    asset_class=ac,
                    cost_model=cost_model,
                    proba=proba,
                    fold=fold,
                    upper_pct=upper_pct,
                    lower_pct=lower_pct,
                    max_days=max_days,
                )

        # ── d) Check exits for open positions ─────────────────────────────────
        for sym in list(portfolio.positions.keys()):
            pos    = portfolio.positions[sym]
            high   = high_prices.get(sym)
            low    = low_prices.get(sym)
            close  = close_prices.get(sym)
            ac     = _asset_class(sym)

            if high is None or low is None:
                continue  # missing data day — hold

            hit_target = (high >= pos.target)
            hit_stop   = (low  <= pos.stop)
            hit_expiry = (date >= pos.expiry_date)

            if hit_stop and hit_target:
                # Both barriers on same day: STOP wins (worst-case assumption)
                # WHY: on a volatile day that opens near the target and gaps
                # down to the stop, a stop-market order would typically fill
                # at the stop level before we can exit at the target.
                portfolio.exit(sym, date, pos.stop, ac, cost_model, "stop")
                last_exit_date[sym] = date

            elif hit_target:
                portfolio.exit(sym, date, pos.target, ac, cost_model, "target")
                last_exit_date[sym] = date

            elif hit_stop:
                portfolio.exit(sym, date, pos.stop, ac, cost_model, "stop")
                last_exit_date[sym] = date

            elif hit_expiry:
                exit_price = close if close else pos.entry_price
                portfolio.exit(sym, date, exit_price, ac, cost_model, "expiry")
                last_exit_date[sym] = date

    # ── End of simulation: force-close any remaining positions ────────────────
    # WHY force-close?
    #   OOF predictions end at a fixed date.  Leaving positions open would
    #   ignore their unrealised P&L and make the equity curve look flat at the end.
    #   We close at the last available close price.
    if all_dates:
        last_date = all_dates[-1]
        portfolio.close_all(close_prices, last_date, cost_model)

    # ── Build output DataFrames ────────────────────────────────────────────────
    trades_df = pd.DataFrame(portfolio.closed_trades)
    if not trades_df.empty:
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])

    equity_df = pd.DataFrame(portfolio.equity_curve)
    if not equity_df.empty:
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")

    log.info("Backtest complete: %d trades, final equity=%.2f",
             len(trades_df), equity_df["equity"].iloc[-1] if not equity_df.empty else 0)

    return trades_df, equity_df
