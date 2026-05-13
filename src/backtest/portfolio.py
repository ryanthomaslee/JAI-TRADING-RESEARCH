"""
Portfolio: cash accounting, position management, equity curve.

Design decisions
────────────────
max_position_pct=0.20:
  No single position can use more than 20% of current equity.
  At 5 max positions this guarantees the portfolio is never 100% concentrated
  in one name, even if confidence_weighted sizing allocates the full 4%.

max_open_positions=5:
  Limits concentration risk.  With 10 symbols and a 1-4 week hold, we
  expect 3-5 signals at any given time.  5 is a natural ceiling.

Sizing modes:
  fixed_fractional: 2% of equity, every trade.  Simple, well-understood.
    Comes from the Kelly criterion literature — 2% is a conservative Kelly
    fraction that limits ruin probability.

  confidence_weighted: linearly scale from 1% (threshold) to 4% (1.0 score).
    Signal at 0.60 → 1.0% of equity.  Signal at 0.80 → 2.6%.  Signal at 1.0 → 4%.
    This is not pure Kelly but a practical approximation: bet more when
    the model is more confident.

  equal_weight: 1/max_positions of equity each.  Same as a 20% position
    in an equally-weighted portfolio.  Useful as a naive baseline.

Whole shares vs fractional:
  Stocks: you can't buy 2.7 shares.  We floor to whole shares.
  WHY: fractional shares aren't universally available, and using them
  in a backtest while trading with a broker that doesn't offer them
  creates a systematic bias (you always execute exactly).
  Crypto: exchanges allow arbitrary precision, so we keep full fractional
  amount (computed as $ value / price).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.backtest.costs import TransactionCostModel

log = logging.getLogger(__name__)

SizingMode = Literal["fixed_fractional", "confidence_weighted", "equal_weight"]


@dataclass
class Position:
    symbol:      str
    entry_date:  pd.Timestamp
    entry_price: float       # effective entry (after costs)
    shares:      float       # whole number for stocks, fractional for crypto
    target:      float       # price level triggering +1 exit
    stop:        float       # price level triggering -1 exit
    expiry_date: pd.Timestamp
    asset_class: str         # "stock" | "crypto"
    proba:       float = 0.0 # signal probability at entry (for attribution)
    fold:        int   = -1  # which CV fold generated this signal


class Portfolio:
    """
    Tracks cash, open positions, and a daily equity curve.

    NOT thread-safe.  Designed for single-pass chronological simulation.
    """

    def __init__(
        self,
        initial_cash:        float = 10_000.0,
        max_position_pct:    float = 0.20,
        max_open_positions:  int   = 5,
        sizing_mode:         SizingMode = "fixed_fractional",
        entry_threshold:     float = 0.55,
    ) -> None:
        self.cash             = initial_cash
        self.initial_cash     = initial_cash
        self.max_position_pct = max_position_pct
        self.max_open_positions = max_open_positions
        self.sizing_mode      = sizing_mode
        self.entry_threshold  = entry_threshold

        self.positions: dict[str, Position] = {}   # symbol -> Position
        self.equity_curve: list[dict] = []          # [{date, equity, cash, n_positions}]
        self.closed_trades: list[dict] = []         # completed trade records

    # ── Equity ────────────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        """Cash + mark-to-market value of all open positions."""
        pos_value = sum(
            p.shares * p.entry_price for p in self.positions.values()
        )
        return self.cash + pos_value

    def mark_to_market(self, price_lookup: dict[str, float], date: pd.Timestamp) -> None:
        """
        Record current equity to the curve using today's close prices.
        Called once per trading day before entering/exiting positions.

        WHY close prices for mark-to-market?
          Close is the standard convention for daily P&L.  Open would be
          stale (yesterday's close for stocks), and using the entry price
          would hide unrealised gains/losses entirely.
        """
        pos_value = 0.0
        for sym, pos in self.positions.items():
            price = price_lookup.get(sym)
            if price is not None:
                pos_value += pos.shares * price
            else:
                # Price missing (e.g. holiday) — carry at last known value
                pos_value += pos.shares * pos.entry_price

        equity = self.cash + pos_value
        self.equity_curve.append({
            "date":        date,
            "equity":      equity,
            "cash":        self.cash,
            "n_positions": len(self.positions),
        })

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _position_dollar_size(self, proba: float) -> float:
        """
        Compute dollar amount to allocate to a new position.
        Returns 0.0 if insufficient cash or max positions reached.
        """
        equity = self.equity

        if self.sizing_mode == "fixed_fractional":
            frac = 0.02   # 2% of equity

        elif self.sizing_mode == "confidence_weighted":
            # Linear interpolation: threshold → 1%, 1.0 → 4%
            t = self.entry_threshold
            frac = 0.01 + (proba - t) / (1.0 - t) * 0.03 if proba < 1.0 else 0.04
            frac = max(0.01, min(0.04, frac))

        elif self.sizing_mode == "equal_weight":
            frac = 1.0 / self.max_open_positions  # e.g. 20% each

        else:
            raise ValueError(f"Unknown sizing_mode: {self.sizing_mode!r}")

        # Cap at max_position_pct
        frac = min(frac, self.max_position_pct)
        return equity * frac

    # ── Entry / exit guards ───────────────────────────────────────────────────

    def can_enter(self, symbol: str) -> bool:
        """
        Return True if we can open a new position in `symbol`.
        Rejects if: symbol already held, max positions reached, insufficient cash.
        """
        if symbol in self.positions:
            return False
        if len(self.positions) >= self.max_open_positions:
            return False
        # Minimum position check: can we afford at least 1 unit?
        if self.cash <= 0:
            return False
        return True

    # ── Entry ─────────────────────────────────────────────────────────────────

    def enter(
        self,
        symbol:      str,
        date:        pd.Timestamp,
        price:       float,         # t+1 open price (BEFORE cost adjustment)
        asset_class: str,
        cost_model:  TransactionCostModel,
        proba:       float,
        fold:        int,
        upper_pct:   float = 0.05,
        lower_pct:   float = 0.03,
        max_days:    int   = 20,
    ) -> bool:
        """
        Open a position at `price` (t+1 open).

        Returns True if entry succeeded, False if rejected (insufficient cash
        after sizing, or price is zero/NaN).
        """
        if not price or not math.isfinite(price) or price <= 0:
            return False

        effective_entry = cost_model.apply_entry_cost(price, asset_class)
        dollar_size     = self._position_dollar_size(proba)

        if dollar_size <= 0:
            return False

        # Cap at available cash (never go negative)
        dollar_size = min(dollar_size, self.cash)
        if dollar_size < effective_entry:
            # Can't even buy 1 unit
            return False

        # Compute shares
        if asset_class == "crypto":
            shares = dollar_size / effective_entry   # fractional OK
        else:
            shares = math.floor(dollar_size / effective_entry)  # whole shares only
            if shares < 1:
                return False

        cost = shares * effective_entry
        if cost > self.cash:
            # Floating-point edge: reduce by 1 share for stocks
            if asset_class != "crypto":
                shares -= 1
                cost = shares * effective_entry
            if shares <= 0 or cost > self.cash:
                return False

        self.cash -= cost
        assert self.cash >= -1e-6, f"Cash went negative: {self.cash}"
        self.cash = max(0.0, self.cash)  # clip floating-point dust

        # Set target and stop levels from the effective entry price
        # WHY from effective entry, not raw price?
        #   Costs raise the break-even.  The 5% target measured from raw
        #   price is actually only 4.9% from effective entry.  Using raw
        #   price as the reference is slightly generous to the backtest.
        #   We use the raw price as the barrier reference (matches label
        #   definition) but track P&L from effective entry.
        target = price * (1.0 + upper_pct)
        stop   = price * (1.0 - lower_pct)

        from datetime import timedelta
        # Expiry: max_days trading days from entry — approximate with calendar days × 1.4
        # For an exact trading-day calendar we'd need a market calendar library.
        # 20 trading days ≈ 28 calendar days; we use 30 for a small buffer.
        expiry = date + timedelta(days=int(max_days * 1.5))

        self.positions[symbol] = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=effective_entry,
            shares=shares,
            target=target,
            stop=stop,
            expiry_date=expiry,
            asset_class=asset_class,
            proba=proba,
            fold=fold,
        )
        log.debug("ENTER %s @ %.4f ×%.4f shares (cost %.2f) date=%s",
                  symbol, effective_entry, shares, cost, date.date())
        return True

    # ── Exit ──────────────────────────────────────────────────────────────────

    def exit(
        self,
        symbol:      str,
        date:        pd.Timestamp,
        price:       float,         # exit price BEFORE cost adjustment
        asset_class: str,
        cost_model:  TransactionCostModel,
        reason:      str,           # "target" | "stop" | "expiry" | "forced"
    ) -> dict | None:
        """
        Close an open position.  Returns the closed trade record.
        """
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        if not price or not math.isfinite(price) or price <= 0:
            # Fallback: close at entry price (no gain/loss)
            price = pos.entry_price

        effective_exit = cost_model.apply_exit_cost(price, asset_class)
        proceeds       = pos.shares * effective_exit
        self.cash      += proceeds

        holding_days   = (date - pos.entry_date).days
        ret            = (effective_exit / pos.entry_price) - 1.0

        trade = {
            "symbol":        symbol,
            "entry_date":    pos.entry_date,
            "exit_date":     date,
            "entry_price":   pos.entry_price,
            "exit_price":    effective_exit,
            "shares":        pos.shares,
            "return_pct":    ret,
            "dollar_pnl":    proceeds - (pos.shares * pos.entry_price),
            "holding_days":  holding_days,
            "exit_reason":   reason,
            "asset_class":   asset_class,
            "proba":         pos.proba,
            "fold":          pos.fold,
        }
        self.closed_trades.append(trade)
        log.debug("EXIT %s @ %.4f reason=%s ret=%.2f%%",
                  symbol, effective_exit, reason, ret * 100)
        return trade

    def close_all(
        self,
        price_lookup:dict[str, float],
        date: pd.Timestamp,
        cost_model: TransactionCostModel,
    ) -> None:
        """Force-close all open positions at given prices (end-of-backtest cleanup)."""
        for symbol in list(self.positions.keys()):
            pos   = self.positions[symbol]
            price = price_lookup.get(symbol, pos.entry_price)
            self.exit(symbol, date, price, pos.asset_class, cost_model, "forced")
