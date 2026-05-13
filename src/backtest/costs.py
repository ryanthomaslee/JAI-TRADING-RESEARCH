"""
Transaction cost model.

WHY model costs at all for a backtesting system?
  Zero-cost backtests are lying to you.  The same signal that produces a
  20% CAGR at zero cost may produce 8% after realistic costs — or lose money.
  The purpose of the cost model is to find the *cost break-even threshold*:
  the signal strength below which trading destroys value.

WHY these specific cost numbers?
  Stocks (5 bps commission + 5 bps slippage = 10 bps round trip):
    Interactive Brokers Pro charges ~$0.005/share; a $50 stock = 1 bp.
    We use 5 bps to cover both commission AND bid-ask spread.
    Slippage on liquid large-caps (SPY, AAPL) is typically 1-3 bps;
    we use 5 bps as a conservative mid-estimate.

  Crypto (10 bps commission + 5 bps slippage = 15 bps round trip):
    Binance charges 7.5 bps (taker) with BNB discount; Coinbase ~15 bps.
    We use 10 bps as a realistic blended rate.
    Crypto spreads are wider than stocks, hence higher slippage.

WHY model entry and exit costs separately?
  Entry cost raises your effective buy price (you pay more than the mark).
  Exit cost lowers your effective sell price (you receive less than the mark).
  Modelling them separately allows the engine to correctly compute P&L:
    realized_return = (exit_price_effective / entry_price_effective) - 1
  If we just applied one round-trip adjustment at the end, we'd get the
  right total but wrong per-trade P&L shape.
"""

from __future__ import annotations


class TransactionCostModel:
    """Realistic transaction cost model with separate entry/exit costs."""

    # Fee rates by asset class (one-way)
    ASSET_CLASS_FEES: dict[str, float] = {
        "stock":  0.0005,   # 5 bps
        "crypto": 0.0010,   # 10 bps
    }
    SLIPPAGE: float = 0.0005  # 5 bps, one-way

    def apply_entry_cost(self, price: float, asset_class: str) -> float:
        """
        Return effective entry price after fees and slippage.
        Price is HIGHER than market (we pay more to buy).
        """
        fee      = self.ASSET_CLASS_FEES.get(asset_class, self.ASSET_CLASS_FEES["stock"])
        one_way  = fee + self.SLIPPAGE
        return price * (1.0 + one_way)

    def apply_exit_cost(self, price: float, asset_class: str) -> float:
        """
        Return effective exit price after fees and slippage.
        Price is LOWER than market (we receive less when selling).
        """
        fee      = self.ASSET_CLASS_FEES.get(asset_class, self.ASSET_CLASS_FEES["stock"])
        one_way  = fee + self.SLIPPAGE
        return price * (1.0 - one_way)

    def total_round_trip_pct(self, asset_class: str) -> float:
        """
        Total round-trip cost as a fraction of trade value.
        Useful for diagnostics and break-even analysis.

        Example: stock = 2 × (0.0005 + 0.0005) = 0.20% round trip.
        A trade must return > 0.20% just to break even on costs.
        """
        fee = self.ASSET_CLASS_FEES.get(asset_class, self.ASSET_CLASS_FEES["stock"])
        return 2.0 * (fee + self.SLIPPAGE)


class ZeroCostModel(TransactionCostModel):
    """
    Zero-cost model for diagnostics.

    Returns price unchanged — lets us isolate the pure signal quality
    from cost drag.  If a strategy is only profitable at zero cost,
    it's not tradeable and we say so explicitly.
    """

    def apply_entry_cost(self, price: float, asset_class: str) -> float:
        return price

    def apply_exit_cost(self, price: float, asset_class: str) -> float:
        return price

    def total_round_trip_pct(self, asset_class: str) -> float:
        return 0.0
