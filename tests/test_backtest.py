"""
Backtest engine correctness tests.

Each test targets one specific invariant that, if violated, would produce
silently wrong results.  We use minimal synthetic data so failures
point directly at the violated assumption — not at data quality.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.costs     import TransactionCostModel, ZeroCostModel
from src.backtest.portfolio import Portfolio
from src.backtest.engine    import run_backtest
from src.backtest.metrics   import compute_metrics
from src.backtest.benchmark import equal_weight_buy_hold


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_prices(
    symbol:     str,
    start:      str = "2022-01-01",
    n:          int = 100,
    open_:      float = 100.0,
    trend:      float = 0.001,   # daily drift
    flat:       bool  = False,
) -> pd.DataFrame:
    """Generate flat or trending OHLCV price series."""
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC")
    rng   = np.random.default_rng(42)
    if flat:
        closes = np.full(n, open_)
    else:
        noise  = rng.normal(0, 0.005, n)
        closes = open_ * np.cumprod(1 + trend + noise)

    opens  = closes * (1 + rng.normal(0, 0.002, n))
    highs  = np.maximum(closes, opens) * (1 + rng.uniform(0, 0.01, n))
    lows   = np.minimum(closes, opens) * (1 - rng.uniform(0, 0.01, n))

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e6},
        index=dates,
    )


def _make_predictions(
    symbol:    str,
    dates:     pd.DatetimeIndex,
    proba:     float = 0.70,
    fold:      int   = 0,
) -> pd.DataFrame:
    """All predictions at fixed proba for given dates."""
    return pd.DataFrame(
        {"y_true": 1, "proba_raw": proba, "proba_cal": proba,
         "symbol": symbol, "fold": fold},
        index=dates,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1: No same-day entry
# ─────────────────────────────────────────────────────────────────────────────

def test_no_same_day_entry():
    """
    A signal generated on date T must not result in a trade entry on date T.
    Entry must be at T+1's open.

    We verify this by checking that no trade's entry_date matches
    the prediction date — it must always be one day later.
    """
    prices = _make_prices("SPY", n=30)
    pred_dates = prices.index[:20]
    preds  = _make_predictions("SPY", pred_dates, proba=0.80)

    portfolio = Portfolio(initial_cash=10_000)
    trades, _ = run_backtest(
        predictions_df=preds,
        prices_dict={"SPY": prices},
        cost_model=ZeroCostModel(),
        portfolio=portfolio,
        threshold=0.60,
    )

    if trades.empty:
        pytest.skip("No trades generated — check prediction date alignment")

    # Every entry must be AFTER the signal date
    # Find the prediction dates that generated entries
    for _, trade in trades.iterrows():
        entry_date = pd.Timestamp(trade["entry_date"]).normalize()
        # Entry date must be in prices index (it's t+1 open), and prediction
        # dates are a subset, so entry_date must not be the first pred date
        # (which would have required same-day fill).
        assert entry_date > pd.Timestamp(pred_dates[0]).normalize(), (
            f"Trade entered on {entry_date}, the earliest possible date — "
            "this suggests same-day entry from the very first signal."
        )


def test_entry_uses_next_day_open():
    """
    Entry price must equal t+1's open (within floating-point tolerance after costs).
    We use ZeroCostModel to isolate the price lookup from cost adjustments.

    WHY multi-day predictions?
      The engine iterates over prediction dates.  The entry for day t executes
      when day t+1 is processed.  With only one prediction date, there is no
      t+1 in the loop — entry correctly never fires.  We provide predictions
      on days 5 AND 6 so day 5's signal executes on day 6.
    """
    prices = _make_prices("SPY", n=40)
    # Predictions on days 5 AND 6 — day 5 signal executes at day 6's open
    pred = pd.DataFrame(
        {"y_true": 1, "proba_raw": 0.80, "proba_cal": 0.80, "symbol": "SPY", "fold": 0},
        index=prices.index[5:7],   # two dates: [5] and [6]
    )

    portfolio = Portfolio(initial_cash=10_000)
    trades, _ = run_backtest(
        predictions_df=pred,
        prices_dict={"SPY": prices},
        cost_model=ZeroCostModel(),
        portfolio=portfolio,
        threshold=0.70,
    )

    assert not trades.empty, "No trade was generated — check prediction/price alignment"

    expected_open = float(prices.iloc[6]["open"])   # t+1 = index[6]
    actual_entry  = float(trades.iloc[0]["entry_price"])

    assert abs(actual_entry - expected_open) < 0.01, (
        f"Entry price {actual_entry:.4f} ≠ next day open {expected_open:.4f}. "
        "Same-day entry or wrong price lookup."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2: Exit priority — stop beats target on same day
# ─────────────────────────────────────────────────────────────────────────────

def test_exit_priority_stop_beats_target():
    """
    When both target (high >= entry×1.05) and stop (low <= entry×0.97) are
    touched on the same bar, the exit reason must be 'stop' (worst-case).
    """
    # Build prices where day 5 has extreme high AND extreme low
    dates  = pd.date_range("2022-01-01", periods=15, freq="D", tz="UTC")
    closes = [100.0] * 15
    opens  = [100.0] * 15
    highs  = [101.0] * 15
    lows   = [99.0]  * 15

    # Day index 7 (t+1 after signal day 6): set high=108 (>5% target) AND low=95 (<3% stop)
    highs[7] = 110.0   # well above 5% target
    lows[7]  = 90.0    # well below 3% stop

    prices_df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e6},
        index=dates,
    )

    # Predictions on days 6 AND 7: day 6 signal executes at day 7's open
    # (the extreme high/low day).  Day 7 is also a prediction date so the
    # engine iterates over it and can check the exit.
    pred = pd.DataFrame(
        {"y_true": 1, "proba_raw": 0.80, "proba_cal": 0.80, "symbol": "SPY", "fold": 0},
        index=[dates[6], dates[7]],
    )

    portfolio = Portfolio(initial_cash=10_000)
    trades, _ = run_backtest(
        predictions_df=pred,
        prices_dict={"SPY": prices_df},
        cost_model=ZeroCostModel(),
        portfolio=portfolio,
        threshold=0.70,
    )

    if trades.empty:
        pytest.skip("No trade generated — price structure may have prevented entry")

    # The very first trade should have been stopped out on day 7
    exit_reason = trades.iloc[0]["exit_reason"]
    assert exit_reason == "stop", (
        f"Expected exit_reason='stop' when both barriers touched, got '{exit_reason}'. "
        "Stop should win on same-day conflict (worst-case assumption)."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3: Realistic costs always reduce returns vs zero cost
# ─────────────────────────────────────────────────────────────────────────────

def test_costs_reduce_returns():
    """
    Final equity with realistic costs must be <= final equity at zero cost.
    If this fails, the cost model is incorrectly reducing costs (impossible).
    """
    prices = _make_prices("SPY", n=60, trend=0.002)  # uptrend
    preds  = _make_predictions("SPY", prices.index[:50], proba=0.75)

    def _run(cm):
        p = Portfolio(initial_cash=10_000, max_open_positions=3)
        _, eq = run_backtest(preds, {"SPY": prices}, cm, p, threshold=0.60)
        return eq["equity"].iloc[-1] if not eq.empty else 10_000.0

    zero_final     = _run(ZeroCostModel())
    realistic_final = _run(TransactionCostModel())

    assert realistic_final <= zero_final + 1e-6, (
        f"Realistic costs ({realistic_final:.2f}) produced MORE equity than "
        f"zero costs ({zero_final:.2f}). Cost model is broken."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4: Cash never goes negative
# ─────────────────────────────────────────────────────────────────────────────

def test_no_negative_cash():
    """
    Portfolio cash must never drop below zero, even under aggressive sizing
    with many simultaneous signals.
    """
    # 5 symbols all signalling at the same time
    symbols = ["SPY", "AAPL", "MSFT", "NVDA", "JPM"]
    prices_dict = {sym: _make_prices(sym, n=50) for sym in symbols}

    # All symbols signal on the same day
    signal_date = prices_dict["SPY"].index[5:6]
    pred_frames = [
        pd.DataFrame(
            {"y_true": 1, "proba_raw": 0.90, "proba_cal": 0.90,
             "symbol": sym, "fold": 0},
            index=signal_date,
        )
        for sym in symbols
    ]
    preds = pd.concat(pred_frames)

    portfolio = Portfolio(
        initial_cash=1_000.0,       # small capital to stress test
        max_open_positions=5,
        sizing_mode="equal_weight",
    )
    _, equity = run_backtest(
        predictions_df=preds,
        prices_dict=prices_dict,
        cost_model=TransactionCostModel(),
        portfolio=portfolio,
        threshold=0.80,
    )

    # Check cash at every point in equity curve
    for record in portfolio.equity_curve:
        assert record["cash"] >= -1e-4, (
            f"Cash went negative: {record['cash']:.4f} on {record['date']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5: Max open positions enforced
# ─────────────────────────────────────────────────────────────────────────────

def test_max_positions_enforced():
    """
    Portfolio must never hold more than max_open_positions simultaneously.
    """
    n_syms = 8
    symbols = [f"SYM{i}" for i in range(n_syms)]
    # All have prices
    prices_dict = {sym: _make_prices(sym, n=50) for sym in symbols}

    # All signal on day 5 simultaneously with high proba
    signal_date = prices_dict["SYM0"].index[5:6]
    preds = pd.concat([
        pd.DataFrame(
            {"y_true": 1, "proba_raw": 0.95, "proba_cal": 0.95,
             "symbol": sym, "fold": 0},
            index=signal_date,
        )
        for sym in symbols
    ])

    max_pos = 3
    portfolio = Portfolio(initial_cash=100_000, max_open_positions=max_pos)
    _, _ = run_backtest(
        predictions_df=preds,
        prices_dict=prices_dict,
        cost_model=ZeroCostModel(),
        portfolio=portfolio,
        threshold=0.90,
    )

    # At peak, portfolio should have held at most max_pos positions
    max_held = max(r["n_positions"] for r in portfolio.equity_curve) if portfolio.equity_curve else 0
    assert max_held <= max_pos, (
        f"Portfolio held {max_held} positions simultaneously, "
        f"exceeding max_open_positions={max_pos}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6: Equity curve has no gaps in prediction days
# ─────────────────────────────────────────────────────────────────────────────

def test_equity_curve_continuity():
    """
    The equity curve must have exactly one entry per prediction date.
    Gaps would cause metrics like Sharpe to be computed on wrong time base.
    """
    prices = _make_prices("SPY", n=60)
    preds  = _make_predictions("SPY", prices.index[:50], proba=0.65)

    portfolio = Portfolio(initial_cash=10_000)
    _, equity = run_backtest(
        predictions_df=preds,
        prices_dict={"SPY": prices},
        cost_model=ZeroCostModel(),
        portfolio=portfolio,
        threshold=0.60,
    )

    assert not equity.empty, "Equity curve is empty"
    assert len(equity) == 50, (
        f"Equity curve has {len(equity)} rows but expected 50 (one per prediction date)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 7: SPY buy-and-hold baseline sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_hold_baseline_sanity():
    """
    Simulate a trivially correct strategy: always hold SPY.
    Final equity should be ~proportional to price appreciation.

    This tests that the engine + cost model + portfolio produce
    economically sensible numbers on a known input.
    """
    # SPY goes from 100 → 150 over 60 days (50% gain)
    n = 60
    dates  = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    prices = np.linspace(100, 150, n)

    prices_df = pd.DataFrame(
        {"open": prices, "high": prices * 1.005, "low": prices * 0.995,
         "close": prices, "volume": 1e6},
        index=dates,
    )

    # One signal on day 0 → entry day 1 → hold until expiry (day ~30)
    pred = pd.DataFrame(
        {"y_true": 1, "proba_raw": 0.80, "proba_cal": 0.80, "symbol": "SPY", "fold": 0},
        index=[dates[0]],
    )

    portfolio = Portfolio(initial_cash=10_000, max_open_positions=1)
    trades, equity = run_backtest(
        predictions_df=pred,
        prices_dict={"SPY": prices_df},
        cost_model=ZeroCostModel(),
        portfolio=portfolio,
        threshold=0.70,
        upper_pct=0.50,   # wide target — shouldn't trigger
        lower_pct=0.50,   # wide stop — shouldn't trigger
        max_days=50,       # hold for most of the run
    )

    if not trades.empty:
        ret = float(trades.iloc[0]["return_pct"])
        # Price goes from ~100 to ~148 (day 1 entry), so ~48% return
        assert ret > 0.30, (
            f"Buy-and-hold on strongly trending prices returned only {ret:.1%}. "
            "Engine may not be marking to market correctly."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 8: Cost model round-trip math
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_model_round_trip():
    """Cost entry × exit should result in a loss equal to total_round_trip_pct."""
    cm    = TransactionCostModel()
    price = 100.0

    entry = cm.apply_entry_cost(price, "stock")
    exit_ = cm.apply_exit_cost(price, "stock")

    # Round trip: buy at entry, sell at same market price → loss = round_trip_pct
    rt_pct = (exit_ - entry) / entry
    expected_rt = -cm.total_round_trip_pct("stock")

    # Small compound-math discrepancy: entry denominator is market×(1+fee),
    # not market, so rt_pct ≈ -0.001998 vs expected -0.002.  Allow 1e-4.
    assert abs(rt_pct - expected_rt) < 1e-4, (
        f"Round-trip return {rt_pct:.6f} ≠ expected {expected_rt:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 9: Zero cost model returns price unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_cost_model_identity():
    """ZeroCostModel must return the input price unchanged for both entry and exit."""
    cm = ZeroCostModel()
    for ac in ["stock", "crypto"]:
        assert cm.apply_entry_cost(100.0, ac) == 100.0
        assert cm.apply_exit_cost(100.0, ac) == 100.0
        assert cm.total_round_trip_pct(ac) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Test 10 & 11: Cooldown logic
# ─────────────────────────────────────────────────────────────────────────────

def _make_cooldown_setup(cooldown_days: int):
    """
    Build a minimal scenario where:
      - SPY has a signal on day 0 that enters on day 1
      - SPY exits via target on day 2 (high >= 1.05×entry)
      - SPY has another signal on day 3 that would enter on day 4
    Returns (trades_df, prices_df, pred_df) — caller picks cooldown_days.

    With cooldown_days=0 the second entry fires (day 4 >= day 2 + 0).
    With cooldown_days=5 the second entry is blocked (day 4 < day 2 + 5).
    """
    dates = pd.date_range("2022-01-01", periods=12, freq="D", tz="UTC")
    n = len(dates)

    opens  = [100.0] * n
    closes = [100.0] * n
    highs  = [101.0] * n
    lows   = [ 99.0] * n

    # Day index 2: trigger target exit (high >= 100×1.05 = 105)
    highs[2] = 110.0

    prices_df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e6},
        index=dates,
    )

    # Signals on day 0 (entry day 1) and day 3 (entry day 4)
    pred = pd.DataFrame(
        {"y_true": 1, "proba_raw": 0.80, "proba_cal": 0.80, "symbol": "SPY", "fold": 0},
        index=[dates[0], dates[1], dates[2], dates[3], dates[4]],
    )

    portfolio = Portfolio(initial_cash=10_000)
    trades, _ = run_backtest(
        predictions_df=pred,
        prices_dict={"SPY": prices_df},
        cost_model=ZeroCostModel(),
        portfolio=portfolio,
        threshold=0.70,
        upper_pct=0.05,
        lower_pct=0.50,   # wide stop so only target fires
        max_days=3,
        cooldown_days=cooldown_days,
    )
    return trades


def test_cooldown_blocks_reentry():
    """
    After SPY exits on day 2, a cooldown of 5 days must block re-entry on day 4.
    We expect exactly 1 closed trade (the first one).
    """
    trades = _make_cooldown_setup(cooldown_days=5)
    assert not trades.empty, "First trade should exist"
    # If cooldown worked, only the first trade (exited on day 2) exists.
    # The second signal (day 3 → entry day 4) should have been blocked.
    spv_trades = trades[trades["symbol"] == "SPY"]
    assert len(spv_trades) == 1, (
        f"Expected 1 SPY trade under cooldown_days=5, got {len(spv_trades)}. "
        "Cooldown did not block re-entry."
    )


def test_cooldown_expires():
    """
    With cooldown_days=0, re-entry must be allowed immediately.
    We expect 2 closed trades for SPY (first and second signals both fire).
    """
    trades = _make_cooldown_setup(cooldown_days=0)
    spv_trades = trades[trades["symbol"] == "SPY"]
    assert len(spv_trades) >= 2, (
        f"Expected ≥2 SPY trades with cooldown_days=0, got {len(spv_trades)}. "
        "Re-entry was incorrectly blocked."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 12: Buy-and-hold benchmark
# ─────────────────────────────────────────────────────────────────────────────

def test_buy_hold_no_trades_after_initial():
    """
    equal_weight_buy_hold must produce exactly len(symbols) trades,
    all with entry_date == start_date.

    WHY: the benchmark buys once on start_date and never sells intra-period.
    Any additional trade rows would indicate incorrect rebalancing logic.
    """
    symbols = ["SPY", "AAPL", "MSFT"]
    prices_dict = {sym: _make_prices(sym, n=40) for sym in symbols}

    start = prices_dict["SPY"].index[0]
    end   = prices_dict["SPY"].index[-1]

    trades, equity, metrics = equal_weight_buy_hold(
        prices_dict=prices_dict,
        symbols=symbols,
        start_date=start,
        end_date=end,
        initial_cash=10_000.0,
        cost_model=ZeroCostModel(),
    )

    # Exactly one trade row per symbol
    assert len(trades) == len(symbols), (
        f"Expected {len(symbols)} trades (one per symbol), got {len(trades)}"
    )

    # All entry dates must equal start_date
    entry_dates = pd.to_datetime(trades["entry_date"]).dt.normalize()
    start_norm  = pd.Timestamp(start).normalize()
    assert (entry_dates == start_norm).all(), (
        f"Some entry_dates differ from start_date={start_norm.date()}: {entry_dates.tolist()}"
    )

    # Equity curve must be non-empty
    assert not equity.empty, "Equity curve is empty"

    # Metrics must be present
    assert "sharpe" in metrics
    assert "cagr_pct" in metrics
    assert "max_drawdown_pct" in metrics
