"""
Triple-barrier labelling method.

WHY triple-barrier labels instead of simple "up N% in 20 days"?
─────────────────────────────────────────────────────────────────
Simple binary labels ("was the price higher after 20 days?") have two
critical flaws for ML:

  1. Path ignorance: a trade that hit -15% on day 10 and recovered to +1%
     on day 20 is labelled as a WIN.  In reality, any stop-loss would have
     exited at day 10.  The label lies about actual trading experience.

  2. Equal class weight at fixed horizon: with an asymmetric market (e.g.
     60% of 20-day windows are positive), the model is implicitly being
     trained to predict "up" because of market drift, not because of its
     features.

Triple-barrier fixes both problems by asking: "which wall does price
touch FIRST — the profit target (upper), the stop-loss (lower), or the
time limit?"

  upper barrier: entry × (1 + upper)   → label = +1
  lower barrier: entry × (1 - lower)   → label = -1
  time barrier:  max_days from entry   → label =  0

WHY asymmetric barriers (5% profit / 3% stop)?
  This reflects real trading psychology and math:
  - A 5% profit target with a 3% stop gives a reward-to-risk ratio of 1.67.
  - The asymmetry means the model must find genuinely good setups to show
    profit — it can't just take lots of small winners and blow up on a few
    large losers.

LEAKAGE RULE — critical:
  Labels use future prices (rows t+1 to t+max_days).  This is CORRECT and
  INTENTIONAL — labels are the *thing we're trying to predict*, not a feature.
  The leakage rule is: features must NEVER use future prices.  Labels must
  use only the forward window they claim to represent.

  Rows in the last `max_days` positions of the time series have incomplete
  forward windows.  We drop them rather than fabricating labels — this is the
  "unobservable label" problem.  A label computed over only 10 days when we
  claimed 20 days is NOT the same label; including it corrupts the training
  distribution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def triple_barrier_labels(
    df: pd.DataFrame,
    upper: float = 0.05,   # +5% profit target
    lower: float = 0.03,   # -3% stop loss
    max_days: int = 20,    # time barrier
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for every row in `df`.

    Parameters
    ----------
    df       : OHLCV DataFrame (must contain 'close', 'high', 'low').
               Rows are assumed to be daily bars in chronological order.
    upper    : profit-target fraction (e.g. 0.05 = +5%)
    lower    : stop-loss fraction      (e.g. 0.03 = -3%)
    max_days : maximum holding period in trading days

    Returns
    -------
    DataFrame with columns:
        label           : +1 (target hit), -1 (stop hit),  0 (time barrier)
        barrier_hit     : "upper" | "lower" | "time"
        days_held       : integer, 1 to max_days
        entry_price     : close price at row t
        exit_price      : price when barrier was hit
        realized_return : (exit_price - entry_price) / entry_price

    Rows for which the full forward window is unavailable (the last max_days
    rows) are DROPPED.  Never fabricate labels for incomplete windows.
    """
    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    n      = len(df)

    # Pre-allocate result arrays with sentinels so any bug is obvious
    labels    = np.full(n, np.nan)
    barrier   = np.full(n, "", dtype=object)
    days_held = np.full(n, -1, dtype=int)
    entry_px  = np.full(n, np.nan)
    exit_px   = np.full(n, np.nan)

    # WHY iterate row by row instead of a vectorised approach?
    #   The triple-barrier check is inherently sequential: we stop as soon
    #   as the FIRST barrier is touched.  A vectorised "did price ever exceed
    #   X in the next 20 days?" would find the *final* state, not the *first*
    #   touch.  Row-by-row is slower but correct.  For 1600 rows it runs in
    #   milliseconds; at 1M rows we'd vectorise with numba.
    #
    # WHY n - max_days (not n)?
    #   The last max_days rows don't have a complete forward window.
    #   We compute labels only for rows [0 … n - max_days - 1].

    for t in range(n - max_days):
        ep = close[t]
        if ep == 0 or np.isnan(ep):
            continue

        upper_level = ep * (1 + upper)
        lower_level = ep * (1 - lower)

        hit_label   = 0
        hit_barrier = "time"
        hit_day     = max_days
        hit_price   = close[t + max_days]  # default: time barrier exit

        # Walk forward day by day
        for d in range(1, max_days + 1):
            fwd = t + d
            h   = high[fwd]
            l   = low[fwd]

            # WHY check high >= upper AND low <= lower on the same bar?
            #   Both could be true on the same bar (a wide-range day).
            #   The convention is: whichever barrier is touched FIRST within
            #   the bar wins.  Since we can't observe intra-day order on daily
            #   bars, we break the tie in favour of the upper barrier
            #   (optimistic, but consistently applied).
            if h >= upper_level:
                hit_label   = 1
                hit_barrier = "upper"
                hit_day     = d
                hit_price   = upper_level  # assume filled at barrier
                break
            elif l <= lower_level:
                hit_label   = -1
                hit_barrier = "lower"
                hit_day     = d
                hit_price   = lower_level
                break

        labels[t]    = hit_label
        barrier[t]   = hit_barrier
        days_held[t] = hit_day
        entry_px[t]  = ep
        exit_px[t]   = hit_price

    # Build result DataFrame aligned to the original index
    result = pd.DataFrame(
        {
            "label":            labels,
            "barrier_hit":      barrier,
            "days_held":        days_held,
            "entry_price":      entry_px,
            "exit_price":       exit_px,
        },
        index=df.index,
    )

    result["realized_return"] = (
        (result["exit_price"] - result["entry_price"]) / result["entry_price"]
    )

    # Drop rows with no label (last max_days rows + any NaN entry prices)
    # WHY drop instead of fill with 0?
    #   Filling with 0 ("time barrier") would be a fabricated label — we
    #   literally don't know what would have happened.  The model must never
    #   be trained on fake data.
    result = result.dropna(subset=["label"])
    result = result[result["days_held"] > 0]  # drop sentinel rows

    result["label"]     = result["label"].astype(int)
    result["days_held"] = result["days_held"].astype(int)

    n_dropped = n - len(result)
    log.info(
        "triple_barrier_labels: %d → %d rows (%d dropped, upper=%.1f%%, lower=%.1f%%, max_days=%d)",
        n, len(result), n_dropped, upper * 100, lower * 100, max_days,
    )

    return result
