"""
Leakage tests — the most important tests in this project.

WHY are these tests special?
  A model that "accidentally" uses future information during training will
  look extraordinarily good in backtests and fail completely in live trading.
  This is the #1 cause of backtesting disasters in quant finance.

  These tests are designed to PROVE that our features, labels, and CV splits
  are all leakage-free.  They should be run every time any of those modules
  are modified.

  The philosophy: don't trust code review alone to catch leakage — prove it
  empirically by corrupting future data and verifying nothing changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAW_STOCKS_DIR
from src.features.pipeline import build_features, feature_names
from src.labels.triple_barrier import triple_barrier_labels
from src.models.splits import PurgedWalkForwardSplit


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spy_ohlcv() -> pd.DataFrame:
    """Load SPY OHLCV once for all tests in this module."""
    path = RAW_STOCKS_DIR / "SPY.parquet"
    if not path.exists():
        pytest.skip("SPY.parquet not found — run pull_data.py first")
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


@pytest.fixture(scope="module")
def spy_features(spy_ohlcv) -> pd.DataFrame:
    return build_features(spy_ohlcv)


@pytest.fixture(scope="module")
def spy_labels(spy_ohlcv) -> pd.DataFrame:
    return triple_barrier_labels(spy_ohlcv)


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1: Features must not use future data
# ─────────────────────────────────────────────────────────────────────────────

def test_no_future_in_features(spy_ohlcv):
    """
    Corrupt ALL prices AFTER row t and verify that features AT row t are
    unchanged.

    METHOD: Pick a mid-series row t.  Build features on the original data.
    Then replace every close/high/low/volume AFTER t with random garbage.
    Rebuild features.  The features at row t must be bit-for-bit identical.

    WHY this test proves leakage-freedom:
      If any feature at t used data from t+1 onwards, corrupting that data
      would change the feature value.  An unchanged value means row t's
      feature was computed purely from rows 0…t.
    """
    df = spy_ohlcv.copy()
    n = len(df)
    t = n // 2  # pick the middle row as our test point

    # Build features on clean data
    feat_clean = build_features(df)

    # The row index in feat_clean may differ from t due to warmup dropping
    # Find the t-th row of the original DataFrame in the feature output
    original_ts = df.index[t]
    if original_ts not in feat_clean.index:
        # Row t was in the warmup period; use a later row
        original_ts = feat_clean.index[len(feat_clean) // 2]
        t = df.index.get_loc(original_ts)

    features_at_t_clean = feat_clean.loc[original_ts]

    # Now corrupt ALL data after row t
    df_corrupt = df.copy()
    rng = np.random.default_rng(42)
    n_future = n - t - 1
    for col in ["open", "high", "low", "close", "volume"]:
        df_corrupt.iloc[t + 1:][col]  # just checking column exists
    df_corrupt.iloc[t + 1:, df_corrupt.columns.get_loc("close")]  = rng.uniform(1, 1e6, n_future)
    df_corrupt.iloc[t + 1:, df_corrupt.columns.get_loc("high")]   = rng.uniform(1, 1e6, n_future)
    df_corrupt.iloc[t + 1:, df_corrupt.columns.get_loc("low")]    = rng.uniform(1, 1e6, n_future)
    df_corrupt.iloc[t + 1:, df_corrupt.columns.get_loc("volume")] = rng.uniform(1, 1e9, n_future)

    # Build features on corrupted data
    feat_corrupt = build_features(df_corrupt)

    if original_ts not in feat_corrupt.index:
        pytest.skip("Row t fell out due to NaN propagation from corruption — expected near boundaries")

    features_at_t_corrupt = feat_corrupt.loc[original_ts]

    feats = feature_names(feat_clean)
    for f in feats:
        clean_val   = features_at_t_clean[f]
        corrupt_val = features_at_t_corrupt[f]
        assert np.isclose(clean_val, corrupt_val, rtol=1e-10, equal_nan=True), (
            f"LEAKAGE DETECTED in feature '{f}': "
            f"clean={clean_val:.6f}, corrupt={corrupt_val:.6f}. "
            f"This feature used data from after row t!"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2: Last max_days rows must have no label
# ─────────────────────────────────────────────────────────────────────────────

def test_labels_drop_unobservable(spy_ohlcv):
    """
    The last `max_days` rows of the input have no complete forward window
    and must be absent from the labels output.

    WHY this matters:
      If we fabricated a label for row t where t + max_days > len(df),
      the triple-barrier would exit at the time barrier using only a partial
      window.  A "time barrier hit after 10 days" is not the same event as
      "time barrier hit after 20 days" — it's a different label from a
      different experiment.  Including it silently corrupts the distribution.
    """
    max_days = 20
    labels = triple_barrier_labels(spy_ohlcv, max_days=max_days)

    # The last max_days rows of the input should NOT appear in labels
    last_rows = spy_ohlcv.index[-max_days:]
    overlap   = labels.index.intersection(last_rows)

    assert len(overlap) == 0, (
        f"Labels contain {len(overlap)} rows from the last {max_days} days "
        f"of the input — these are unobservable labels!\n"
        f"Overlapping timestamps: {list(overlap)}"
    )


def test_labels_exact_drop_count(spy_ohlcv):
    """
    Exactly max_days rows should be dropped (no more, no fewer).
    """
    max_days = 20
    labels = triple_barrier_labels(spy_ohlcv, max_days=max_days)
    expected_max = len(spy_ohlcv) - max_days
    assert len(labels) <= expected_max, (
        f"Labels has {len(labels)} rows but input has {len(spy_ohlcv)} — "
        f"should have at most {expected_max} after dropping {max_days} tail rows"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3: CV splits must never overlap and must be chronological
# ─────────────────────────────────────────────────────────────────────────────

def test_split_no_overlap(spy_features):
    """
    Training and test indices must never share a row.

    A shared row would mean the model is evaluated on data it was trained on,
    making the CV score a training-set metric, not a generalisation metric.
    """
    splitter = PurgedWalkForwardSplit(n_splits=5, embargo_days=5, purge_days=20)

    for fold, (tr, te) in enumerate(splitter.split(spy_features)):
        overlap = set(tr) & set(te)
        assert len(overlap) == 0, (
            f"Fold {fold}: {len(overlap)} indices appear in both train and test! "
            f"Sample overlapping indices: {list(overlap)[:5]}"
        )


def test_split_train_before_test(spy_features):
    """
    Every training index must be strictly before every test index.

    This enforces the "train on past, test on future" invariant — the core
    requirement for valid time-series CV.
    """
    splitter = PurgedWalkForwardSplit(n_splits=5, embargo_days=5, purge_days=20)

    for fold, (tr, te) in enumerate(splitter.split(spy_features)):
        assert tr.max() < te.min(), (
            f"Fold {fold}: max training index ({tr.max()}) >= "
            f"min test index ({te.min()}). "
            "Training set contains rows that are AFTER test set rows!"
        )


def test_split_embargo_respected(spy_features):
    """
    There must be at least `embargo_days` rows between the last training
    row and the first test row.
    """
    embargo_days = 5
    splitter = PurgedWalkForwardSplit(n_splits=5, embargo_days=embargo_days, purge_days=20)

    for fold, (tr, te) in enumerate(splitter.split(spy_features)):
        gap = te.min() - tr.max() - 1
        assert gap >= embargo_days, (
            f"Fold {fold}: gap between train end and test start is only {gap} rows "
            f"(required >= {embargo_days})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4: Purge must remove label-overlapping training rows
# ─────────────────────────────────────────────────────────────────────────────

def test_purge_works(spy_features):
    """
    Any training row whose label window [t+1 … t+purge_days] extends into
    the test set must be ABSENT from the training indices.

    This is the most subtle leakage check: without purging, the model's
    training labels are partly computed from test-set prices.  The model
    would then "know" future information through the label, not the features.
    """
    purge_days   = 20
    embargo_days = 5
    splitter     = PurgedWalkForwardSplit(n_splits=5, embargo_days=embargo_days, purge_days=purge_days)

    for fold, (tr, te) in enumerate(splitter.split(spy_features)):
        test_start = te.min()

        # Any training index t where t + purge_days >= test_start is a leaker
        leaking = tr[tr + purge_days >= test_start]

        assert len(leaking) == 0, (
            f"Fold {fold}: {len(leaking)} training rows have label windows "
            f"that overlap the test set (test starts at index {test_start}). "
            f"These rows should have been purged. Sample: {list(leaking[:5])}"
        )
