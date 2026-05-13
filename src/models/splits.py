"""
Purged Walk-Forward Cross-Validation split.

WHY can't we use standard k-fold CV for time-series?
────────────────────────────────────────────────────
Standard k-fold randomly shuffles data before splitting.  For time-series
this causes *temporal leakage*: a training fold may contain data from
AFTER the test fold.  The model effectively "sees the future" during
training, so CV scores are wildly optimistic and the live trading result
is a rude shock.

WHY not just use sklearn's TimeSeriesSplit?
  TimeSeriesSplit grows the training set each fold (anchored start).
  This is fine, but it doesn't account for:

  1. OVERLAP between label windows and the test set ("purging"):
     Our triple-barrier label for row t uses prices from t+1 to t+20.
     If row t is in the TRAINING set but t+15 is in the TEST set,
     the training label for row t was computed using test-set prices.
     This is a subtle but real form of leakage.  We must PURGE any
     training rows whose label window extends into the test period.

  2. AUTOCORRELATION near the train/test boundary ("embargo"):
     Even after purging, rows just before the test set are highly
     autocorrelated with rows just after it.  A momentum feature on
     day t-1 will partly predict what happens on day t+1.  Adding an
     embargo gap of e.g. 5 days between the last training row and the
     first test row reduces this correlation to near zero.

Reference: Marcos López de Prado, "Advances in Financial Machine Learning"
           Chapters 7 (cross-validation) and 4 (labelling).

sklearn compatibility:
  The class implements get_n_splits() and split() matching sklearn's
  BaseCrossValidator API so you can pass it directly to
  cross_val_score, GridSearchCV, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedWalkForwardSplit:
    """
    Walk-forward CV with purging and embargo.

    Parameters
    ----------
    n_splits     : number of test folds (5 → roughly annual folds over 5 years)
    embargo_days : rows to gap between train end and test start
                   (reduces autocorrelation leakage)
    purge_days   : length of the label window (= triple_barrier max_days)
                   any training row whose window overlaps the test set is removed

    The training set for each fold uses ALL data before the embargo gap
    (expanding window), which matches how a live system is retrained:
    "use everything available up to today."

    Example for n_splits=5 over 1000 rows, purge=20, embargo=5:
        fold 0: train=[0..179], test=[200..399]
        fold 1: train=[0..379], test=[400..599]
        fold 2: train=[0..579], test=[600..799]
        fold 3: train=[0..779], test=[800..999]
        (fold 4 not shown — n_splits=4 usable splits from 5-fold partition)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        purge_days: int = 20,
    ) -> None:
        self.n_splits     = n_splits
        self.embargo_days = embargo_days
        self.purge_days   = purge_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """sklearn compatibility: return number of splits."""
        # We get n_splits - 1 usable folds because the first "chunk" is
        # used only as training data for fold 0.
        return self.n_splits - 1

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y=None,
        groups=None,
    ):
        """
        Yield (train_indices, test_indices) for each fold.

        X must be sorted by time (ascending).  The indices returned are
        integer positional indices (iloc-compatible), not index labels.

        WHY positional indices?
          sklearn's fit() and predict() use iloc internally.  Returning
          positional indices makes this splitter a drop-in replacement for
          TimeSeriesSplit without any iloc/loc confusion.
        """
        n = len(X)
        if n < self.n_splits * (self.embargo_days + self.purge_days + 10):
            raise ValueError(
                f"Not enough rows ({n}) for {self.n_splits} splits with "
                f"purge={self.purge_days} and embargo={self.embargo_days}. "
                "Reduce n_splits or use more data."
            )

        # Divide data into n_splits equal-sized chunks
        indices    = np.arange(n)
        fold_size  = n // self.n_splits
        # Each "test" fold is one chunk; training is everything before it
        # minus the embargo gap.

        for fold in range(1, self.n_splits):
            # Test fold boundaries
            test_start = fold * fold_size
            test_end   = (fold + 1) * fold_size if fold < self.n_splits - 1 else n

            # Training: everything before the embargo gap
            train_end  = test_start - self.embargo_days

            if train_end <= 0:
                continue  # not enough data for this fold

            # Raw training indices (everything before embargo)
            train_idx = indices[:train_end]

            # ── Purge ──────────────────────────────────────────────────────
            # Remove training rows whose label window [t+1 … t+purge_days]
            # overlaps the test set [test_start … test_end).
            #
            # A training row t overlaps the test set if:
            #   t + purge_days >= test_start
            # i.e. t >= test_start - purge_days
            #
            # WHY this matters: if row 195 is in train and its label uses
            # prices from rows 196–215, and the test set starts at row 200,
            # then the label for row 195 was partly computed from test data.
            purge_cutoff = test_start - self.purge_days
            train_idx    = train_idx[train_idx < purge_cutoff]

            if len(train_idx) == 0:
                continue

            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def summary(self, X: pd.DataFrame) -> None:
        """Print a human-readable summary of the folds (useful for debugging)."""
        print(f"\nPurgedWalkForwardSplit: n_splits={self.n_splits}, "
              f"purge={self.purge_days}d, embargo={self.embargo_days}d")
        print(f"{'Fold':<6} {'Train rows':<12} {'Train end':<22} "
              f"{'Test rows':<12} {'Test start':<22} {'Test end'}")
        print("─" * 90)
        for fold, (tr, te) in enumerate(self.split(X)):
            tr_end   = X.index[tr[-1]]  if hasattr(X, 'index') else tr[-1]
            te_start = X.index[te[0]]   if hasattr(X, 'index') else te[0]
            te_end   = X.index[te[-1]]  if hasattr(X, 'index') else te[-1]
            print(
                f"{fold:<6} {len(tr):<12} {str(tr_end)[:19]:<22} "
                f"{len(te):<12} {str(te_start)[:19]:<22} {str(te_end)[:19]}"
            )
        print()
