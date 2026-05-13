"""
XGBoost classifier wrapper.

WHY these specific defaults?
────────────────────────────
n_estimators=300, learning_rate=0.05:
  Shrinkage (low LR) + many trees is the standard XGBoost recipe for
  generalisation.  Early stopping (patience=30) will usually stop us
  well before 300 trees when a val set is provided.

max_depth=5:
  Depth-5 trees can capture 5-way interactions (e.g. "RSI is high AND
  volume is spiking AND price is above SMA200").  Deeper trees overfit
  noise; shallower trees miss important interactions.  5 is the sweet
  spot for most tabular financial datasets.

subsample=0.8, colsample_bytree=0.8:
  Stochastic gradient boosting — each tree sees 80% of rows and 80%
  of features randomly sampled.  This is the single most effective
  regulariser after learning_rate.  It forces the ensemble to learn
  from different subsets, preventing any one pattern from dominating.

scale_pos_weight=(neg/pos):
  Our dataset is imbalanced: ~55-60% stop-outs (label 0) vs 20-44%
  targets (label 1).  scale_pos_weight tells XGBoost to weight the
  minority class proportionally so it doesn't learn to always predict
  "stop."  It's equivalent to oversampling the positive class.
  Value = count(negatives) / count(positives) per training fold.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from src.models.base import Model


class XGBoostModel(Model):
    """XGBoost binary classifier with calibration support."""

    def __init__(
        self,
        n_estimators:    int   = 300,
        max_depth:       int   = 5,
        learning_rate:   float = 0.05,
        subsample:       float = 0.8,
        colsample_bytree: float = 0.8,
        random_state:    int   = 42,
        early_stopping_rounds: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )
        self.early_stopping_rounds = early_stopping_rounds

    def _build_estimator(self, with_early_stopping: bool = False) -> xgb.XGBClassifier:
        """
        Build the XGBClassifier, including early_stopping_rounds only when a
        validation set will actually be provided.

        WHY conditional early stopping?
          XGBoost 3.x made early_stopping_rounds a constructor-level callback
          that fires on EVERY iteration.  If no eval_set is passed to fit(),
          the callback raises an error because it has nothing to monitor.
          We solve this by only enabling early stopping when we have a val set.
        """
        kw = {k: v for k, v in self.kwargs.items()
              if k != "early_stopping_rounds"}
        if with_early_stopping:
            kw["early_stopping_rounds"] = self.early_stopping_rounds
        return xgb.XGBClassifier(
            **kw,
            eval_metric="logloss",
            verbosity=0,
        )

    def _fit_estimator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None,
        y_val:   np.ndarray | None,
    ) -> None:
        has_val = (X_val is not None and y_val is not None)

        # Rebuild the estimator with early stopping only if we have a val set
        self.estimator = self._build_estimator(with_early_stopping=has_val)

        # Compute class imbalance ratio on this fold's training data
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        self.estimator.set_params(scale_pos_weight=spw)

        fit_kwargs: dict = {"verbose": False}
        if has_val:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.estimator.fit(X_train, y_train, **fit_kwargs)

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]
