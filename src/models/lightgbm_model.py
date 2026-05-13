"""
LightGBM classifier wrapper.

WHY LightGBM alongside XGBoost?
  XGBoost and LightGBM are both gradient boosted trees but differ in:
  - Split finding: XGBoost uses exact greedy; LightGBM uses histogram binning
    (faster, slightly less precise on small datasets, often better on large ones)
  - Leaf growth: XGBoost grows level-wise; LightGBM grows leaf-wise
    (leaf-wise can model sharper non-linearities, but risks overfitting)
  - Categorical handling: LightGBM has native support

  Running both and comparing their OOF PR-AUC gives us a free ensemble
  signal: if both models agree on high probability, that's a stronger
  signal than either alone.  The compare_models script will show us which
  is better on each symbol.

Comparable hyperparameters:
  We keep n_estimators, learning_rate, max_depth semantically equivalent
  so the comparison is apples-to-apples.  LightGBM's num_leaves (which
  controls tree complexity more directly than max_depth in leaf-wise mode)
  is set to 2^(max_depth-1) = 16 as a sensible default.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb

from src.models.base import Model


class LightGBMModel(Model):
    """LightGBM binary classifier with calibration support."""

    def __init__(
        self,
        n_estimators:    int   = 300,
        max_depth:       int   = 5,
        learning_rate:   float = 0.05,
        subsample:       float = 0.8,
        colsample_bytree: float = 0.8,
        num_leaves:      int   = 16,
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
            num_leaves=num_leaves,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )
        self.early_stopping_rounds = early_stopping_rounds

    def _build_estimator(self) -> lgb.LGBMClassifier:
        kw = {k: v for k, v in self.kwargs.items()
              if k not in ("early_stopping_rounds",)}
        return lgb.LGBMClassifier(
            **kw,
            verbose=-1,          # suppress all LightGBM output
            n_jobs=-1,
        )

    def _fit_estimator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None,
        y_val:   np.ndarray | None,
    ) -> None:
        # Mirror XGBoost's scale_pos_weight using class_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        self.estimator.set_params(class_weight={0: 1.0, 1: spw})

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            # LightGBM's early stopping callback API (v4+)
            callbacks = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),   # silence per-iter output
            ]
            fit_kwargs["eval_set"]   = [(X_val, y_val)]
            fit_kwargs["callbacks"]  = callbacks

        self.estimator.fit(X_train, y_train, **fit_kwargs)

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]
