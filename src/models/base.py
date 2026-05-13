"""
Abstract Model base class.

WHY wrap the estimator AND calibrator together in one object?
  Raw XGBoost/LightGBM probabilities are not well-calibrated — XGBoost
  tends to push scores toward 0 and 1 (overconfident), while LightGBM
  can be underconfident.  A "probability" of 0.72 that is actually 0.55
  when measured against ground truth is dangerous: it will cause a
  position-sizing formula to over-allocate capital.

  By making the calibrator part of the model object, every call to
  predict_proba() returns a calibrated probability automatically.
  The consumer never has to remember to apply calibration separately.

WHY save both together?
  A model saved without its calibrator is incomplete — loading just the
  sklearn/xgb weights and calling predict_proba() would return raw,
  uncalibrated scores that look like calibrated ones.  Saving them as a
  unit prevents this footgun.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Model(ABC):
    """
    Abstract wrapper around a fitted estimator + fitted calibrator.

    Concrete subclasses implement _build_estimator() and _raw_proba().
    The base class owns the calibration fit/apply lifecycle.
    """

    def __init__(self, **kwargs) -> None:
        # Model hyperparameters — stored so save/load can reconstruct
        self.kwargs     = kwargs
        self.estimator  = None   # set by fit()
        self.calibrator = None   # set by fit() via src.models.calibration
        self._is_fitted = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _build_estimator(self) -> object:
        """Return a fresh, unfitted sklearn-compatible estimator."""
        ...

    @abstractmethod
    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw (uncalibrated) positive-class probabilities.
        Called internally; consumers should use predict_proba().
        """
        ...

    # ── Concrete lifecycle ────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
        X_cal:   np.ndarray | None = None,
        y_cal:   np.ndarray | None = None,
    ) -> "Model":
        """
        Fit the estimator and then fit the calibrator.

        Parameters
        ----------
        X_train, y_train : primary training data
        X_val,   y_val   : optional validation set for early stopping
        X_cal,   y_cal   : calibration set (must be disjoint from train + test)
                           If None, calibration is skipped (scores are raw).

        WHY separate cal from val?
          Early stopping uses val to prevent over-fitting iterations.
          Calibration uses cal to learn the mapping from raw score → probability.
          Using the same data for both conflates two distinct objectives and
          can underfit the calibration mapping.
        """
        from src.models.calibration import fit_calibrator

        self.estimator = self._build_estimator()
        self._fit_estimator(X_train, y_train, X_val, y_val)

        if X_cal is not None and y_cal is not None and len(np.unique(y_cal)) > 1:
            raw_cal = self._raw_proba(X_cal)
            self.calibrator = fit_calibrator(raw_cal, y_cal)
        else:
            self.calibrator = None

        self._is_fitted = True
        return self

    @abstractmethod
    def _fit_estimator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None,
        y_val:   np.ndarray | None,
    ) -> None:
        """Fit self.estimator in place.  Subclasses handle early stopping."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated positive-class probabilities, shape (n,).

        WHY shape (n,) not (n, 2)?
          This is a binary problem.  Returning the positive-class probability
          directly avoids the common bug of using [:, 0] instead of [:, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        raw = self._raw_proba(X)

        if self.calibrator is not None:
            return self.calibrator.predict(raw.reshape(-1, 1))
        return raw

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted model + calibrator to a pickle file.

        WHY pickle and not joblib or ONNX?
          For this project size (10 symbols × 4 folds × 2 algorithms = 80
          model files), pickle is fine.  Joblib is better for large numpy
          arrays; ONNX is better for cross-language serving.  We can migrate
          later if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "estimator":  self.estimator,
            "calibrator": self.calibrator,
            "kwargs":     self.kwargs,
            "class":      self.__class__.__name__,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=5)

    @classmethod
    def load(cls, path: str | Path) -> "Model":
        """Restore a saved model from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(**payload["kwargs"])
        obj.estimator  = payload["estimator"]
        obj.calibrator = payload["calibrator"]
        obj._is_fitted = True
        return obj
