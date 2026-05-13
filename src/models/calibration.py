"""
Probability calibration via isotonic regression.

WHY isotonic regression over Platt scaling (sigmoid)?
  Platt scaling assumes the raw scores follow a sigmoid-shaped miscalibration.
  XGBoost and LightGBM scores often have more complex miscalibration shapes
  (e.g. a "bathtub" pattern where scores near 0.5 are underconfident and
  scores near 0 or 1 are overconfident).

  Isotonic regression is non-parametric — it fits a monotone step function
  that maps raw scores to empirical frequencies.  It adapts to any shape
  without assuming one.  The cost: it needs more calibration data (~200+
  examples) to be reliable; with fewer samples, Platt scaling is more stable.
  At ~200-400 rows per calibration fold we're in the safe range for isotonic.

WHY a separate calibration set, not cross-validation calibration (CalibratedCV)?
  sklearn's CalibratedClassifierCV uses k-fold internally — fine for iid data
  but it has the same temporal leakage problem as standard k-fold on time series.
  By explicitly passing a temporally held-out calibration slice (the last 20% of
  each training fold), we preserve time ordering end to end.

Expected Calibration Error (ECE):
  ECE measures how far the mean predicted probability in each bin deviates from
  the actual fraction of positives.  A perfectly calibrated model has ECE = 0.
  We report ECE before and after calibration in the evaluation module.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(raw_scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """
    Fit an isotonic regression calibrator.

    Parameters
    ----------
    raw_scores : shape (n,), uncalibrated positive-class probabilities
    y_true     : shape (n,), binary labels {0, 1}

    Returns
    -------
    Fitted IsotonicRegression that maps raw_score → calibrated_probability.
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_scores.reshape(-1, 1), y_true)
    return cal


def expected_calibration_error(
    y_true: np.ndarray,
    proba:  np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ_b (n_b / n) × |accuracy_b - confidence_b|

    where b ranges over equal-width probability bins, n_b is the number
    of samples in bin b, accuracy_b is the fraction of positives in that
    bin, and confidence_b is the mean predicted probability in that bin.

    WHY equal-width bins?
      Equal-width bins are standard for calibration curves and ECE, and
      are directly comparable to sklearn's calibration_curve output.
      Equal-frequency (quantile) bins are more robust for skewed score
      distributions but less common in practice.
    """
    bins      = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids   = np.digitize(proba, bins[1:-1])  # which bin each prediction falls in
    ece       = 0.0
    n         = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()    # fraction of true positives in bin
        conf = proba[mask].mean()     # mean predicted probability in bin
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)
