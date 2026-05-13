"""
Model evaluation metrics and diagnostic plots.

WHY these specific metrics?
────────────────────────────
PR-AUC (Precision-Recall Area Under Curve):
  This is the primary metric for imbalanced binary classification.
  ROC-AUC is misleading when the positive class is rare — a model that
  mostly predicts "stop" will have high ROC-AUC but low PR-AUC.
  PR-AUC forces the model to be precise when it makes positive predictions.

ROC-AUC:
  Reported for completeness and literature comparability, but secondary.

Brier Score:
  Mean squared error between predicted probabilities and true labels.
  Measures "sharpness" — a model that confidently gets things right has
  a lower Brier score than one that hedges everything at 0.5.
  Lower is better. Random classifier: 0.25.

Precision at top decile:
  Of the trades the model is MOST confident about (top 10% by score),
  what fraction are actual wins?  This is the most operationally relevant
  metric — in practice you'd only trade the highest-conviction signals.

Precision at threshold (0.5, 0.6, 0.65, 0.7):
  As you raise the entry threshold, precision goes up but coverage goes
  down.  The threshold sweep plot shows this trade-off visually.

Expected Calibration Error (ECE):
  Measures whether predicted probabilities match empirical frequencies.
  A model that says 0.7 when the true frequency is 0.45 is miscalibrated —
  dangerous for position sizing formulas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
)

from src.models.calibration import expected_calibration_error


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """
    Compute all evaluation metrics for one (y_true, proba) pair.

    Returns a flat dict suitable for a DataFrame row.
    """
    y_true = np.asarray(y_true)
    proba  = np.asarray(proba)

    metrics: dict = {}

    # ── Core metrics ──────────────────────────────────────────────────────────
    metrics["pr_auc"]     = float(average_precision_score(y_true, proba))
    metrics["roc_auc"]    = float(roc_auc_score(y_true, proba))
    metrics["brier"]      = float(brier_score_loss(y_true, proba))
    metrics["ece"]        = float(expected_calibration_error(y_true, proba))

    # ── Precision at top decile ───────────────────────────────────────────────
    # Sort by descending score, take top 10%
    k               = max(1, int(len(y_true) * 0.10))
    top_k_idx       = np.argsort(proba)[::-1][:k]
    metrics["prec_top10"] = float(y_true[top_k_idx].mean())

    # ── Coverage and precision at fixed thresholds ───────────────────────────
    for thresh in [0.50, 0.60, 0.65, 0.70]:
        key     = f"prec_at_{int(thresh*100)}"
        cov_key = f"cov_at_{int(thresh*100)}"
        mask    = proba >= thresh
        if mask.sum() > 0:
            metrics[key]     = float(precision_score(y_true, mask.astype(int), zero_division=0))
            metrics[cov_key] = float(mask.sum() / len(y_true))
        else:
            metrics[key]     = float("nan")
            metrics[cov_key] = 0.0

    metrics["n_samples"]  = int(len(y_true))
    metrics["n_positive"] = int(y_true.sum())
    metrics["base_rate"]  = float(y_true.mean())

    return metrics


def plot_pr_curve(
    y_true: np.ndarray,
    proba:  np.ndarray,
    savepath: str | Path,
    label: str = "",
) -> None:
    """Precision-Recall curve saved to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    base_rate = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"{label} AP={ap:.3f}")
    ax.axhline(base_rate, ls="--", color="gray", lw=1, label=f"Baseline={base_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()

    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=120)
    plt.close(fig)


def plot_calibration_curve(
    y_true:   np.ndarray,
    proba:    np.ndarray,
    savepath: str | Path,
    n_bins:   int = 10,
    label:    str = "",
) -> None:
    """
    Calibration reliability diagram.

    Each dot represents one probability bin.  Perfect calibration = all dots
    on the diagonal.  Points above the diagonal = model is underconfident
    (predicted 0.4, actual rate 0.6).  Below = overconfident.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins)
    ece = expected_calibration_error(y_true, proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label=f"{label} ECE={ece:.4f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Reliability Diagram")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()

    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=120)
    plt.close(fig)


def plot_threshold_sweep(
    y_true:   np.ndarray,
    proba:    np.ndarray,
    savepath: str | Path,
    label:    str = "",
) -> None:
    """
    Precision, recall, and coverage as a function of decision threshold.

    WHY this plot matters operationally:
      You need to pick a threshold to decide "signal vs no signal."
      This plot shows the three-way trade-off:
        - Higher threshold → higher precision (fewer false signals)
        - Higher threshold → lower recall (miss more true signals)
        - Higher threshold → lower coverage (fewer trades)
      The "right" threshold depends on your transaction costs and
      how many positions you want to hold simultaneously.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0.3, 0.95, 60)
    precisions, recalls, coverages = [], [], []

    for t in thresholds:
        mask = proba >= t
        if mask.sum() == 0:
            precisions.append(float("nan"))
            recalls.append(0.0)
            coverages.append(0.0)
        else:
            precisions.append(precision_score(y_true, mask.astype(int), zero_division=0))
            recalls.append(float((y_true[mask]).sum() / max(y_true.sum(), 1)))
            coverages.append(float(mask.sum() / len(y_true)))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, precisions, "b-",  lw=2, label="Precision")
    ax1.plot(thresholds, recalls,    "g--", lw=2, label="Recall")
    ax2.plot(thresholds, coverages,  "r:",  lw=2, label="Coverage")

    ax1.axhline(np.nanmean(y_true), ls=":", color="gray", lw=1, label="Base rate")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Precision / Recall")
    ax2.set_ylabel("Coverage (fraction of rows)", color="r")
    ax1.set_title(f"Threshold Sweep{' — ' + label if label else ''}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=120)
    plt.close(fig)
