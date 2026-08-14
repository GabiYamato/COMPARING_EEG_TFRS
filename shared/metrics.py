"""
shared/metrics.py
=================
All evaluation metrics required by the study:
  Accuracy, Balanced Accuracy
  F1       – micro, macro, weighted  (+ per-class)
  Precision– micro, macro, weighted  (+ per-class)
  Recall   – micro, macro, weighted  (+ per-class)
  AUC      – micro, macro, weighted OvR (+ per-class)
  FPR      – per-class ROC curve
  95% Confidence Intervals via bootstrap resampling

Notes
-----
  micro   : aggregate TP/FP/FN globally → equals overall Accuracy for F1
  macro   : mean of per-class scores, equal weight regardless of class size
  weighted: mean of per-class scores, weighted by class support (# true samples)
"""

from typing import Any, Dict, List

import numpy as np
from scipy.stats import sem, t as t_dist
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    y_true      : integer class labels   (N,)
    y_prob      : softmax probabilities  (N, C)
    class_names : list of string labels of length C

    Returns
    -------
    dict with all metrics; serialisable as JSON via save_json.
    """
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = len(class_names)

    # ── Accuracy ──────────────────────────────────────────────────────────
    acc  = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))

    # ── F1 / Precision / Recall — micro, macro, weighted ──────────────────
    prec_micro,    rec_micro,    f1_micro,    _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro",    zero_division=0
    )
    prec_macro,    rec_macro,    f1_macro,    _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro",    zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ── Per-class F1 / Precision / Recall ─────────────────────────────────
    prec_pc, rec_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_scores: Dict[str, Any] = {}
    for i, cname in enumerate(class_names):
        per_class_scores[cname] = {
            "precision": float(prec_pc[i]),
            "recall":    float(rec_pc[i]),
            "f1":        float(f1_pc[i]),
            "support":   int(support_pc[i]),
        }

    # ── Binarise for ROC / AUC ────────────────────────────────────────────
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # Per-class FPR, TPR, AUC
    fpr_per_class: Dict[str, List[float]] = {}
    tpr_per_class: Dict[str, List[float]] = {}
    auc_per_class: Dict[str, float]       = {}
    for i, cname in enumerate(class_names):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        fpr_per_class[cname] = fpr_i.tolist()
        tpr_per_class[cname] = tpr_i.tolist()
        auc_per_class[cname] = float(auc(fpr_i, tpr_i))

    def _safe_auc(average: str) -> float:
        try:
            return float(
                roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average=average)
            )
        except ValueError:
            return float("nan")

    roc_auc_macro    = _safe_auc("macro")
    roc_auc_micro    = _safe_auc("micro")
    roc_auc_weighted = _safe_auc("weighted")

    # Mean FPR across classes (scalar summary for logging)
    mean_fpr = float(np.mean([np.mean(v) for v in fpr_per_class.values()]))

    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    return {
        # ── Accuracy ──────────────────────────────────────────────────────
        "accuracy":              acc,
        "balanced_accuracy":     bacc,

        # ── F1 ────────────────────────────────────────────────────────────
        "f1_micro":              float(f1_micro),       # == accuracy in multiclass
        "f1_macro":              float(f1_macro),       # equal class weight
        "f1_weighted":           float(f1_weighted),    # support-weighted

        # ── Precision ─────────────────────────────────────────────────────
        "precision_micro":       float(prec_micro),
        "precision_macro":       float(prec_macro),
        "precision_weighted":    float(prec_weighted),

        # ── Recall ────────────────────────────────────────────────────────
        "recall_micro":          float(rec_micro),
        "recall_macro":          float(rec_macro),
        "recall_weighted":       float(rec_weighted),

        # ── Per-class breakdown ───────────────────────────────────────────
        "per_class":             per_class_scores,

        # ── AUC (OvR) ─────────────────────────────────────────────────────
        "roc_auc_micro":         roc_auc_micro,
        "roc_auc_macro":         roc_auc_macro,
        "roc_auc_weighted":      roc_auc_weighted,
        "roc_auc_per_class":     auc_per_class,

        # ── FPR / TPR curves ──────────────────────────────────────────────
        "mean_fpr":              mean_fpr,
        "fpr":                   fpr_per_class,
        "tpr":                   tpr_per_class,

        # ── Confusion matrix ──────────────────────────────────────────────
        "confusion_matrix":             cm.tolist(),
        "confusion_matrix_normalized":  cm_norm.tolist(),

        # ── Full sklearn classification report ────────────────────────────
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Bootstrap 95% confidence interval
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    Non-parametric bootstrap CI over a list of per-fold scalar values.
    Also returns the t-distribution CI for comparison.
    """
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n   = len(arr)

    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "se": float("nan"),
                "ci_low_bootstrap": float("nan"),
                "ci_high_bootstrap": float("nan"),
                "ci_low_t": float("nan"),
                "ci_high_t": float("nan")}

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1) if n > 1 else 0.0)
    se   = float(sem(arr)) if n > 1 else 0.0

    # t-distribution CI
    if n > 1:
        t_lo, t_hi = t_dist.interval(ci, df=n - 1, loc=mean, scale=se)
    else:
        t_lo = t_hi = mean

    # Bootstrap CI
    rng   = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(arr, size=n, replace=True))
             for _ in range(n_bootstrap)]
    alpha = 1.0 - ci
    b_lo  = float(np.quantile(boots, alpha / 2.0))
    b_hi  = float(np.quantile(boots, 1.0 - alpha / 2.0))

    return {
        "mean":               mean,
        "std":                std,
        "se":                 se,
        "ci_low_bootstrap":   b_lo,
        "ci_high_bootstrap":  b_hi,
        "ci_low_t":           float(t_lo),
        "ci_high_t":          float(t_hi),
    }


# ---------------------------------------------------------------------------
# Aggregate across outer folds
# ---------------------------------------------------------------------------
def aggregate_outer_folds(
    fold_metrics: List[Dict[str, Any]],
    key: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Pull `key` from each fold's metrics dict and compute CI."""
    values = [m[key] for m in fold_metrics if np.isfinite(m.get(key, float("nan")))]
    return bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)
