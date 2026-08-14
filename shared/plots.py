"""
shared/plots.py
===============
All reusable plotting functions. Every figure is saved to disk (PNG, 150 dpi).
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from shared.utils import ensure_dir

# ── Consistent visual style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":    150,
})
PALETTE = sns.color_palette("tab10")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: List[List[float]],
    class_names: List[str],
    title: str,
    out_path: str,
    normalised: bool = True,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    fmt = ".2f" if normalised else "d"
    arr = np.array(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(arr, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="grey")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# ROC curves  (per-class OvR)
# ---------------------------------------------------------------------------
def plot_roc_curves(
    fpr: Dict[str, List[float]],
    tpr: Dict[str, List[float]],
    auc_per_class: Dict[str, float],
    title: str,
    out_path: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(7, 6))
    for i, (cname, fpr_vals) in enumerate(fpr.items()):
        auc_val = auc_per_class.get(cname, float("nan"))
        plt.plot(fpr_vals, tpr[cname], color=PALETTE[i % len(PALETTE)],
                 lw=2, label=f"{cname} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Metric bar chart with CI error bars
# ---------------------------------------------------------------------------
def plot_metric_comparison(
    summary_records: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    out_path: str,
    ylabel: Optional[str] = None,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    models  = [r["model"] for r in summary_records]
    means   = np.array([r["mean"] for r in summary_records])
    ci_lo   = np.array([r.get("ci_low_bootstrap",  r["mean"]) for r in summary_records])
    ci_hi   = np.array([r.get("ci_high_bootstrap", r["mean"]) for r in summary_records])
    err_lo  = means - ci_lo
    err_hi  = ci_hi - means

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    bars = ax.bar(x, means, color=PALETTE[:len(models)], alpha=0.85, width=0.55,
                  zorder=2)
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt="none",
                capsize=5, color="black", lw=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Training loss curve
# ---------------------------------------------------------------------------
def plot_training_history(
    history: Dict[str, List[float]],
    title: str,
    out_path: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], label="Train loss", lw=2)
    plt.plot(epochs, history["val_loss"],   label="Val loss",   lw=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical cross-entropy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Cross-fold violin / box plot
# ---------------------------------------------------------------------------
def plot_fold_distribution(
    fold_csv_path: str,
    metric_col: str,
    title: str,
    out_path: str,
    group_col: str = "model",
) -> None:
    ensure_dir(os.path.dirname(out_path))
    df = pd.read_csv(fold_csv_path)
    if metric_col not in df.columns:
        return

    plt.figure(figsize=(max(6, df[group_col].nunique() * 1.8), 5))
    sns.violinplot(data=df, x=group_col, y=metric_col,
                   palette="tab10", inner="box", cut=0)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(metric_col)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Summary heatmap  (models × metrics)
# ---------------------------------------------------------------------------
def plot_summary_heatmap(
    records: List[Dict[str, Any]],
    metric_keys: List[str],
    title: str,
    out_path: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    model_names = [r["model"] for r in records]
    data = np.array([[r.get(k, float("nan")) for k in metric_keys]
                     for r in records])
    df = pd.DataFrame(data, index=model_names, columns=metric_keys)
    plt.figure(figsize=(max(8, len(metric_keys) * 1.4), max(4, len(model_names) * 0.8)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="grey", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
