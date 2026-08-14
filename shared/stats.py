"""
shared/stats.py
===============
Statistical analysis of nested CV results.
"""

import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

from shared.utils import ensure_dir, save_json, get_logger

logger = get_logger(__name__)


def load_fold_scores(
    metrics_dir: str,
    model_names: List[str],
    metric_col: str = "test_f1_macro",
) -> Dict[str, List[float]]:
    scores = {}
    for model in model_names:
        csv_path = os.path.join(metrics_dir, model, "fold_metrics.csv")
        if not os.path.isfile(csv_path):
            logger.warning("stats: missing %s – skipping.", csv_path)
            continue
        df = pd.read_csv(csv_path)
        if metric_col not in df.columns:
            logger.warning("stats: column '%s' not in %s.", metric_col, csv_path)
            continue
        scores[model] = df[metric_col].dropna().tolist()
    return scores


def shapiro_wilk(scores: Dict[str, List[float]]) -> Dict[str, Any]:
    results = {}
    for model, vals in scores.items():
        if len(vals) < 3:
            results[model] = {"stat": None, "p_value": None, "normal": None,
                               "note": "insufficient samples"}
            continue
        stat, p = sp_stats.shapiro(vals)
        results[model] = {
            "stat":    float(stat),
            "p_value": float(p),
            "normal":  bool(p > 0.05),
        }
    return results


def friedman_test(scores: Dict[str, List[float]]) -> Dict[str, Any]:
    if len(scores) < 3:
        return {"stat": None, "p_value": None,
                "note": "Friedman requires ≥ 3 groups."}

    min_len = min(len(v) for v in scores.values())
    arrays  = [np.array(v[:min_len]) for v in scores.values()]

    stat, p = sp_stats.friedmanchisquare(*arrays)
    return {
        "stat":    float(stat),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
        "models_tested": list(scores.keys()),
        "n_folds":        min_len,
    }


def _cohens_d(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    n    = min(len(a), len(b))
    diff = a[:n] - b[:n]
    if np.std(diff) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def pairwise_wilcoxon(
    scores: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    models  = list(scores.keys())
    results = {}

    for m1, m2 in itertools.combinations(models, 2):
        key     = f"{m1}_vs_{m2}"
        vals_a  = scores[m1]
        vals_b  = scores[m2]
        n       = min(len(vals_a), len(vals_b))
        a, b    = np.array(vals_a[:n]), np.array(vals_b[:n])

        if np.allclose(a, b):
            results[key] = {"stat": 0.0, "p_value": 1.0,
                            "significant": False, "cohens_d": 0.0,
                            "note": "identical distributions"}
            continue

        try:
            stat, p = sp_stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError as e:
            results[key] = {"stat": None, "p_value": None,
                            "significant": None, "cohens_d": None,
                            "note": str(e)}
            continue

        cd = _cohens_d(vals_a, vals_b)
        results[key] = {
            "stat":        float(stat),
            "p_value":     float(p),
            "significant": bool(p < alpha),
            "cohens_d":    cd,
            "effect_size": (
                "negligible" if abs(cd) < 0.2 else
                "small"      if abs(cd) < 0.5 else
                "medium"     if abs(cd) < 0.8 else
                "large"
            ),
        }
    return results


def build_summary_table(
    scores: Dict[str, List[float]],
    pairwise: Dict[str, Any],
    ci_results: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    rows = []
    for model, vals in scores.items():
        arr  = np.array(vals)
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        sig_better = 0
        for key, res in pairwise.items():
            if model not in key:
                continue
            if not res.get("significant"):
                continue
            parts = key.split("_vs_")
            if len(parts) == 2:
                m1, m2 = parts
                other_model = m2 if m1 == model else m1
                other_mean  = float(np.mean(scores.get(other_model, [0])))
                if mean > other_mean:
                    sig_better += 1

        row = {
            "model":       model,
            "mean":        round(mean, 4),
            "std":         round(std, 4),
            "n_folds":     len(vals),
            "sig_better_than_n_models": sig_better,
        }
        if ci_results and model in ci_results:
            ci = ci_results[model]
            row["ci_low"]  = round(ci.get("ci_low_bootstrap", float("nan")), 4)
            row["ci_high"] = round(ci.get("ci_high_bootstrap", float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def plot_pvalue_heatmap(
    pairwise: Dict[str, Any],
    models: List[str],
    title: str,
    out_path: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    n   = len(models)
    mat = np.ones((n, n))

    for key, res in pairwise.items():
        parts = key.split("_vs_")
        if len(parts) != 2:
            continue
        m1, m2 = parts
        if m1 not in models or m2 not in models:
            continue
        i, j = models.index(m1), models.index(m2)
        p = res.get("p_value")
        if p is not None:
            mat[i, j] = float(p)
            mat[j, i] = float(p)

    np.fill_diagonal(mat, 1.0)
    df_mat = pd.DataFrame(mat, index=models, columns=models)

    plt.figure(figsize=(max(5, n), max(4, n - 1)))
    sns.heatmap(df_mat, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=0.1,
                linewidths=0.5, linecolor="grey",
                cbar_kws={"label": "p-value (Wilcoxon)"})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_effect_size_heatmap(
    pairwise: Dict[str, Any],
    models: List[str],
    title: str,
    out_path: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    n   = len(models)
    mat = np.zeros((n, n))

    for key, res in pairwise.items():
        parts = key.split("_vs_")
        if len(parts) != 2:
            continue
        m1, m2 = parts
        if m1 not in models or m2 not in models:
            continue
        i, j = models.index(m1), models.index(m2)
        cd = res.get("cohens_d")
        if cd is not None:
            mat[i, j] =  float(cd)
            mat[j, i] = -float(cd)

    df_mat = pd.DataFrame(mat, index=models, columns=models)
    plt.figure(figsize=(max(5, n), max(4, n - 1)))
    sns.heatmap(df_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, linecolor="grey",
                cbar_kws={"label": "Cohen's d (row − col)"})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_boxplot(
    scores: Dict[str, List[float]],
    metric_label: str,
    title: str,
    out_path: str,
) -> None:
    ensure_dir(os.path.dirname(out_path))
    data   = []
    models = []
    for model, vals in scores.items():
        data.extend(vals)
        models.extend([model] * len(vals))

    df = pd.DataFrame({"model": models, metric_label: data})
    plt.figure(figsize=(max(6, len(scores) * 1.5), 5))
    sns.boxplot(data=df, x="model", y=metric_label,
                palette="tab10", linewidth=1.5)
    sns.stripplot(data=df, x="model", y=metric_label,
                  color="black", size=5, alpha=0.5, jitter=True)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(metric_label)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_statistical_analysis(
    modality: str,
    model_names: List[str],
    metrics_dir: str,
    plots_dir: str,
    metric_col: str = "test_f1_macro",
    ci_results: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    stats_out_dir  = os.path.join(metrics_dir, modality, "stats")
    plots_out_dir  = os.path.join(plots_dir, modality)
    ensure_dir(stats_out_dir)
    ensure_dir(plots_out_dir)

    modality_metrics = os.path.join(metrics_dir, modality)
    scores = load_fold_scores(modality_metrics, model_names, metric_col)

    if len(scores) < 2:
        logger.warning("stats: fewer than 2 models with results — skipping for %s.", modality)
        return {"status": "skipped", "reason": "< 2 models with fold data"}

    normality = shapiro_wilk(scores)
    friedman  = friedman_test(scores)
    pairwise  = pairwise_wilcoxon(scores)

    summary_df = build_summary_table(scores, pairwise, ci_results)

    save_json(os.path.join(stats_out_dir, "normality_shapiro_wilk.json"), normality)
    save_json(os.path.join(stats_out_dir, "friedman_test.json"),          friedman)
    save_json(os.path.join(stats_out_dir, "pairwise_wilcoxon.json"),      pairwise)
    summary_df.to_csv(os.path.join(stats_out_dir, "stats_summary.csv"), index=False)

    logger.info(
        "[%s] Friedman p=%.4f | Significant: %s",
        modality.upper(),
        friedman.get("p_value") or float("nan"),
        friedman.get("significant_at_0.05"),
    )

    active_models = list(scores.keys())

    plot_pvalue_heatmap(
        pairwise, active_models,
        title=f"{modality.upper()} – Wilcoxon p-value matrix ({metric_col})",
        out_path=os.path.join(plots_out_dir, "stats_pvalue_heatmap.png"),
    )
    plot_effect_size_heatmap(
        pairwise, active_models,
        title=f"{modality.upper()} – Cohen's d effect size ({metric_col})",
        out_path=os.path.join(plots_out_dir, "stats_effect_size_heatmap.png"),
    )
    plot_score_boxplot(
        scores, metric_col,
        title=f"{modality.upper()} – Per-fold {metric_col} distribution",
        out_path=os.path.join(plots_out_dir, "stats_score_boxplot.png"),
    )

    return {
        "normality":  normality,
        "friedman":   friedman,
        "pairwise":   pairwise,
        "summary":    summary_df.to_dict(orient="records"),
    }
