"""
main.py
=======
Entry point for cross-validation EEG time-frequency representation (TFR) classification experiments.

Usage examples:
--------------
# Run all ablations and models for MEL spectrograms:
  python main.py --dataset mel --mode all

# Run ablation experiments only for STFT:
  python main.py --dataset stft --mode ablation

# Run new models (CNN_LSTM, CNN_BiLSTM, Swin_Transformer) for CWT:
  python main.py --dataset cwt --mode new

# Run single specific model:
  python main.py --dataset mel --model ResBiLSTM

# Smoke test (2 outer / 2 inner folds):
  python main.py --dataset mel --smoke
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import ExperimentConfig
from shared.kfold import run_nested_cv
from shared.plots import plot_metric_comparison, plot_summary_heatmap
from shared.stats import run_statistical_analysis
from shared.utils import ensure_dir, get_logger, save_json, set_global_seed
from notebook_generator import generate_all_notebooks
from csv_generator import generate_all_csvs

logger = get_logger("main")

MODALITY_LOADERS = {
    "stft": ("stft.dataset", "load_stft_dataset", "stft_root"),
    "cwt":  ("cwt.dataset",  "load_cwt_dataset",  "cwt_root"),
    "mel":  ("mel.dataset",  "load_mel_dataset",  "mel_root"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nested Stratified K-Fold CV – EEG TFR Classification & Ablation Benchmark"
    )
    p.add_argument("--dataset", choices=["stft", "cwt", "mel", "all"],
                   default="stft", help="Which dataset modality to run.")
    p.add_argument("--mode", choices=["ablation", "models", "new", "all"],
                   default="all", help="Which suite to run: ablation, models, new models, or all.")
    p.add_argument("--model", default=None,
                   help="Restrict to a single model (e.g., cCNN_A1_NoPooling, CNN_LSTM, Swin_Transformer, ResBiLSTM).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 2-outer / 2-inner folds for quick verification.")
    p.add_argument("--outer", type=int, default=None,
                   help="Override outer folds (default: 10).")
    p.add_argument("--inner", type=int, default=None,
                   help="Override inner folds (default: 3).")
    p.add_argument("--img_size", type=int, default=64,
                   help="Resize frames to img_size × img_size (default 64).")
    p.add_argument("--stft_root", type=str, default=None,
                   help="Path to STFT dataset root directory.")
    p.add_argument("--cwt_root", type=str, default=None,
                   help="Path to CWT dataset root directory.")
    p.add_argument("--mel_root", type=str, default=None,
                   help="Path to MEL dataset root directory.")
    p.add_argument("--outputs_root", type=str, default=None,
                   help="Path to root output directory.")
    p.add_argument("--verbose", type=int, default=0,
                   help="Verbosity: 0=silent, 1=epoch losses.")
    return p.parse_args()


def load_modality(modality: str, cfg: ExperimentConfig, img_size: int):
    module_name, fn_name, root_attr = MODALITY_LOADERS[modality]
    import importlib
    mod = importlib.import_module(module_name)
    loader_fn = getattr(mod, fn_name)
    root = getattr(cfg, root_attr)
    logger.info("Loading %s from %s …", modality.upper(), root)
    return loader_fn(root=root, img_size=img_size)


def encode_labels(labels_raw: np.ndarray):
    labels_raw = labels_raw.astype(str)
    enc = LabelEncoder()
    y = enc.fit_transform(labels_raw)
    return y, enc


def select_models(cfg: ExperimentConfig, mode: str, model_filter: str = None) -> list:
    if model_filter:
        return [model_filter]

    if mode == "ablation":
        return cfg.ablation_ccnn_models + cfg.ablation_crnn_models
    elif mode == "new":
        return cfg.new_models
    elif mode == "models":
        return cfg.new_models + cfg.reference_models
    else:  # "all"
        return cfg.models_to_run


def run_modality(
    modality: str,
    cfg: ExperimentConfig,
    smoke: bool,
    img_size: int,
    mode: str,
    model_filter=None,
) -> list:
    data, labels_raw, patients = load_modality(modality, cfg, img_size)
    y, encoder = encode_labels(labels_raw)
    num_classes = len(encoder.classes_)
    y_one_hot   = np.eye(num_classes, dtype=np.float32)[y]
    class_names = [str(c) for c in encoder.classes_]

    logger.info(
        "%s | %d samples | %d classes: %s",
        modality.upper(), len(data), num_classes, class_names,
    )

    modality_out_dir = os.path.join(cfg.outputs_root, modality)
    modality_metrics_dir = os.path.join(modality_out_dir, "metrics")
    modality_plots_dir = os.path.join(modality_out_dir, "plots")
    ensure_dir(modality_metrics_dir)
    ensure_dir(modality_plots_dir)

    save_json(
        os.path.join(modality_metrics_dir, "dataset_summary.json"),
        {
            "modality":    modality,
            "num_samples": int(len(data)),
            "num_classes": num_classes,
            "class_names": class_names,
            "class_counts": {c: int(np.sum(labels_raw == c)) for c in class_names},
            "num_patients":  int(len(set(patients.tolist()))),
        },
    )

    models = select_models(cfg, mode, model_filter)
    logger.info("Selected models to run for %s (%s mode): %s", modality.upper(), mode, models)

    all_summaries = []
    for model_name in models:
        logger.info("=" * 60)
        logger.info("Modality: %s | Model: %s", modality.upper(), model_name)
        logger.info("=" * 60)
        try:
            summary = run_nested_cv(
                model_name=model_name,
                data=data,
                y=y,
                y_one_hot=y_one_hot,
                class_names=class_names,
                patients=patients,
                cfg=cfg,
                modality=modality,
                smoke=smoke,
            )
            summary["status"] = "ok"
        except Exception as exc:
            logger.exception("FAILED – %s | %s: %s", modality, model_name, exc)
            summary = {"model": model_name, "modality": modality,
                       "status": "failed", "error": str(exc)}
            save_json(
                os.path.join(modality_metrics_dir, model_name, "failure.json"),
                summary,
            )
        all_summaries.append(summary)

    ok_summaries = [s for s in all_summaries if s.get("status") == "ok"]

    for metric_key, ylabel in [
        ("test_accuracy",    "Accuracy"),
        ("test_f1_macro",    "F1 (macro)"),
        ("test_auc_macro",   "AUC (macro)"),
        ("test_mean_fpr",    "Mean FPR"),
    ]:
        if not ok_summaries:
            break
        records = []
        for s in ok_summaries:
            ci = s.get(metric_key, {})
            if isinstance(ci, dict):
                records.append({"model": s["model"], **ci})
        if records:
            plot_metric_comparison(
                summary_records=records,
                metric_key=metric_key,
                title=f"{modality.upper()} – {ylabel} across models (95% CI)",
                out_path=os.path.join(modality_plots_dir, f"comparison_{metric_key}.png"),
                ylabel=ylabel,
            )

    if ok_summaries:
        heatmap_records = []
        for s in ok_summaries:
            row = {"model": s["model"]}
            for k in ["test_accuracy", "test_f1_macro", "test_auc_macro",
                      "test_precision_macro", "test_recall_macro"]:
                ci = s.get(k, {})
                row[k] = ci.get("mean", float("nan")) if isinstance(ci, dict) else float("nan")
            heatmap_records.append(row)
        plot_summary_heatmap(
            records=heatmap_records,
            metric_keys=["test_accuracy", "test_f1_macro", "test_auc_macro",
                         "test_precision_macro", "test_recall_macro"],
            title=f"{modality.upper()} – Model comparison heatmap",
            out_path=os.path.join(modality_plots_dir, "summary_heatmap.png"),
        )

    pd.DataFrame(all_summaries).to_csv(
        os.path.join(modality_metrics_dir, "all_models_summary.csv"), index=False
    )

    completed_models = [s["model"] for s in ok_summaries]
    if len(completed_models) >= 2:
        logger.info("Running statistical analysis for %s …", modality.upper())
        ci_results = {}
        for s in ok_summaries:
            ci = s.get("test_f1_macro", {})
            if isinstance(ci, dict):
                ci_results[s["model"]] = ci
        try:
            run_statistical_analysis(
                modality=modality,
                model_names=completed_models,
                metrics_dir=modality_metrics_dir,
                plots_dir=modality_plots_dir,
                metric_col="test_f1_macro",
                ci_results=ci_results,
            )
        except Exception as exc:
            logger.warning("Statistical analysis failed for %s: %s", modality, exc)

    return all_summaries


def main() -> None:
    args = parse_args()
    cfg  = ExperimentConfig()

    if args.outer is not None:
        cfg.outer_folds = args.outer
    if args.inner is not None:
        cfg.inner_folds = args.inner
    if args.stft_root is not None:
        cfg.stft_root = args.stft_root
    if args.cwt_root is not None:
        cfg.cwt_root = args.cwt_root
    if args.mel_root is not None:
        cfg.mel_root = args.mel_root
    if args.outputs_root is not None:
        cfg.outputs_root = args.outputs_root

    cfg.verbose = args.verbose

    set_global_seed(cfg.seed)

    modalities = (
        ["stft", "cwt", "mel"] if args.dataset == "all" else [args.dataset]
    )

    global_start = time.time()
    all_results  = {}

    for modality in modalities:
        logger.info("▶ Starting modality: %s", modality.upper())
        summaries = run_modality(
            modality=modality,
            cfg=cfg,
            smoke=args.smoke,
            img_size=args.img_size,
            mode=args.mode,
            model_filter=args.model,
        )
        all_results[modality] = summaries

    total_time = time.time() - global_start
    logger.info("✓ All runs complete. Total wall time: %.1f s (%.1f min)",
                total_time, total_time / 60.0)

    save_json(
        os.path.join(cfg.outputs_root, "global_summary.json"),
        {"total_runtime_s": total_time, "results": all_results},
    )

    # Generate summary CSV and Excel files
    logger.info("Generating output CSV & Excel summary files...")
    generate_all_csvs()

    # Generate/update per-modality Jupyter notebooks
    logger.info("Generating Jupyter notebooks (ablation_test.ipynb & view_*_output.ipynb)...")
    generate_all_notebooks(cfg.outputs_root)
    logger.info("✓ All CSVs and notebooks generated successfully.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
