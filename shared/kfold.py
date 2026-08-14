"""
shared/kfold.py
===============
10-outer / 5-inner Stratified Nested Cross-Validation engine.
Enhanced with training_time, test_time, and avg_time_per_sample tracking.
"""

import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score

from shared.models import build_model, DEVICE
from shared.trainer import fit, lazy_init, predict_batched
from shared.metrics import compute_metrics, bootstrap_ci
from shared.transforms import prepare_inputs, standardize
from shared.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)
from shared.utils import ensure_dir, save_json, get_logger, free_memory

logger = get_logger(__name__)


def _write_fold_composition(
    patients: np.ndarray,
    labels_decoded: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out_csv: str,
) -> None:
    df = pd.DataFrame({
        "idx":     np.arange(len(labels_decoded)),
        "patient": patients.astype(str),
        "label":   labels_decoded.astype(str),
        "split":   "unused",
    })
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx,  "split"] = "test"
    summary = (
        df[df["split"].isin(["train", "test"])]
        .groupby(["split", "patient", "label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    summary.to_csv(out_csv, index=False)


def _to_tensor(x_np: np.ndarray) -> torch.Tensor:
    return torch.tensor(x_np.astype(np.float32))


def _downsample_to_minority(
    x: np.ndarray,
    y: np.ndarray,
    y_oh: np.ndarray,
    factor: int = 1,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_count  = max(1, int(counts.min()) // factor)

    x_parts    = []
    y_parts    = []
    y_oh_parts = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) > target_count:
            idx = rng.choice(idx, size=target_count, replace=False)
        x_parts.append(x[idx])
        y_parts.append(y[idx])
        y_oh_parts.append(y_oh[idx])

    x_bal    = np.concatenate(x_parts, axis=0)
    y_bal    = np.concatenate(y_parts, axis=0)
    y_oh_bal = np.concatenate(y_oh_parts, axis=0)
    perm = rng.permutation(len(y_bal))
    return x_bal[perm], y_bal[perm], y_oh_bal[perm]


def run_nested_cv(
    model_name: str,
    data: np.ndarray,
    y: np.ndarray,
    y_one_hot: np.ndarray,
    class_names: List[str],
    patients: np.ndarray,
    cfg,
    modality: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    outer_folds = cfg.smoke_outer_folds if smoke else cfg.outer_folds
    inner_folds = cfg.smoke_inner_folds if smoke else cfg.inner_folds

    # Output directories - point to modality/outputs
    modality_out_dir = os.path.join(cfg.outputs_root, modality)
    ckpt_dir    = os.path.join(modality_out_dir, "checkpoints", model_name)
    metrics_dir = os.path.join(modality_out_dir, "metrics",     model_name)
    plots_dir   = os.path.join(modality_out_dir, "plots",       model_name)
    for d in [ckpt_dir, metrics_dir, plots_dir]:
        ensure_dir(d)

    prep = prepare_inputs(data, model_name, volume_size=cfg.cnn3d_volume_size)
    x_full = prep["x"]

    num_classes = len(class_names)
    time_steps  = prep.get("time_steps", 32)
    input_dim   = prep.get("input_dim",  32)

    param_grid = list(ParameterGrid({
        "lr":         cfg.lr_values,
        "batch_size": cfg.batch_values,
        "patience":   cfg.patience_values,
    }))[: cfg.max_trials_per_model]

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True,
                                random_state=cfg.seed)

    fold_rows:        List[Dict] = []
    fold_test_metrics: List[Dict] = []
    global_start = time.time()

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(x_full, y), start=1
    ):
        logger.info("[%s | %s] Outer fold %d/%d", modality, model_name, fold_id, outer_folds)
        fold_ckpt_dir    = os.path.join(ckpt_dir,    f"fold_{fold_id:02d}")
        fold_metrics_dir = os.path.join(metrics_dir, f"fold_{fold_id:02d}")
        fold_plots_dir   = os.path.join(plots_dir,   f"fold_{fold_id:02d}")
        for d in [fold_ckpt_dir, fold_metrics_dir, fold_plots_dir]:
            ensure_dir(d)

        _write_fold_composition(
            patients=patients,
            labels_decoded=np.array(class_names)[y],
            train_idx=outer_train_idx,
            test_idx=outer_test_idx,
            out_csv=os.path.join(fold_metrics_dir, "fold_composition.csv"),
        )

        x_outer_train = x_full[outer_train_idx]
        y_outer_train = y[outer_train_idx]
        y_oh_train    = y_one_hot[outer_train_idx]
        x_outer_test  = x_full[outer_test_idx]
        y_oh_test     = y_one_hot[outer_test_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg.seed + fold_id)
        fit_ix, val_ix = next(sss.split(x_outer_train, y_outer_train))

        x_fit_bal, y_fit_int_bal, y_fit_oh_bal = _downsample_to_minority(
            x_outer_train[fit_ix], y_outer_train[fit_ix], y_oh_train[fit_ix],
            factor=cfg.balance_factor, seed=cfg.seed,
        )

        x_fit_std, x_val_std = standardize(x_fit_bal, x_outer_train[val_ix])
        _, x_tr_full_std = standardize(x_fit_bal, x_outer_train)
        _, x_te_std = standardize(x_fit_bal, x_outer_test)

        x_fit_t = _to_tensor(x_fit_std)
        y_fit_t = _to_tensor(y_fit_oh_bal)
        x_val_t = _to_tensor(x_val_std)
        y_val_t = _to_tensor(y_oh_train[val_ix])
        x_te_t  = _to_tensor(x_te_std)
        x_tr_full_t = _to_tensor(x_tr_full_std)

        best_params = None
        best_val_score = -np.inf
        best_model_state = None
        best_history = None

        fold_train_start = time.time()

        for params in param_grid:
            model, optimizer = build_model(
                model_name, num_classes, time_steps, input_dim, params["lr"]
            )
            lazy_init(model, x_fit_t[:1].to(DEVICE))
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)

            history = fit(
                model, optimizer,
                x_fit_t, y_fit_t, x_val_t, y_val_t,
                epochs=cfg.epochs,
                batch_size=max(1, params["batch_size"]),
                patience=params["patience"],
                verbose=cfg.verbose,
            )

            y_vl_prob = predict_batched(model, x_val_t)
            y_vl_pred = np.argmax(y_vl_prob, axis=1)
            y_vl_true = np.argmax(y_oh_train[val_ix], axis=1)
            score = f1_score(y_vl_true, y_vl_pred, average="macro", zero_division=0)
            
            logger.info("  Params %s → val F1_macro=%.4f", params, score)
            if score > best_val_score:
                best_val_score = score
                best_params = params
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_history = history

            del model, optimizer
            free_memory()

        fold_train_time = time.time() - fold_train_start

        if best_params is None:
            raise RuntimeError(f"No valid params found – model={model_name}, fold={fold_id}")

        logger.info("  Best params: %s (val F1=%.4f, train time=%.2fs)", best_params, best_val_score, fold_train_time)
        save_json(os.path.join(fold_metrics_dir, "best_params.json"), best_params)

        # Evaluate Best Model on Test Set
        ckpt_path = os.path.join(fold_ckpt_dir, "best_model.pt")
        
        model, _ = build_model(model_name, num_classes, time_steps, input_dim, best_params["lr"])
        lazy_init(model, x_fit_t[:1].to(DEVICE))
        model.load_state_dict(best_model_state)
        
        torch.save({"model_state_dict": best_model_state, "history": best_history}, ckpt_path)
        history = best_history

        # Measure test inference timing
        test_start = time.time()
        y_te_prob = predict_batched(model, x_te_t)
        test_time = time.time() - test_start
        num_test_samples = len(x_te_t)
        avg_time_per_sample = test_time / max(1, num_test_samples)

        y_tr_prob = predict_batched(model, x_tr_full_t)

        y_tr_true = np.argmax(y_oh_train, axis=1)
        y_te_true = np.argmax(y_oh_test,  axis=1)

        train_metrics = compute_metrics(y_tr_true, y_tr_prob, class_names)
        test_metrics  = compute_metrics(y_te_true, y_te_prob, class_names)
        
        # Add timing metrics
        test_metrics["training_time_s"] = fold_train_time
        test_metrics["test_time_s"] = test_time
        test_metrics["avg_time_per_sample_s"] = avg_time_per_sample
        test_metrics["execution_time_s"] = test_time

        save_json(os.path.join(fold_metrics_dir, "train_metrics.json"), train_metrics)
        save_json(os.path.join(fold_metrics_dir, "test_metrics.json"),  test_metrics)
        save_json(os.path.join(fold_metrics_dir, "history.json"),       history)

        plot_confusion_matrix(
            train_metrics["confusion_matrix_normalized"], class_names,
            title=f"{model_name} Fold {fold_id} Train CM (norm)",
            out_path=os.path.join(fold_plots_dir, "cm_train_norm.png"),
        )
        plot_confusion_matrix(
            test_metrics["confusion_matrix_normalized"], class_names,
            title=f"{model_name} Fold {fold_id} Test CM (norm)",
            out_path=os.path.join(fold_plots_dir, "cm_test_norm.png"),
        )
        plot_roc_curves(
            fpr=test_metrics["fpr"], tpr=test_metrics["tpr"],
            auc_per_class=test_metrics["roc_auc_per_class"],
            title=f"{model_name} Fold {fold_id} ROC",
            out_path=os.path.join(fold_plots_dir, "roc_test.png"),
        )
        plot_training_history(
            history=history,
            title=f"{model_name} Fold {fold_id} Loss",
            out_path=os.path.join(fold_plots_dir, "training_loss.png"),
        )

        fold_rows.append({
            "fold":                  fold_id,
            "train_samples":         int(len(outer_train_idx)),
            "test_samples":          int(len(outer_test_idx)),
            "inner_best_f1_macro":   float(best_val_score),
            "best_lr":               float(best_params["lr"]),
            "best_batch_size":       int(best_params["batch_size"]),
            "best_patience":         int(best_params["patience"]),
            # Timing Metrics
            "training_time_s":       float(fold_train_time),
            "test_time_s":           float(test_time),
            "avg_time_per_sample_s": float(avg_time_per_sample),
            "execution_time_s":      float(test_time),
            # Train
            "train_accuracy":        float(train_metrics["accuracy"]),
            "train_f1_micro":        float(train_metrics["f1_micro"]),
            "train_f1_macro":        float(train_metrics["f1_macro"]),
            "train_f1_weighted":     float(train_metrics["f1_weighted"]),
            "train_precision_micro": float(train_metrics["precision_micro"]),
            "train_precision_macro": float(train_metrics["precision_macro"]),
            "train_prec_weighted":   float(train_metrics["precision_weighted"]),
            "train_recall_micro":    float(train_metrics["recall_micro"]),
            "train_recall_macro":    float(train_metrics["recall_macro"]),
            "train_recall_weighted": float(train_metrics["recall_weighted"]),
            "train_auc_micro":       float(train_metrics["roc_auc_micro"]),
            "train_auc_macro":       float(train_metrics["roc_auc_macro"]),
            "train_auc_weighted":    float(train_metrics["roc_auc_weighted"]),
            "train_mean_fpr":        float(train_metrics["mean_fpr"]),
            # Test
            "test_accuracy":         float(test_metrics["accuracy"]),
            "test_f1_micro":         float(test_metrics["f1_micro"]),
            "test_f1_macro":         float(test_metrics["f1_macro"]),
            "test_f1_weighted":      float(test_metrics["f1_weighted"]),
            "test_precision_micro":  float(test_metrics["precision_micro"]),
            "test_precision_macro":  float(test_metrics["precision_macro"]),
            "test_prec_weighted":    float(test_metrics["precision_weighted"]),
            "test_recall_micro":     float(test_metrics["recall_micro"]),
            "test_recall_macro":     float(test_metrics["recall_macro"]),
            "test_recall_weighted":  float(test_metrics["recall_weighted"]),
            "test_auc_micro":        float(test_metrics["roc_auc_micro"]),
            "test_auc_macro":        float(test_metrics["roc_auc_macro"]),
            "test_auc_weighted":     float(test_metrics["roc_auc_weighted"]),
            "test_mean_fpr":         float(test_metrics["mean_fpr"]),
        })
        fold_test_metrics.append(test_metrics)

        del model, x_tr_full_std, x_te_std
        free_memory()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(metrics_dir, "fold_metrics.csv"), index=False)

    n_boot = cfg.n_bootstrap
    seed   = cfg.seed

    def _ci(col): return bootstrap_ci(fold_df[col].tolist(), n_boot, seed)

    summary = {
        "model":           model_name,
        "modality":        modality,
        "runtime_s":       float(time.time() - global_start),
        "folds_completed": int(len(fold_df)),
        # Timing Metrics CIs
        "training_time_s":       _ci("training_time_s"),
        "test_time_s":           _ci("test_time_s"),
        "avg_time_per_sample_s": _ci("avg_time_per_sample_s"),
        "execution_time_s":      _ci("execution_time_s"),
        # Test
        "test_accuracy":         _ci("test_accuracy"),
        "test_f1_micro":         _ci("test_f1_micro"),
        "test_f1_macro":         _ci("test_f1_macro"),
        "test_f1_weighted":      _ci("test_f1_weighted"),
        "test_precision_micro":  _ci("test_precision_micro"),
        "test_precision_macro":  _ci("test_precision_macro"),
        "test_prec_weighted":    _ci("test_prec_weighted"),
        "test_recall_micro":     _ci("test_recall_micro"),
        "test_recall_macro":     _ci("test_recall_macro"),
        "test_recall_weighted":  _ci("test_recall_weighted"),
        "test_auc_micro":        _ci("test_auc_micro"),
        "test_auc_macro":        _ci("test_auc_macro"),
        "test_auc_weighted":     _ci("test_auc_weighted"),
        "test_mean_fpr":         _ci("test_mean_fpr"),
        # Train
        "train_accuracy":        _ci("train_accuracy"),
        "train_f1_micro":        _ci("train_f1_micro"),
        "train_f1_macro":        _ci("train_f1_macro"),
        "train_f1_weighted":     _ci("train_f1_weighted"),
        "train_auc_micro":       _ci("train_auc_micro"),
        "train_auc_macro":       _ci("train_auc_macro"),
        "train_auc_weighted":    _ci("train_auc_weighted"),
    }
    save_json(os.path.join(metrics_dir, "summary_ci.json"), summary)
    logger.info("[%s | %s] Done. Test Acc=%.3f±%.3f | Avg Train Time=%.2fs | Avg Test Time=%.4fs",
                modality, model_name,
                summary["test_accuracy"]["mean"],
                summary["test_accuracy"]["std"],
                summary["training_time_s"]["mean"],
                summary["test_time_s"]["mean"])
    return summary
