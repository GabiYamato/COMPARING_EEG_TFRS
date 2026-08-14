"""
csv_generator.py
================
Generates:
1. <spectrogram>_results_expanded_summary.csv & .xlsx
   - Reordered columns: [Metric, Std, 95% CI] side-by-side
   - Includes timing metrics: Training Time (s), Test Time (s), Avg Time/Sample (ms)
   - Calculates exact fold and aggregate metrics, CIs, and Wilcoxon statistical test comparisons
2. <spectrogram>_ablation_study.csv & .xlsx
   - Contains cCNN and cRNN ablation study results with full metrics:
     Accuracy, Δ Acc, F1 Macro, Δ F1, AUC Macro, Δ AUC, Precision Macro, Δ Prec,
     Recall Macro, Δ Rec, FPR Mean, Δ FPR, Training Time, Test Time, Avg Time/Sample
"""

import json
import os
import glob
import re
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
VIEW_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "view_outputs")
RYAN_CODE_RUNS_DIR = os.path.join(PROJECT_ROOT, "RYAN_CODE_RUNS")

MODALITIES = ["mel", "stft", "cwt"]

NON_ABLATION_MODELS = ["AlexNet", "cCNN", "cRNN", "LSTM", "RNN", "VGGNET", "CNN_LSTM", "CNN_BiLSTM", "Swin_Transformer", "ResBiLSTM"]

CCNN_ABLATIONS = [
    ("Baseline", "cCNN", "Original cCNN", "Reference Baseline"),
    ("A1", "cCNN_A1_NoPooling", "Remove MaxPooling", "Importance of pooling"),
    ("A2", "cCNN_A2_NoDropout", "Remove Dropout", "Regularization effect"),
    ("A3", "cCNN_A3_Dense64", "Dense=64 instead of 128", "Dense layer size"),
    ("A4", "cCNN_A4_Filters16", "Conv Filters=16", "Model capacity (reduced)"),
    ("A5", "cCNN_A5_Filters64", "Conv Filters=64", "Model capacity (expanded)"),
    ("A6", "cCNN_A6_Kernel5", "Kernel 5x5 instead of 3x3", "Kernel influence"),
]

CRNN_ABLATIONS = [
    ("Baseline", "cRNN", "Original cRNN", "Reference Baseline"),
    ("B1", "cRNN_B1_OneLayer", "One RNN layer only", "Recurrent depth (1 layer)"),
    ("B2", "cRNN_B2_TwoLayers", "Two RNN layers", "Recurrent depth (2 layers)"),
    ("B3", "cRNN_B3_NoDropout", "No Dropout", "Regularization effect"),
    ("B4", "cRNN_B4_Hidden1024", "1024 units instead of 256", "Recurrent capacity (1024)"),
    ("B5", "cRNN_B5_Hidden512", "512 units", "Recurrent capacity (512)"),
    ("B6", "cRNN_B6_GRU", "Replace SimpleRNN with GRU", "Gating mechanism"),
]


def load_previous_expanded_summary(modality: str) -> pd.DataFrame:
    """Load previous results from view_outputs if existing."""
    if modality == "stft":
        path = os.path.join(VIEW_OUTPUTS_DIR, "results_expanded_summary_generated.csv")
    else:
        path = os.path.join(VIEW_OUTPUTS_DIR, f"{modality}_results_expanded_summary_generated.csv")
        
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"[{modality.upper()}] Loaded previous summary from {path} ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"[{modality.upper()}] Could not load {path}: {e}")
    return pd.DataFrame()


def format_ci_str(ci_dict: dict) -> str:
    if not isinstance(ci_dict, dict):
        return "[N/A, N/A]"
    low = ci_dict.get("ci_low_bootstrap", ci_dict.get("mean", 0.0))
    high = ci_dict.get("ci_high_bootstrap", ci_dict.get("mean", 0.0))
    return f"[{low:.4f}, {high:.4f}]"


def get_legacy_model_timing(modality: str, model_name: str) -> dict:
    """Lookup legacy model timing info from RYAN_CODE_RUNS & view_outputs notebooks."""
    mod_upper = modality.upper()
    paths_to_check = [
        os.path.join(RYAN_CODE_RUNS_DIR, "outputs", "metrics", modality, model_name),
        os.path.join(RYAN_CODE_RUNS_DIR, "outputs", "metrics", mod_upper, model_name),
        os.path.join(RYAN_CODE_RUNS_DIR, "results", mod_upper, model_name),
        os.path.join(RYAN_CODE_RUNS_DIR, "results", f"{mod_upper}_SMOKE", model_name),
    ]

    total_runtime = None
    test_time = None
    
    for p in paths_to_check:
        ci_file = os.path.join(p, "summary_ci.json")
        ci_with_file = os.path.join(p, "summary_with_ci.json")
        
        if os.path.exists(ci_file):
            with open(ci_file) as f:
                d = json.load(f)
            total_runtime = d.get("runtime_s", d.get("runtime_seconds"))
            tt = d.get("execution_time_s")
            test_time = tt.get("mean") if isinstance(tt, dict) else tt
            break
        elif os.path.exists(ci_with_file):
            with open(ci_with_file) as f:
                d = json.load(f)
            total_runtime = d.get("runtime_seconds", d.get("runtime_s"))
            tt = d.get("execution_time_s")
            test_time = tt.get("mean") if isinstance(tt, dict) else tt
            break

    if test_time is None:
        nb_path = os.path.join(VIEW_OUTPUTS_DIR, f"{modality}.ipynb")
        if os.path.exists(nb_path):
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    nb = json.load(f)
                model_found = False
                times = []
                for cell in nb.get("cells", []):
                    for out in cell.get("outputs", []):
                        text = "".join(out.get("text", []))
                        if f"MODEL: {model_name}" in text:
                            model_found = True
                        if model_found:
                            html_list = out.get("data", {}).get("text/html", [])
                            if html_list:
                                html_str = "".join(html_list)
                                if "execution_time_s" in html_str:
                                    soup_rows = re.findall(r"<tr>(.*?)</tr>", html_str, re.DOTALL)
                                    for r in soup_rows:
                                        tds = re.findall(r"<td.*?>(.*?)</td>", r)
                                        if tds and len(tds) >= 10:
                                            try:
                                                times.append(float(tds[-1]))
                                            except ValueError:
                                                pass
                if times:
                    test_time = float(np.mean(times))
            except Exception:
                pass

    train_time_per_fold = (total_runtime / 10.0) if total_runtime is not None else None
    avg_per_sample_ms = (test_time / 450.0 * 1000.0) if test_time is not None else None

    return {
        "total_runtime_s": total_runtime,
        "train_time_per_fold_s": train_time_per_fold,
        "test_time_s": test_time,
        "avg_time_per_sample_ms": avg_per_sample_ms,
    }


def compute_wilcoxon_tests(modality_metrics_dir: str, non_ablation_models: list) -> dict:
    """Compute pairwise Wilcoxon tests between non-ablation models."""
    model_scores = {}
    for model in non_ablation_models:
        csv_path = os.path.join(modality_metrics_dir, model, "fold_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "test_f1_macro" in df.columns:
                model_scores[model] = df["test_f1_macro"].dropna().tolist()

    wilcoxon_notes = {}
    models = list(model_scores.keys())
    for m1 in models:
        sig_diff_vs = []
        for m2 in models:
            if m1 == m2:
                continue
            s1 = model_scores[m1]
            s2 = model_scores[m2]
            n = min(len(s1), len(s2))
            if n >= 3 and not np.allclose(s1[:n], s2[:n]):
                try:
                    stat, p = sp_stats.wilcoxon(s1[:n], s2[:n])
                    if p < 0.05:
                        sig_diff_vs.append(m2)
                except Exception:
                    pass
        if sig_diff_vs:
            wilcoxon_notes[m1] = f"Sig. diff vs: {', '.join(sorted(sig_diff_vs))}"
        else:
            wilcoxon_notes[m1] = "No sig. diff vs evaluated models"

    return wilcoxon_notes


def generate_expanded_summary_csv(modality: str):
    """Generates <modality>_results_expanded_summary.csv and .xlsx."""
    mod_dir = os.path.join(BASE_DIR, modality)
    metrics_dir = os.path.join(mod_dir, "metrics")
    out_csv_path = os.path.join(mod_dir, f"{modality}_results_expanded_summary.csv")
    out_xlsx_path = os.path.join(mod_dir, f"{modality}_results_expanded_summary.xlsx")
    view_out_csv_path = os.path.join(VIEW_OUTPUTS_DIR, f"{modality}_results_expanded_summary.csv")
    view_out_xlsx_path = os.path.join(VIEW_OUTPUTS_DIR, f"{modality}_results_expanded_summary.xlsx")

    prev_df = load_previous_expanded_summary(modality)

    new_rows = []
    
    if os.path.exists(metrics_dir):
        available_models = [d for d in os.listdir(metrics_dir) 
                            if os.path.isdir(os.path.join(metrics_dir, d)) and d in NON_ABLATION_MODELS]
        
        wilcoxon_notes = compute_wilcoxon_tests(metrics_dir, available_models)
        
        for model in available_models:
            model_dir = os.path.join(metrics_dir, model)
            fold_csv = os.path.join(model_dir, "fold_metrics.csv")
            summary_json = os.path.join(model_dir, "summary_ci.json")
            
            if not (os.path.exists(fold_csv) and os.path.exists(summary_json)):
                continue
                
            df_fold = pd.read_csv(fold_csv)
            with open(summary_json, "r") as f:
                summary_ci = json.load(f)
                
            wilcox_str = wilcoxon_notes.get(model, "No sig. diff vs evaluated models")
            
            for idx, row in df_fold.iterrows():
                fold_num = int(row.get("fold", idx + 1))
                fold_name = f"fold_{fold_num:02d}"
                
                fold_json_path = os.path.join(model_dir, fold_name, "test_metrics.json")
                cm_str = "[]"
                if os.path.exists(fold_json_path):
                    try:
                        with open(fold_json_path, "r") as f_json:
                            tm = json.load(f_json)
                            cm_str = str(tm.get("confusion_matrix", []))
                    except Exception:
                        pass
                
                acc_ci = summary_ci.get("test_accuracy", {})
                f1_ci = summary_ci.get("test_f1_macro", {})
                prec_ci = summary_ci.get("test_precision_macro", {})
                rec_ci = summary_ci.get("test_recall_macro", {})
                fpr_ci = summary_ci.get("test_mean_fpr", {})
                auc_ci = summary_ci.get("test_auc_macro", {})
                
                tr_time_fold = row.get("training_time_s", float("nan"))
                te_time_fold = row.get("test_time_s", float("nan"))
                avg_samp_fold = row.get("avg_time_per_sample_s", float("nan"))
                if not np.isnan(avg_samp_fold):
                    avg_samp_fold_ms = avg_samp_fold * 1000.0
                else:
                    avg_samp_fold_ms = float("nan")

                new_rows.append({
                    "Model": model,
                    "Fold": fold_name,
                    
                    # Accuracy & 95% CI
                    "Test Accuracy": row.get("test_accuracy", float("nan")),
                    "Accuracy Std (Model)": round(acc_ci.get("std", 0.0), 4),
                    "Accuracy 95% CI (Model)": format_ci_str(acc_ci),

                    # F1 Score (Macro) & 95% CI
                    "F1 Score (Macro)": row.get("test_f1_macro", float("nan")),
                    "F1 Std (Model)": round(f1_ci.get("std", 0.0), 4),
                    "F1 95% CI (Model)": format_ci_str(f1_ci),

                    # Precision (Macro) & 95% CI
                    "Precision (Macro)": row.get("test_precision_macro", float("nan")),
                    "Precision Std (Model)": round(prec_ci.get("std", 0.0), 4),
                    "Precision 95% CI (Model)": format_ci_str(prec_ci),

                    # Recall (Macro) & 95% CI
                    "Recall (Macro)": row.get("test_recall_macro", float("nan")),
                    "Recall Std (Model)": round(rec_ci.get("std", 0.0), 4),
                    "Recall 95% CI (Model)": format_ci_str(rec_ci),

                    # FPR (Mean) & 95% CI
                    "FPR (Mean)": row.get("test_mean_fpr", float("nan")),
                    "FPR Std (Model)": round(fpr_ci.get("std", 0.0), 4),
                    "FPR 95% CI (Model)": format_ci_str(fpr_ci),

                    # AUC (Macro) & 95% CI
                    "AUC (Macro)": row.get("test_auc_macro", float("nan")),
                    "AUC Std (Model)": round(auc_ci.get("std", 0.0), 4),
                    "AUC 95% CI (Model)": format_ci_str(auc_ci),

                    # Timing Metrics
                    "Training Time (s)": round(tr_time_fold, 2) if not np.isnan(tr_time_fold) else "N/A",
                    "Test Time (s)": round(te_time_fold, 4) if not np.isnan(te_time_fold) else "N/A",
                    "Avg Time/Sample (ms)": round(avg_samp_fold_ms, 3) if not np.isnan(avg_samp_fold_ms) else "N/A",

                    "Confusion Matrix": cm_str,
                    "Statistical Tests (Wilcoxon)": wilcox_str,
                })

    df_new = pd.DataFrame(new_rows)

    if not prev_df.empty:
        newly_computed_models = set(df_new["Model"].unique()) if not df_new.empty else set()
        prev_retained = prev_df[~prev_df["Model"].isin(newly_computed_models)].copy()

        cols_to_drop = [c for c in prev_retained.columns if "Mean (Model)" in c or "SE (Model)" in c]
        prev_retained = prev_retained.drop(columns=cols_to_drop, errors="ignore")

        for model_name in prev_retained["Model"].unique():
            timing = get_legacy_model_timing(modality, model_name)
            mask = (prev_retained["Model"] == model_name)
            
            tr_fold = timing["train_time_per_fold_s"]
            te_time = timing["test_time_s"]
            avg_samp_ms = timing["avg_time_per_sample_ms"]
            
            prev_retained.loc[mask, "Training Time (s)"] = round(tr_fold, 2) if tr_fold is not None else "N/A"
            prev_retained.loc[mask, "Test Time (s)"] = round(te_time, 4) if te_time is not None else "N/A"
            prev_retained.loc[mask, "Avg Time/Sample (ms)"] = round(avg_samp_ms, 3) if avg_samp_ms is not None else "N/A"

        target_columns = [
            "Model", "Fold",
            "Test Accuracy", "Accuracy Std (Model)", "Accuracy 95% CI (Model)",
            "F1 Score (Macro)", "F1 Std (Model)", "F1 95% CI (Model)",
            "Precision (Macro)", "Precision Std (Model)", "Precision 95% CI (Model)",
            "Recall (Macro)", "Recall Std (Model)", "Recall 95% CI (Model)",
            "FPR (Mean)", "FPR Std (Model)", "FPR 95% CI (Model)",
            "AUC (Macro)", "AUC Std (Model)", "AUC 95% CI (Model)",
            "Training Time (s)", "Test Time (s)", "Avg Time/Sample (ms)",
            "Confusion Matrix", "Statistical Tests (Wilcoxon)"
        ]

        prev_retained = prev_retained.reindex(columns=target_columns)
        if not df_new.empty:
            df_new = df_new.reindex(columns=target_columns)

        combined_df = pd.concat([prev_retained, df_new], ignore_index=True)
    else:
        combined_df = df_new

    if not combined_df.empty:
        combined_df.to_csv(out_csv_path, index=False)
        combined_df.to_excel(out_xlsx_path, index=False)
        combined_df.to_csv(view_out_csv_path, index=False)
        combined_df.to_excel(view_out_xlsx_path, index=False)
        print(f"[{modality.upper()}] Wrote expanded summary CSV & Excel to:\n  - {out_csv_path}\n  - {out_xlsx_path}")
    else:
        print(f"[{modality.upper()}] No expanded summary data available to write.")


def generate_ablation_study_csv(modality: str):
    """Generates <modality>_ablation_study.csv and .xlsx with all metrics (Acc, F1, AUC, Precision, Recall, FPR, Timings)."""
    mod_dir = os.path.join(BASE_DIR, modality)
    metrics_dir = os.path.join(mod_dir, "metrics")
    out_csv_path = os.path.join(mod_dir, f"{modality}_ablation_study.csv")
    out_xlsx_path = os.path.join(mod_dir, f"{modality}_ablation_study.xlsx")
    view_out_csv_path = os.path.join(VIEW_OUTPUTS_DIR, f"{modality}_ablation_study.csv")
    view_out_xlsx_path = os.path.join(VIEW_OUTPUTS_DIR, f"{modality}_ablation_study.xlsx")

    rows = []

    # 1. cCNN Ablations
    baseline_acc_ccnn = None
    baseline_f1_ccnn = None
    baseline_auc_ccnn = None
    baseline_prec_ccnn = None
    baseline_rec_ccnn = None
    baseline_fpr_ccnn = None
    
    for label, model_name, desc, purpose in CCNN_ABLATIONS:
        ci_path = os.path.join(metrics_dir, model_name, "summary_ci.json")
        if os.path.exists(ci_path):
            with open(ci_path, "r") as f:
                d = json.load(f)
            acc = d.get("test_accuracy", {}).get("mean", 0.0) * 100
            f1 = d.get("test_f1_macro", {}).get("mean", 0.0) * 100
            auc_val = d.get("test_auc_macro", {}).get("mean", 0.0) * 100
            prec = d.get("test_precision_macro", {}).get("mean", 0.0) * 100
            rec = d.get("test_recall_macro", {}).get("mean", 0.0) * 100
            fpr_val = d.get("test_mean_fpr", {}).get("mean", 0.0) * 100
            
            tr_time = d.get("training_time_s", {}).get("mean", float("nan"))
            te_time = d.get("test_time_s", {}).get("mean", float("nan"))
            avg_samp_time = d.get("avg_time_per_sample_s", {}).get("mean", float("nan")) * 1000 # ms

            if label == "Baseline" or model_name == "cCNN":
                baseline_acc_ccnn = acc
                baseline_f1_ccnn = f1
                baseline_auc_ccnn = auc_val
                baseline_prec_ccnn = prec
                baseline_rec_ccnn = rec
                baseline_fpr_ccnn = fpr_val
                d_acc = d_f1 = d_auc = d_prec = d_rec = d_fpr = 0.0
            else:
                d_acc  = acc - baseline_acc_ccnn if baseline_acc_ccnn is not None else 0.0
                d_f1   = f1 - baseline_f1_ccnn if baseline_f1_ccnn is not None else 0.0
                d_auc  = auc_val - baseline_auc_ccnn if baseline_auc_ccnn is not None else 0.0
                d_prec = prec - baseline_prec_ccnn if baseline_prec_ccnn is not None else 0.0
                d_rec  = rec - baseline_rec_ccnn if baseline_rec_ccnn is not None else 0.0
                d_fpr  = fpr_val - baseline_fpr_ccnn if baseline_fpr_ccnn is not None else 0.0

            rows.append({
                "Category": "cCNN Ablation",
                "Model ID": label,
                "Model Name": model_name,
                "Modification": desc,
                "Purpose": purpose,
                
                "Accuracy (%)": round(acc, 2),
                "Δ Accuracy (%)": round(d_acc, 2),
                
                "F1 Macro (%)": round(f1, 2),
                "Δ F1 Macro (%)": round(d_f1, 2),

                "AUC Macro (%)": round(auc_val, 2),
                "Δ AUC Macro (%)": round(d_auc, 2),
                
                "Precision Macro (%)": round(prec, 2),
                "Δ Precision Macro (%)": round(d_prec, 2),

                "Recall Macro (%)": round(rec, 2),
                "Δ Recall Macro (%)": round(d_rec, 2),

                "FPR Mean (%)": round(fpr_val, 2),
                "Δ FPR Mean (%)": round(d_fpr, 2),

                "Training Time (s)": round(tr_time, 2) if not np.isnan(tr_time) else "N/A",
                "Test Time (s)": round(te_time, 4) if not np.isnan(te_time) else "N/A",
                "Avg Time/Sample (ms)": round(avg_samp_time, 3) if not np.isnan(avg_samp_time) else "N/A",
            })

    # 2. cRNN Ablations
    baseline_acc_crnn = None
    baseline_f1_crnn = None
    baseline_auc_crnn = None
    baseline_prec_crnn = None
    baseline_rec_crnn = None
    baseline_fpr_crnn = None

    for label, model_name, desc, purpose in CRNN_ABLATIONS:
        ci_path = os.path.join(metrics_dir, model_name, "summary_ci.json")
        if os.path.exists(ci_path):
            with open(ci_path, "r") as f:
                d = json.load(f)
            acc = d.get("test_accuracy", {}).get("mean", 0.0) * 100
            f1 = d.get("test_f1_macro", {}).get("mean", 0.0) * 100
            auc_val = d.get("test_auc_macro", {}).get("mean", 0.0) * 100
            prec = d.get("test_precision_macro", {}).get("mean", 0.0) * 100
            rec = d.get("test_recall_macro", {}).get("mean", 0.0) * 100
            fpr_val = d.get("test_mean_fpr", {}).get("mean", 0.0) * 100
            
            tr_time = d.get("training_time_s", {}).get("mean", float("nan"))
            te_time = d.get("test_time_s", {}).get("mean", float("nan"))
            avg_samp_time = d.get("avg_time_per_sample_s", {}).get("mean", float("nan")) * 1000 # ms

            if label == "Baseline" or model_name == "cRNN":
                baseline_acc_crnn = acc
                baseline_f1_crnn = f1
                baseline_auc_crnn = auc_val
                baseline_prec_crnn = prec
                baseline_rec_crnn = rec
                baseline_fpr_crnn = fpr_val
                d_acc = d_f1 = d_auc = d_prec = d_rec = d_fpr = 0.0
            else:
                d_acc  = acc - baseline_acc_crnn if baseline_acc_crnn is not None else 0.0
                d_f1   = f1 - baseline_f1_crnn if baseline_f1_crnn is not None else 0.0
                d_auc  = auc_val - baseline_auc_crnn if baseline_auc_crnn is not None else 0.0
                d_prec = prec - baseline_prec_crnn if baseline_prec_crnn is not None else 0.0
                d_rec  = rec - baseline_rec_crnn if baseline_rec_crnn is not None else 0.0
                d_fpr  = fpr_val - baseline_fpr_crnn if baseline_fpr_crnn is not None else 0.0

            rows.append({
                "Category": "cRNN Ablation",
                "Model ID": label,
                "Model Name": model_name,
                "Modification": desc,
                "Purpose": purpose,
                
                "Accuracy (%)": round(acc, 2),
                "Δ Accuracy (%)": round(d_acc, 2),
                
                "F1 Macro (%)": round(f1, 2),
                "Δ F1 Macro (%)": round(d_f1, 2),

                "AUC Macro (%)": round(auc_val, 2),
                "Δ AUC Macro (%)": round(d_auc, 2),
                
                "Precision Macro (%)": round(prec, 2),
                "Δ Precision Macro (%)": round(d_prec, 2),

                "Recall Macro (%)": round(rec, 2),
                "Δ Recall Macro (%)": round(d_rec, 2),

                "FPR Mean (%)": round(fpr_val, 2),
                "Δ FPR Mean (%)": round(d_fpr, 2),

                "Training Time (s)": round(tr_time, 2) if not np.isnan(tr_time) else "N/A",
                "Test Time (s)": round(te_time, 4) if not np.isnan(te_time) else "N/A",
                "Avg Time/Sample (ms)": round(avg_samp_time, 3) if not np.isnan(avg_samp_time) else "N/A",
            })

    if rows:
        df_abl = pd.DataFrame(rows)
        df_abl.to_csv(out_csv_path, index=False)
        df_abl.to_excel(out_xlsx_path, index=False)
        df_abl.to_csv(view_out_csv_path, index=False)
        df_abl.to_excel(view_out_xlsx_path, index=False)
        print(f"[{modality.upper()}] Wrote ablation study CSV & Excel to:\n  - {out_csv_path}\n  - {out_xlsx_path}")
    else:
        print(f"[{modality.upper()}] No ablation data found yet.")


def generate_all_csvs():
    os.makedirs(VIEW_OUTPUTS_DIR, exist_ok=True)
    for modality in MODALITIES:
        generate_expanded_summary_csv(modality)
        generate_ablation_study_csv(modality)


if __name__ == "__main__":
    generate_all_csvs()
