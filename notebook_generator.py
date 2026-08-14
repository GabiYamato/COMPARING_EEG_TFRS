"""
notebook_generator.py
===================
Generates per-modality ablation_test.ipynb and view_{mod}_output.ipynb notebooks,
displaying generated CSV summary tables.
"""

import json
import os

MODALITIES = ["mel", "stft", "cwt"]


def create_cell_markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    }


def create_cell_code(code_str: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_str.strip("\n").split("\n")]
    }


def make_notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def generate_ablation_notebook(modality: str, out_path: str):
    cells = []

    # 1. Title & Documentation Cell
    cells.append(create_cell_markdown(f"""# Ablation Study — {modality.upper()} Spectrograms

This notebook evaluates structural and regularization ablations for **cCNN** and **cRNN** models on the **{modality.upper()}** dataset.

### cCNN Ablation Experiments
| Model ID | Modification | Purpose |
|---|---|---|
| **cCNN (Baseline)** | Original cCNN (Conv3D 32 -> MaxPool2 -> Drop 0.5 -> Dense 128 -> Drop 0.5 -> Softmax) | Reference Baseline |
| **A1** | Remove MaxPooling3D | Assess spatial/temporal downsampling importance |
| **A2** | Remove Dropout layers | Regularization effect |
| **A3** | Dense=64 instead of 128 | Impact of classifier capacity |
| **A4** | Conv Filters=16 instead of 32 | Model capacity (reduced) |
| **A5** | Conv Filters=64 instead of 32 | Model capacity (expanded) |
| **A6** | Kernel 5×5×5 instead of 3×3×3 | Receptive field influence |

### cRNN Ablation Experiments
| Model ID | Modification | Purpose |
|---|---|---|
| **cRNN (Baseline)** | Original (4-Layer RNN, hidden=256, tanh -> Drop 0.3 -> Dense 128 -> Drop 0.3 -> Softmax) | Reference Baseline |
| **B1** | 1 RNN Layer only | Impact of recurrent depth |
| **B2** | 2 RNN Layers | Impact of recurrent depth |
| **B3** | No Dropout | Regularization effect |
| **B4** | 1024 units instead of 256 | Recurrent capacity (large) |
| **B5** | 512 units instead of 256 | Recurrent capacity (medium) |
| **B6** | Replace SimpleRNN with GRU | Impact of gating mechanism |
"""))

    # 2. Setup Code Cell
    cells.append(create_cell_code(f"""import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from IPython.display import display, HTML

%matplotlib inline

MODALITY = "{modality}"
BASE_DIR = os.path.abspath(f"./{modality}")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
ABLATION_CSV = os.path.join(BASE_DIR, f"{{MODALITY}}_ablation_study.csv")

print(f"Modality     : {{MODALITY.upper()}}")
print(f"Ablation CSV : {{ABLATION_CSV}}")
"""))

    # 3. Load & Display Ablation Study CSV
    cells.append(create_cell_code("""# Load generated Ablation Study CSV
if os.path.exists(ABLATION_CSV):
    print("=" * 80)
    print(f" ABLATION STUDY RESULTS ({MODALITY.upper()}) ")
    print("=" * 80)
    df_abl = pd.read_csv(ABLATION_CSV)
    display(df_abl)
else:
    print(f"Ablation study CSV not found at {ABLATION_CSV}.")
    print("Run csv_generator.py or main.py first to compute ablation results.")
"""))

    # 4. Detailed cCNN & cRNN Breakdowns
    cells.append(create_cell_code("""# Separate cCNN & cRNN breakdown if CSV exists
if os.path.exists(ABLATION_CSV):
    df_abl = pd.read_csv(ABLATION_CSV)
    
    if "Category" in df_abl.columns:
        df_ccnn = df_abl[df_abl["Category"] == "cCNN Ablation"]
        df_crnn = df_abl[df_abl["Category"] == "cRNN Ablation"]
        
        if not df_ccnn.empty:
            print("\n--- cCNN Ablations ---")
            display(df_ccnn)
            
        if not df_crnn.empty:
            print("\n--- cRNN Ablations ---")
            display(df_crnn)
"""))

    # 5. Visual Plots for Ablation Models
    cells.append(create_cell_code("""# Display Plots for Ablation Models
ccnn_models = ["cCNN", "cCNN_A1_NoPooling", "cCNN_A2_NoDropout", "cCNN_A3_Dense64", "cCNN_A4_Filters16", "cCNN_A5_Filters64", "cCNN_A6_Kernel5"]
crnn_models = ["cRNN", "cRNN_B1_OneLayer", "cRNN_B2_TwoLayers", "cRNN_B3_NoDropout", "cRNN_B4_Hidden1024", "cRNN_B5_Hidden512", "cRNN_B6_GRU"]

all_ablation_models = ccnn_models + crnn_models

for model in all_ablation_models:
    model_plots_dir = os.path.join(PLOTS_DIR, model)
    if os.path.exists(model_plots_dir):
        plot_files = glob(os.path.join(model_plots_dir, "**", "*.png"), recursive=True)
        if plot_files:
            print(f"\n--- Plots for {model} ---")
            cm_plots = [p for p in plot_files if "cm_test_norm" in p]
            roc_plots = [p for p in plot_files if "roc_test" in p]
            
            show_plots = (cm_plots[:2] if cm_plots else []) + (roc_plots[:2] if roc_plots else [])
            if not show_plots:
                show_plots = plot_files[:4]
                
            fig, axes = plt.subplots(1, len(show_plots), figsize=(5 * len(show_plots), 4))
            if len(show_plots) == 1:
                axes = [axes]
            for ax, p in zip(axes, show_plots):
                img = Image.open(p)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(os.path.basename(os.path.dirname(p)) + " / " + os.path.basename(p), fontsize=9)
            plt.tight_layout()
            plt.show()
"""))

    nb = make_notebook(cells)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def generate_view_output_notebook(modality: str, out_path: str):
    cells = []

    # Title cell
    cells.append(create_cell_markdown(f"""# {modality.upper()} Models Output Viewer

Interactive report viewer for **{modality.upper()}** spectrogram experiments.
Displays metrics, parameters, timing metrics (**training time**, **test time**, **average time per sample**), confusion matrices, and the expanded results summary CSV across non-ablation models.
"""))

    # Setup cell
    cells.append(create_cell_code(f"""import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from IPython.display import display

%matplotlib inline

MODALITY = "{modality}"
BASE_DIR    = os.path.abspath(f"./{modality}")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
SUMMARY_CSV = os.path.join(BASE_DIR, f"{{MODALITY}}_results_expanded_summary.csv")

print(f"Modality    : {{MODALITY.upper()}}")
print(f"Summary CSV : {{SUMMARY_CSV}}")
"""))

    # Load & display expanded summary CSV
    cells.append(create_cell_code("""if os.path.exists(SUMMARY_CSV):
    print("=" * 90)
    print(f" EXPANDED RESULTS SUMMARY ({MODALITY.upper()}) ")
    print("=" * 90)
    df_exp = pd.read_csv(SUMMARY_CSV)
    display(df_exp.head(20))
else:
    print(f"Expanded summary CSV not found at {SUMMARY_CSV}.")
"""))

    # All models summary table with timing metrics
    cells.append(create_cell_code("""if not os.path.exists(METRICS_DIR):
    print(f"Metrics directory not found: {METRICS_DIR}")
else:
    models = [d for d in os.listdir(METRICS_DIR) if os.path.isdir(os.path.join(METRICS_DIR, d)) and d != 'stats']
    
    summary_rows = []
    for model in models:
        ci_file = os.path.join(METRICS_DIR, model, "summary_ci.json")
        if os.path.exists(ci_file):
            with open(ci_file, "r") as f:
                d = json.load(f)
            
            acc = d.get("test_accuracy", {}).get("mean", float("nan")) * 100
            acc_std = d.get("test_accuracy", {}).get("std", float("nan")) * 100
            f1 = d.get("test_f1_macro", {}).get("mean", float("nan")) * 100
            f1_std = d.get("test_f1_macro", {}).get("std", float("nan")) * 100
            
            tr_time = d.get("training_time_s", {}).get("mean", float("nan"))
            te_time = d.get("test_time_s", {}).get("mean", float("nan"))
            avg_samp_time = d.get("avg_time_per_sample_s", {}).get("mean", float("nan")) * 1000 # ms
            
            summary_rows.append({
                "Model": model,
                "Accuracy (%)": f"{acc:.2f} ± {acc_std:.2f}",
                "F1 Macro (%)": f"{f1:.2f} ± {f1_std:.2f}",
                "Training Time (s)": round(tr_time, 2) if not pd.isna(tr_time) else "N/A",
                "Test Time (s)": round(te_time, 4) if not pd.isna(te_time) else "N/A",
                "Avg Time/Sample (ms)": round(avg_samp_time, 3) if not pd.isna(avg_samp_time) else "N/A"
            })
            
    print("=" * 90)
    print(" MODEL PERFORMANCE & TIMING METRICS OVERVIEW ")
    print("=" * 90)
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        display(df_summary)
    else:
        print("No completed models found in metrics directory yet.")
"""))

    # Detailed fold metrics and plots per model
    cells.append(create_cell_code("""if os.path.exists(METRICS_DIR):
    models = [d for d in os.listdir(METRICS_DIR) if os.path.isdir(os.path.join(METRICS_DIR, d)) and d != 'stats']
    
    for model in models:
        print(f"\n\n{'='*70}\n MODEL: {model} \n{'='*70}")
        
        csv_path = os.path.join(METRICS_DIR, model, "fold_metrics.csv")
        best_fold = None
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("--- Fold Metrics ---")
            display(df.head(5))
            
            if "test_f1_macro" in df.columns:
                df_sorted = df.sort_values(by="test_f1_macro", ascending=False)
                best_fold_val = df_sorted.iloc[0]["fold"]
                best_fold = f"fold_{int(best_fold_val):02d}"
                print(f"--> Best Fold: {best_fold}")
        
        model_plots_dir = os.path.join(PLOTS_DIR, model)
        if os.path.exists(model_plots_dir):
            all_plots = glob(os.path.join(model_plots_dir, "**", "*.png"), recursive=True)
            if all_plots:
                if best_fold:
                    best_plots = [p for p in all_plots if best_fold in p]
                    plots_to_show = best_plots if best_plots else all_plots[:4]
                else:
                    plots_to_show = all_plots[:4]
                
                num_plots = len(plots_to_show)
                if num_plots > 0:
                    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
                    if num_plots == 1:
                        axes = [axes]
                    for ax, plot_path in zip(axes, plots_to_show):
                        img = Image.open(plot_path)
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(os.path.basename(plot_path), fontsize=9)
                    plt.tight_layout()
                    plt.show()
"""))

    nb = make_notebook(cells)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def generate_all_notebooks(root_dir: str):
    for modality in MODALITIES:
        mod_dir = os.path.join(root_dir, modality)
        os.makedirs(mod_dir, exist_ok=True)
        
        # 1. Ablation test notebook
        ablation_nb_path = os.path.join(mod_dir, "ablation_test.ipynb")
        generate_ablation_notebook(modality, ablation_nb_path)
        print(f"Generated: {ablation_nb_path}")
        
        # 2. View output notebook
        view_nb_path = os.path.join(mod_dir, f"view_{modality}_output.ipynb")
        generate_view_output_notebook(modality, view_nb_path)
        print(f"Generated: {view_nb_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_notebooks(current_dir)
