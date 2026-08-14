# Comparative Benchmark of Deep Learning Models & Time-Frequency Representations (TFRs) for EEG Classification

A comprehensive benchmark framework evaluating deep learning architectures across multiple EEG Time-Frequency Representations (**STFT**, **CWT**, and **MEL** Spectrograms).

Includes **10-outer / 3-inner Stratified Nested Cross-Validation**, hyperparameter tuning, empirical runtime tracking, non-parametric bootstrapped 95% Confidence Intervals (CI), non-parametric statistical hypothesis testing (Shapiro-Wilk, Friedman, Wilcoxon Signed-Rank), and automated summary report generation.

---

## 🌟 Model Architectures

The framework evaluates 17 model variants across 4 key categories:

### 1. Configurable 3D-CNN (cCNN) & Ablation Variants
- **`cCNN`** (Baseline): 3D Convolutional Neural Network baseline.
- **`cCNN_A1_NoPooling`**: Removes Max Pooling layers to assess spatial downsampling impact.
- **`cCNN_A2_NoDropout`**: Removes Dropout layers to evaluate regularization effect.
- **`cCNN_A3_Dense64`**: Reduces Dense layer dimensionality from 128 to 64.
- **`cCNN_A4_Filters16`**: Reduces initial conv filters from 32 to 16.
- **`cCNN_A5_Filters64`**: Increases initial conv filters from 32 to 64.
- **`cCNN_A6_Kernel5`**: Increases spatial conv kernel size from 3×3×3 to 5×5×5.

### 2. Configurable Recurrent Neural Network (cRNN) & Ablation Variants
- **`cRNN`** (Baseline): 4-layer Recurrent Neural Network baseline.
- **`cRNN_B1_OneLayer`**: Single RNN layer topology.
- **`cRNN_B2_TwoLayers`**: 2-layer RNN topology.
- **`cRNN_B3_NoDropout`**: Removes Dropout layers across recurrent connections.
- **`cRNN_B4_Hidden1024`**: Increases hidden state units from 256 to 1024.
- **`cRNN_B5_Hidden512`**: Increases hidden state units from 256 to 512.
- **`cRNN_B6_GRU`**: Replaces SimpleRNN cells with Gated Recurrent Units (GRU).

### 3. Advanced / Hybrid Deep Learning Models
- **`CNN_LSTM`**: 3D-CNN feature extractor combined with unidirectional LSTM.
- **`CNN_BiLSTM`**: 3D-CNN feature extractor combined with Bidirectional LSTM.
- **`Swin_Transformer`**: Patch-based Vision Swin Transformer adapted for 3D spectrogram volume frames.
- **`ResBiLSTM`**: 1D Residual Convolutional Network combined with Bidirectional LSTM (*Zhao et al., Frontiers in Computational Neuroscience 2024*).

### 4. Standard Baseline References
- **`AlexNet`**: 3D adaptation of AlexNet architecture.
- **`VGGNET`**: 3D adaptation of VGGNet architecture.
- **`LSTM`**: Bidirectional LSTM with multi-head self-attention.

---

## 🛠️ Repository Structure

```
COMPARING_EEG_TFRS/
├── config.py                 # Central configuration (parameters, grid, default paths)
├── main.py                   # Main CLI entry point for cross-validation runs
├── csv_generator.py          # Summary CSV & Excel report generator
├── notebook_generator.py     # Jupyter Notebook report generator
├── run_mel.ps1 / run_mel.sh  # Automated execution scripts (with smart skip)
├── run_stft.ps1 / run_stft.sh
├── run_cwt.ps1 / run_cwt.sh
├── requirements.txt          # Dependencies
├── shared/                   # Core framework package
│   ├── models.py             # PyTorch implementations of all 17 architectures
│   ├── transforms.py         # Reshaping utilities (3D CNN, RNN, 1D Sequential)
│   ├── kfold.py              # 10-outer / 3-inner Stratified Nested CV engine
│   ├── trainer.py            # PyTorch training & evaluation loops with early stopping
│   ├── metrics.py            # Evaluation metrics & Bootstrapped 95% CIs
│   ├── plots.py              # Confusion matrix, ROC curves, metric comparisons
│   ├── stats.py              # Statistical analysis (Wilcoxon, Friedman, Shapiro-Wilk)
│   └── utils.py              # Directory, logging, seed, and file utilities
├── mel/                      # MEL dataset loader module
│   └── dataset.py
├── stft/                     # STFT dataset loader module
│   └── dataset.py
└── cwt/                      # CWT dataset loader module
    └── dataset.py
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/COMPARING_EEG_TFRS.git
   cd COMPARING_EEG_TFRS
   ```

2. **Create a virtual environment & install requirements:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 📁 Dataset Directory Setup

Set your dataset directory paths via environment variables or pass them as CLI arguments:

- **STFT Data**: Contains class subdirectories with image session frames.
- **CWT Data**: Contains `data.pkl`, `labels.pkl`, `patients.pkl`.
- **MEL Data**: Contains `data.pkl`, `labels.pkl`, `patients.pkl`.

### Option A: Environment Variables
```bash
export STFT_ROOT="/path/to/STFT"
export CWT_ROOT="/path/to/CWT/Variables"
export MEL_ROOT="/path/to/MEL/MEL_Variables"
```

### Option B: CLI Flags
Pass `--stft_root`, `--cwt_root`, `--mel_root`, or `--outputs_root` directly to `main.py`.

---

## ⚡ Usage Instructions

### 1. Run via PowerShell (Windows)
```powershell
.\run_mel.ps1
.\run_stft.ps1
.\run_cwt.ps1
```

### 2. Run via Bash (Linux / macOS)
```bash
chmod +x run_mel.sh run_stft.sh run_cwt.sh
./run_mel.sh
./run_stft.sh
./run_cwt.sh
```

### 3. Run Specific Experiments via Python CLI

#### Run all models for a specific dataset:
```bash
python main.py --dataset mel --mode all
```

#### Run a single model:
```bash
python main.py --dataset mel --model ResBiLSTM
```

#### Quick Smoke Test (2 outer / 2 inner folds):
```bash
python main.py --dataset stft --smoke
```

#### Custom Folds & Data Paths:
```bash
python main.py --dataset mel --model Swin_Transformer --outer 5 --inner 2 --mel_root ./data/MEL
```

---

## 📊 Evaluation & Metrics Tracked

For each fold and overall model aggregate, the framework measures:
- **Classification Performance**: Test Accuracy, F1 (Macro, Micro, Weighted), Precision (Macro, Micro, Weighted), Recall (Macro, Micro, Weighted), AUC (Macro, Micro, Weighted), Mean False Positive Rate (FPR).
- **Ablation $\Delta$ Metrics**: Absolute change relative to baseline for Accuracy ($\Delta\text{Acc}$), F1 ($\Delta\text{F1}$), AUC ($\Delta\text{AUC}$), Precision ($\Delta\text{Prec}$), Recall ($\Delta\text{Rec}$), and FPR ($\Delta\text{FPR}$).
- **Timing & Efficiency**: Training Time (s), Test Time (s), and Average Inference Time per Sample (ms).
- **Uncertainty Bounds**: Non-parametric Bootstrapped 95% Confidence Intervals (1000 resamples).
- **Statistical Tests**: Shapiro-Wilk test for normality, Friedman test for multi-model comparisons, and Wilcoxon Signed-Rank test for pairwise model comparisons.

---

## 📜 Citation & References

If you use this codebase or model implementations in your research, please cite the corresponding paper references:
- **ResBiLSTM**: Zhao W., Wang W.F., Patnaik L.M., et al., *"Residual and bidirectional LSTM for epileptic seizure detection"*, Frontiers in Computational Neuroscience, 2024, 18: 1415967.
