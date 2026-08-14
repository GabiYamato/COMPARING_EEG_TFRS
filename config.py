"""
config.py
=========
Central configuration for cross-validation EEG classification experiments.
All dataset and output paths use relative defaults or environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import List

_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ExperimentConfig:
    # ------------------------------------------------------------------
    # Nested cross-validation topology
    # ------------------------------------------------------------------
    outer_folds: int = 10
    inner_folds: int = 3

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs: int = 30
    patience: int = 5
    seed: int = 42
    verbose: int = 0

    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------
    lr_values: List[float] = field(default_factory=lambda: [1e-3, 3e-4, 1e-4])
    batch_values: List[int] = field(default_factory=lambda: [16, 32])
    patience_values: List[int] = field(default_factory=lambda: [5])
    max_trials_per_model: int = 5
    balance_factor: int = 1

    # ------------------------------------------------------------------
    # Bootstrapped confidence interval
    # ------------------------------------------------------------------
    n_bootstrap: int = 1000

    # ------------------------------------------------------------------
    # Model categories
    # ------------------------------------------------------------------
    ablation_ccnn_models: List[str] = field(
        default_factory=lambda: [
            "cCNN",
            "cCNN_A1_NoPooling",
            "cCNN_A2_NoDropout",
            "cCNN_A3_Dense64",
            "cCNN_A4_Filters16",
            "cCNN_A5_Filters64",
            "cCNN_A6_Kernel5",
        ]
    )

    ablation_crnn_models: List[str] = field(
        default_factory=lambda: [
            "cRNN",
            "cRNN_B1_OneLayer",
            "cRNN_B2_TwoLayers",
            "cRNN_B3_NoDropout",
            "cRNN_B4_Hidden1024",
            "cRNN_B5_Hidden512",
            "cRNN_B6_GRU",
        ]
    )

    new_models: List[str] = field(
        default_factory=lambda: [
            "CNN_LSTM",
            "CNN_BiLSTM",
            "Swin_Transformer",
        ]
    )

    reference_models: List[str] = field(
        default_factory=lambda: [
            "AlexNet",
            "VGGNET",
            "LSTM",
            "ResBiLSTM",
        ]
    )

    @property
    def models_to_run(self) -> List[str]:
        return (
            self.ablation_ccnn_models
            + self.ablation_crnn_models
            + self.new_models
            + self.reference_models
        )

    # ------------------------------------------------------------------
    # Smoke-test overrides (--smoke flag)
    # ------------------------------------------------------------------
    smoke_outer_folds: int = 2
    smoke_inner_folds: int = 2

    # ------------------------------------------------------------------
    # Dataset volume size fed into 3-D CNNs (D, H, W)
    # ------------------------------------------------------------------
    cnn3d_volume_size: tuple = (20, 32, 32)

    # ------------------------------------------------------------------
    # Dataset root paths (Relative defaults, overridable via Env or CLI)
    # ------------------------------------------------------------------
    stft_root: str = field(
        default_factory=lambda: os.getenv(
            "STFT_ROOT", os.path.join(_ROOT, "data", "STFT")
        )
    )
    cwt_root: str = field(
        default_factory=lambda: os.getenv(
            "CWT_ROOT", os.path.join(_ROOT, "data", "CWT")
        )
    )
    mel_root: str = field(
        default_factory=lambda: os.getenv(
            "MEL_ROOT", os.path.join(_ROOT, "data", "MEL")
        )
    )

    # ------------------------------------------------------------------
    # Output root directory
    # ------------------------------------------------------------------
    outputs_root: str = field(
        default_factory=lambda: os.getenv("OUTPUTS_ROOT", _ROOT)
    )
