"""
shared/trainer.py
=================
Inner training loop with early stopping.
Saves the best-validation-loss model checkpoint to disk at the end of each
outer fold so it can be reloaded for result reproduction / plotting.
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from shared.utils import get_logger, ensure_dir

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def _categorical_crossentropy(
    y_pred: torch.Tensor, y_true_onehot: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    
    if label_smoothing > 0.0:
        num_classes = y_true_onehot.size(-1)
        y_true_onehot = y_true_onehot * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        
    return -torch.mean(torch.sum(y_true_onehot * torch.log(y_pred), dim=1))


# ---------------------------------------------------------------------------
# Lazy-init helper  (materialise LazyConv3d / LazyLinear weights)
# ---------------------------------------------------------------------------
def lazy_init(model: nn.Module, sample_input: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        model(sample_input)
    model.train()


# ---------------------------------------------------------------------------
# Batch predict
# ---------------------------------------------------------------------------
def predict_batched(
    model: nn.Module, x: torch.Tensor, batch_size: int = 32
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            out.append(model(x[i : i + batch_size].to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    patience: int,
    verbose: int = 0,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """
    Train model with early stopping on validation loss.

    Parameters
    ----------
    checkpoint_path : if provided, saves best model weights as a .pt file.

    Returns
    -------
    history : dict with lists of train_loss and val_loss per epoch.
    """
    n = x_train.shape[0]
    best_val_loss = float("inf")
    best_state: Optional[Dict] = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        train_losses = []

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb  = x_train[idx].to(DEVICE)
            yb  = y_train[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = _categorical_crossentropy(model(xb), yb, label_smoothing=0.3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, x_val.shape[0], max(1, batch_size)):
                xb = x_val[start : start + batch_size].to(DEVICE)
                yb = y_val[start : start + batch_size].to(DEVICE)
                val_losses.append(_categorical_crossentropy(model(xb), yb, label_smoothing=0.0).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses)) if val_losses else float("inf")
        epoch_time = time.time() - t0
        
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(epoch_time)

        if verbose:
            logger.info(
                "epoch %d/%d  train_loss=%.5f  val_loss=%.5f  (%.1fs)",
                epoch + 1, epochs, train_loss, val_loss, epoch_time,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    logger.info("Early stopping at epoch %d.", epoch + 1)
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist checkpoint
    if checkpoint_path is not None and best_state is not None:
        ensure_dir(os.path.dirname(checkpoint_path))
        torch.save({"model_state_dict": best_state,
                    "best_val_loss": best_val_loss,
                    "history": history},
                   checkpoint_path)
        logger.info("Checkpoint saved → %s", checkpoint_path)

    return history
