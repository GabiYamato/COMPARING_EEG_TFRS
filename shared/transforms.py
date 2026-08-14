"""
shared/transforms.py
====================
Dataset-agnostic data-preparation utilities: reshaping raw arrays into
the tensor shapes expected by each model family (3-D CNNs, RNN/LSTM,
and 1-D sequential models like ResBiLSTM).
"""

from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np


def _downsample_to_shape(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    result = arr
    for axis, target in enumerate(target_shape):
        current = result.shape[axis]
        if current == target:
            continue
        if current < target:
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (0, target - current)
            result = np.pad(result, pad_width, mode="constant", constant_values=0.0)
        else:
            indices = np.linspace(0, current - 1, target).astype(int)
            result = np.take(result, indices, axis=axis)
    return result


def choose_time_steps(feature_count: int) -> int:
    max_candidate = min(64, feature_count)
    for candidate in range(max_candidate, 1, -1):
        if feature_count % candidate == 0:
            return candidate
    return 1


def to_3d_cnn_input(
    data: np.ndarray, volume_size: Tuple[int, int, int] = (32, 32, 32)
) -> np.ndarray:
    converted = []
    for sample in data:
        arr = np.array(sample, dtype=np.float32)

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            pass
        elif arr.ndim == 4:
            arr = np.transpose(arr, (1, 2, 3, 0)) if arr.shape[0] < arr.shape[-1] else arr

        if arr.ndim == 4:
            arr = np.mean(arr, axis=-1)

        arr = _downsample_to_shape(arr, volume_size)
        converted.append(arr.astype(np.float32))

    shape_counter = Counter([x.shape for x in converted])
    target_shape = shape_counter.most_common(1)[0][0]
    aligned = []
    for arr in converted:
        pads, slices = [], []
        for ax, tgt in enumerate(target_shape):
            cur = arr.shape[ax]
            pads.append((0, max(0, tgt - cur)))
            slices.append(slice(0, tgt))
        arr = np.pad(arr, pads, mode="constant")
        arr = arr[tuple(slices)]
        aligned.append(arr)

    stacked = np.stack(aligned, axis=0)              # (N, D, H, W)
    stacked = stacked[:, np.newaxis, :, :, :]        # (N, 1, D, H, W) channel-first
    return stacked.astype(np.float32)


def to_rnn_flat_input(
    data: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    flattened = []
    for sample in data:
        arr = np.array(sample, dtype=np.float32)
        if arr.ndim >= 3:
            arr = np.mean(arr, axis=tuple(range(2, arr.ndim)))
        arr = _downsample_to_shape(arr.reshape(-1, 32) if arr.ndim == 2
                                   else arr.reshape(32, 32), (32, 32))
        flattened.append(arr.reshape(-1))

    x_flat = np.stack(flattened, axis=0).astype(np.float32)
    feature_count = x_flat.shape[1]
    time_steps = choose_time_steps(feature_count)
    input_dim  = feature_count // time_steps
    return x_flat, time_steps, input_dim


def to_seq1d_input(
    data: np.ndarray,
) -> np.ndarray:
    """
    Prepare input for 1-D sequential models (e.g. ResBiLSTM).
    Returns an array of shape (N, 1, L) where L is the flattened feature
    length, matching Conv1d channel-first convention.
    """
    flattened = []
    for sample in data:
        arr = np.array(sample, dtype=np.float32)
        if arr.ndim >= 3:
            arr = np.mean(arr, axis=tuple(range(2, arr.ndim)))
        arr = _downsample_to_shape(arr.reshape(-1, 32) if arr.ndim == 2
                                   else arr.reshape(32, 32), (32, 32))
        flattened.append(arr.reshape(-1))

    x_flat = np.stack(flattened, axis=0).astype(np.float32)  # (N, L)
    # Add channel dimension → (N, 1, L) for Conv1d
    x_seq1d = x_flat[:, np.newaxis, :]                        # (N, 1, L)
    return x_seq1d


def standardize(
    x_train: np.ndarray, x_eval: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    x_tr = x_train.astype(np.float32, copy=False)
    x_ev = x_eval.astype(np.float32, copy=False)
    mean = np.mean(x_tr, axis=0, keepdims=True)
    std  = np.std(x_tr, axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (x_tr - mean) / std, (x_ev - mean) / std


def prepare_inputs(
    data: np.ndarray,
    model_name: str,
    volume_size: Tuple[int, int, int] = (32, 32, 32),
) -> Dict[str, Any]:
    cnn3d_models = {
        "cCNN", "cCNN_A1_NoPooling", "cCNN_A2_NoDropout", "cCNN_A3_Dense64",
        "cCNN_A4_Filters16", "cCNN_A5_Filters64", "cCNN_A6_Kernel5",
        "A1", "A2", "A3", "A4", "A5", "A6",
        "AlexNet", "VGGNET", "CNN_LSTM", "CNN_BiLSTM", "CNN-BiLSTM",
        "Swin_Transformer", "Swin Transformer", "SwinTransformer"
    }

    rnn_models = {
        "cRNN", "RNN", "cRNN_B1_OneLayer", "cRNN_B2_TwoLayers", "cRNN_B3_NoDropout",
        "cRNN_B4_Hidden1024", "cRNN_B5_Hidden512", "cRNN_B6_GRU",
        "B1", "B2", "B3", "B4", "B5", "B6", "LSTM"
    }

    # 1-D sequential models: expects (B, 1, L)
    seq1d_models = {
        "ResBiLSTM", "ResBiLSTM_Net"
    }

    if model_name in cnn3d_models:
        x = to_3d_cnn_input(data, volume_size)
        return {"mode": "cnn3d", "x": x}

    if model_name in rnn_models:
        x_flat, time_steps, input_dim = to_rnn_flat_input(data)
        return {"mode": "rnn", "x": x_flat,
                "time_steps": time_steps, "input_dim": input_dim}

    if model_name in seq1d_models:
        x_seq = to_seq1d_input(data)
        return {"mode": "seq1d", "x": x_seq}

    raise ValueError(f"Unknown model name: '{model_name}'")
