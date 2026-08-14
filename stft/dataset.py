"""
stft/dataset.py
===============
Loads the STFT image dataset.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple
from shared.utils import get_logger

logger = get_logger(__name__)

_IMG_PATTERN = "image"


def _load_session(session_dir: str, img_size: int = 64) -> np.ndarray:
    files = sorted(
        [f for f in os.listdir(session_dir) if f.startswith(_IMG_PATTERN) and f.endswith(".png")],
        key=lambda f: int(f.replace(_IMG_PATTERN, "").replace(".png", "")),
    )
    frames = []
    for fname in files:
        path = os.path.join(session_dir, fname)
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32) / 255.0)

    if not frames:
        return np.zeros((1, img_size, img_size), dtype=np.float32)
    return np.stack(frames, axis=0)


def load_stft_dataset(
    root: str,
    img_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in: {root}")

    data_list:     List[np.ndarray] = []
    labels_list:   List[str]        = []
    patients_list: List[str]        = []

    for class_label in class_dirs:
        class_path = os.path.join(root, class_label)
        patient_dirs = sorted([
            d for d in os.listdir(class_path)
            if os.path.isdir(os.path.join(class_path, d))
        ])
        for patient_id in patient_dirs:
            patient_path = os.path.join(class_path, patient_id)
            session_dirs = sorted([
                d for d in os.listdir(patient_path)
                if os.path.isdir(os.path.join(patient_path, d))
            ])
            for session_name in session_dirs:
                session_path = os.path.join(patient_path, session_name)
                volume = _load_session(session_path, img_size=img_size)
                data_list.append(volume)
                labels_list.append(class_label)
                patients_list.append(patient_id)

    logger.info("STFT: loaded %d samples from %d classes.", len(data_list), len(class_dirs))
    data     = np.empty(len(data_list), dtype=object)
    data[:]  = data_list
    return data, np.array(labels_list), np.array(patients_list)
