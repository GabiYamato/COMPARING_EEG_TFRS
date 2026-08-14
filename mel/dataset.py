"""
mel/dataset.py
==============
Loads the MEL spectrogram dataset.
"""

import os
import pickle
import cv2
import numpy as np
from typing import Tuple
from shared.utils import get_logger

logger = get_logger(__name__)


def load_mel_dataset(
    root: str,
    img_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MEL dataset from pickle files."""
    logger.info("MEL root: %s", root)
    
    data_path = os.path.join(root, "data.pkl")
    labels_path = os.path.join(root, "labels.pkl")
    patients_path = os.path.join(root, "patients.pkl")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}")
        
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
    with open(labels_path, 'rb') as f:
        raw_labels = pickle.load(f)
    with open(patients_path, 'rb') as f:
        raw_patients = pickle.load(f)
        
    data = np.concatenate(raw_data, axis=0)
    labels = np.concatenate(raw_labels, axis=0)
    patients = np.concatenate(raw_patients, axis=0)
    
    current_h, current_w = data.shape[1], data.shape[2]
    if current_h != img_size or current_w != img_size:
        logger.info("Resizing MEL data from (%d, %d) to (%d, %d)...", current_h, current_w, img_size, img_size)
        resized_data = np.zeros((data.shape[0], img_size, img_size, data.shape[3]), dtype=np.float32)
        for i in range(data.shape[0]):
            resized_data[i] = cv2.resize(data[i].astype(np.float32), (img_size, img_size))
        data = resized_data
    else:
        data = data.astype(np.float32)

    data = np.transpose(data, (0, 3, 1, 2))
    labels = labels.astype(np.int64)
    patients = patients.astype(str)
        
    logger.info("MEL: %d samples loaded. Shape: %s", len(data), data.shape)
    return data, labels, patients
