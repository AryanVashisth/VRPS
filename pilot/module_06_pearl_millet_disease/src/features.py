"""Colour features for pearl millet leaf images (30 floats).

Same recipe as development/module_03_pearl_millet_disease/train_baselines.py:
resize to 64x64 RGB, scale to 0-1, then per channel: mean, std, 8-bin histogram
(normalised to sum to 1). Order: 3 means, 3 stds, then r/g/b histograms.

An Android app must reproduce this exactly (same resize, same bins) before it
feeds the ONNX model, or call shared feature code.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

IMAGE_SIZE = (64, 64)
N_BINS = 8


def colour_features(path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    means = pixels.mean(axis=(0, 1))
    stds = pixels.std(axis=(0, 1))
    histograms = []
    for channel in range(3):
        counts, _ = np.histogram(pixels[:, :, channel], bins=N_BINS, range=(0.0, 1.0))
        histograms.append(counts / counts.sum())
    return np.concatenate([means, stds, *histograms]).astype(np.float32)


def feature_names() -> List[str]:
    names = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    for channel in ("r", "g", "b"):
        names.extend(f"hist_{channel}_{i}" for i in range(N_BINS))
    return names
