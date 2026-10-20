"""Pixel diagnostics only; they do not measure semantic success or necessity."""

import numpy as np
from PIL import Image


def load_image(path, size, mode="RGB"):
    with Image.open(path) as image:
        image = image.convert(mode)
        interpolation = Image.Resampling.NEAREST if mode == "L" else Image.Resampling.LANCZOS
        return np.asarray(image.resize(size, interpolation), dtype=np.float32) / 255


def metrics(source, target, prediction, mask, threshold):
    drift = np.abs(prediction - source).mean(axis=2)
    outside = mask < 0.5
    return dict(
        target_mae=float(np.abs(prediction - target).mean()),
        identity_target_mae=float(np.abs(source - target).mean()),
        changed_area_fraction=float((drift > threshold).mean()),
        source_drift_mae=float(drift.mean()),
        outside_mask_drift_proxy=float(drift[outside].mean()) if outside.any() else None,
    )
