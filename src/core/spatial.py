"""Validation-only spatial contracts: pixels, unit maps and T/D/P role probabilities."""

import numpy as np
from PIL import Image

SpatialMap = Image.Image | np.ndarray
ROLE_ORDER = ("T", "D", "P")


def rgb_size(image: Image.Image, name: str) -> tuple[int, int]:
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise TypeError(f"{name} must be a PIL RGB image")
    if min(image.size) < 1:
        raise ValueError(f"{name} dimensions must be positive")
    return image.size


def array(value: SpatialMap, name: str) -> np.ndarray:
    if not isinstance(value, Image.Image | np.ndarray):
        raise TypeError(f"{name} must be a PIL image or NumPy array")
    values = np.asarray(value)
    if values.ndim < 2 or min(values.shape[:2]) < 1:
        raise ValueError(f"{name} dimensions must be nonempty H x W")
    if values.dtype.kind not in "buif" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return values


def unit_map(value: SpatialMap, name: str, *, binary: bool = False) -> tuple[int, int]:
    values = array(value, name)
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape H x W")
    if isinstance(value, Image.Image) and value.mode not in {"1", "L", "F"}:
        raise TypeError(f"{name} PIL mode must be 1, L or F")
    if isinstance(value, Image.Image) and value.mode == "L":
        values = values.astype(np.float32) / 255
    # NumPy maps use unit values; PIL L alone uses the byte-image convention.
    if values.min() < 0 or values.max() > 1:
        raise ValueError(f"{name} values must be in [0, 1]")
    if binary and not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must be binary")
    return values.shape[1], values.shape[0]


def roles(value: SpatialMap, name: str = "roles") -> tuple[int, int]:
    """Hard labels: HxW integers 0/1/2/255(ignore); soft: HxWx3 float T/D/P."""
    values = array(value, name)
    if values.ndim == 2:
        if values.dtype.kind not in "ui" or not np.isin(values, [0, 1, 2, 255]).all():
            raise ValueError(f"{name} hard labels must be integer T=0/D=1/P=2/ignore=255")
    elif values.ndim == 3 and values.shape[2] == 3:
        if values.dtype.kind != "f" or values.min() < 0 or values.max() > 1:
            raise ValueError(f"{name} probabilities must be floating values in [0, 1]")
        if not np.allclose(values.sum(axis=2), 1, rtol=0, atol=1e-5):
            raise ValueError(f"{name} T/D/P probabilities must sum to one per pixel")
    else:
        raise ValueError(f"{name} must have shape H x W or H x W x 3")
    return values.shape[1], values.shape[0]
