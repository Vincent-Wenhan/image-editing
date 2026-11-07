"""Latent-grid T/D/P masks and area-balanced flow-matching loss."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

REGION_NAMES = ("t_old", "t_new", "d_remove", "d_create", "p")


def _binary(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite HxW map")
    return array > 0.5


def build_region_masks(
    evidence: Mapping[str, np.ndarray],
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    shadow_threshold: float = 0.02,
) -> dict[str, np.ndarray]:
    """Build disjoint evaluation/supervision regions from frozen scene evidence."""
    source = np.asarray(evidence["source"], dtype=np.float32)
    target = np.asarray(evidence["target"], dtype=np.float32)
    source_no_shadow = np.asarray(evidence["source_no_shadow"], dtype=np.float32)
    target_no_shadow = np.asarray(evidence["target_no_shadow"], dtype=np.float32)
    old = _binary(evidence["source_occupancy"], "source occupancy")
    new = _binary(evidence["target_occupancy"], "target occupancy")
    labels = np.asarray(labels)
    valid = _binary(valid, "valid")
    if source.ndim != 3 or source.shape[-1] != 3 or source.shape[:2] != old.shape:
        raise ValueError("Role evidence RGB/occupancy geometry differs")
    if any(array.shape != source.shape for array in (target, source_no_shadow, target_no_shadow)):
        raise ValueError("Role evidence RGB geometry differs")
    if labels.shape != old.shape or new.shape != old.shape or valid.shape != old.shape:
        raise ValueError("Role labels and occupancy geometry differ")
    if not all(np.isfinite(x).all() for x in (source, target, source_no_shadow, target_no_shadow)):
        raise ValueError("Role evidence contains non-finite RGB")
    if not np.isfinite(shadow_threshold) or shadow_threshold <= 0:
        raise ValueError("Shadow threshold must be positive and finite")

    response = (target - target_no_shadow) - (source - source_no_shadow)
    dominant = np.take_along_axis(response, np.abs(response).argmax(axis=-1)[..., None], axis=-1)[
        ..., 0
    ]
    dependency = (labels == 1) & valid & ~(old | new)
    masks = {
        "t_old": old & ~new & valid,
        "t_new": new & ~old & valid,
        "d_remove": dependency & (dominant >= shadow_threshold),
        "d_create": dependency & (dominant <= -shadow_threshold),
        "p": (labels == 2) & valid,
    }
    stacked = np.stack([masks[name] for name in REGION_NAMES])
    if np.any(stacked.sum(axis=0) > 1):
        raise ValueError("T/D/P role regions overlap")
    return masks


def latent_masks(masks: Mapping[str, np.ndarray], height: int, width: int) -> np.ndarray:
    """Area-resample binary image masks to the actual VAE latent grid."""
    if height < 1 or width < 1:
        raise ValueError("Latent mask geometry must be positive")
    result = []
    for name in REGION_NAMES:
        values = np.asarray(masks[name], dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Region mask must be HxW: {name}")
        h, w = values.shape
        if h % height or w % width:
            raise ValueError("Latent mask grid must divide the image grid exactly")
        result.append(values.reshape(height, h // height, width, w // width).mean((1, 3)))
    return np.stack(result)


def weighted_region_loss(error, masks, *, min_pixels: float = 4.0) -> tuple[object, dict]:
    """Return equal-weight active-region loss and detached logging values.

    ``error`` is a per-element squared FM error shaped ``[B,...,H,W]`` and masks
    are ``[B,R,H,W]``. Area normalization happens before region averaging.
    """

    if not np.isfinite(min_pixels) or min_pixels <= 0:
        raise ValueError("Minimum latent region area must be positive and finite")
    if error.ndim < 4 or masks.ndim != 4 or error.shape[0] != masks.shape[0]:
        raise ValueError("FM error and role masks have incompatible dimensions")
    if error.shape[-2:] != masks.shape[-2:] or masks.shape[1] != len(REGION_NAMES):
        raise ValueError("FM error and role masks have incompatible geometry")
    pixel_error = error.float().mean(dim=tuple(range(1, error.ndim - 2)))
    areas = masks.float().sum(dim=(-2, -1))
    region_values = (pixel_error[:, None] * masks.float()).sum(dim=(-2, -1)) / areas.clamp_min(1.0)
    active = areas >= min_pixels
    active_count = active.sum(dim=1)
    sample_loss = (region_values * active).sum(dim=1) / active_count.clamp_min(1)
    role_loss = sample_loss.mean()
    records = {
        "role_loss": float(role_loss.detach()),
        "active_regions": float(active.float().mean().detach()),
        "region_loss": {
            name: float(region_values[:, index][active[:, index]].mean().detach())
            if active[:, index].any()
            else None
            for index, name in enumerate(REGION_NAMES)
        },
        "region_pixels": {
            name: float(areas[:, index].mean().detach()) for index, name in enumerate(REGION_NAMES)
        },
    }
    return role_loss, records
