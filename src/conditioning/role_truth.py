"""Recompute T/D/P from occupancy and a shadow-ray intervention, not raw RGB change."""

import numpy as np
from PIL import Image, ImageFilter


def neighborhood(mask: np.ndarray, radius: int, *, erode: bool = False) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    if radius:
        filter_type = ImageFilter.MinFilter if erode else ImageFilter.MaxFilter
        image = image.filter(filter_type(2 * radius + 1))
    return np.asarray(image) > 0


def derive_roles(evidence: dict, strength: float, policy: dict) -> tuple[np.ndarray, np.ndarray]:
    if (
        isinstance(strength, bool)
        or not isinstance(strength, int | float)
        or not np.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("Physical strength must be in [0,1]")
    source, target, source_off, target_off = [
        np.asarray(evidence[key])
        for key in ("source", "target", "source_no_shadow", "target_no_shadow")
    ]
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or any(
            x.shape != source.shape or not np.isfinite(x).all()
            for x in (source, target, source_off, target_off)
        )
    ):
        raise ValueError("Counterfactual evidence must be finite aligned linear RGB")
    old, new = [np.asarray(evidence[key]) for key in ("source_occupancy", "target_occupancy")]
    if any(
        x.shape != source.shape[:2] or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1.00001
        for x in (old, new)
    ):
        raise ValueError("Occupancy must be aligned unit object-ID coverage")
    if strength == 0:
        if not (
            np.array_equal(source, target)
            and np.array_equal(source_off, target_off)
            and np.array_equal(old, new)
        ):
            raise ValueError("Zero strength must be an exact identity")
        return np.full(old.shape, 2, dtype=np.uint8), np.ones(old.shape, dtype=np.uint8) * 255
    direct = (old > 0.5) | (new > 0.5)
    shadow_change = (target - target_off) - (source - source_off)
    effect = np.max(np.abs(shadow_change), axis=2)
    dependency = (effect >= policy["shadow_high"]) & ~direct
    ambiguous = ((effect > policy["shadow_low"]) & (effect < policy["shadow_high"])) & ~direct
    # Outside occupied geometry, a no-shadow difference is an unexplained effect/noise.
    unexplained = np.max(np.abs(target_off - source_off), axis=2) > policy["unexplained_tolerance"]
    radius = policy["boundary_radius"]
    # Each silhouette matters, including old/new boundaries inside their union.
    boundary = np.zeros(old.shape, dtype=bool)
    for footprint in (old > 0.5, new > 0.5):
        boundary |= neighborhood(footprint, radius) ^ neighborhood(footprint, radius, erode=True)
    partial = ((old > 1e-5) & (old < 0.99999)) | ((new > 1e-5) & (new < 0.99999))
    invalid = boundary | partial | ambiguous | (unexplained & ~direct)
    labels = np.full(old.shape, 2, dtype=np.uint8)
    labels[direct], labels[dependency], labels[invalid] = 0, 1, 255
    return labels, (~invalid).astype(np.uint8) * 255
