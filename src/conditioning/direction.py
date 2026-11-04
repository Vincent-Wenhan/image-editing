"""Geometry-only directional conditions for a future role ablation.

The condition is derived from occupancy and valid geometry only. RGB source/target
images and role supervision are deliberately absent from this module.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

DIRECTION_SCHEMA = "geometry_direction_field_v1"
CHANNELS = ("old", "new", "dependency", "dx", "dy", "boundary", "valid")


def _map(value, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite HxW map")
    if shape is not None and result.shape != shape:
        raise ValueError("Direction condition maps must share geometry")
    if result.min() < 0 or result.max() > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _centroid(occupancy: np.ndarray, name: str) -> np.ndarray:
    weight = np.clip(occupancy, 0, 1)
    total = float(weight.sum())
    if total <= np.finfo(np.float32).eps:
        raise ValueError(f"{name} occupancy is empty")
    y, x = np.indices(weight.shape, dtype=np.float32)
    return np.array([(x * weight).sum() / total, (y * weight).sum() / total], np.float32)


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


@dataclass(frozen=True)
class DirectionCondition:
    """Aligned geometry fields with signed normalized displacement channels."""

    values: np.ndarray
    metadata: dict

    def __post_init__(self):
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != len(CHANNELS):
            raise ValueError("Direction condition must be CxHxW with declared channels")
        if not np.isfinite(values).all():
            raise ValueError("Direction condition must be finite")
        if (
            values[:3].min() < 0
            or values[:3].max() > 1
            or values[5:].min() < 0
            or values[5:].max() > 1
            or values[3:5].min() < -1
            or values[3:5].max() > 1
        ):
            raise ValueError("Direction condition values exceed their declared ranges")
        if (values[3:5, values[6] == 0] != 0).any():
            raise ValueError("Displacement must be zero outside the valid region")
        frozen = np.array(values, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "values", frozen)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def size(self) -> tuple[int, int]:
        return self.values.shape[2], self.values.shape[1]

    def tokens(self, stride: int = 16) -> np.ndarray:
        if type(stride) is not int or stride < 1 or any(n % stride for n in self.values.shape[1:]):
            raise ValueError("Direction token stride must divide both spatial dimensions")
        channels, height, width = self.values.shape
        return (
            self.values.reshape(channels, height // stride, stride, width // stride, stride)
            .mean((2, 4))
            .transpose(1, 2, 0)
            .reshape(-1, channels)
            .copy()
        )

    @property
    def digest(self) -> str:
        header = json.dumps(
            dict(schema=DIRECTION_SCHEMA, channels=CHANNELS, shape=self.values.shape),
            sort_keys=True,
        )
        return hashlib.sha256(header.encode() + self.values.tobytes()).hexdigest()


def build_direction_condition(
    old, new, dependency, valid, *, normalize_by: str = "image"
) -> DirectionCondition:
    """Build a target-free displacement field from aligned geometry maps.

    ``dx`` and ``dy`` are the old-to-new centroid displacement normalized by image
    width/height and broadcast over valid pixels. The field is zero for identity
    occupancy and outside ``valid``. It is a geometric upper-bound condition only.
    """
    if normalize_by != "image":
        raise ValueError("Only image-relative displacement normalization is supported")
    old = _map(old, "old")
    shape = old.shape
    new = _map(new, "new", shape)
    dependency = _map(dependency, "dependency", shape)
    valid = _map(valid, "valid", shape)
    old_center, new_center = _centroid(old, "old"), _centroid(new, "new")
    height, width = shape
    displacement = (new_center - old_center) / np.array([width, height], np.float32)
    union = (old > 0.5) | (new > 0.5)
    boundary = _dilate(union) & ~_erode(union)
    valid_mask = valid > 0.5
    dx = np.full(shape, displacement[0], np.float32) * valid_mask
    dy = np.full(shape, displacement[1], np.float32) * valid_mask
    values = np.stack([old, new, dependency, dx, dy, boundary.astype(np.float32), valid])
    return DirectionCondition(
        values,
        dict(
            schema=DIRECTION_SCHEMA,
            channel_order=",".join(CHANNELS),
            old_centroid=old_center.tolist(),
            new_centroid=new_center.tolist(),
            displacement=displacement.tolist(),
            normalize_by=normalize_by,
        ),
    )
