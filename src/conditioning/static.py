"""Deterministic binary mask transforms; no target pixels or learned components."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from src.conditioning.controls import ControlKind
from src.core.spatial import SpatialMap, unit_map

MASK_KINDS = (ControlKind.STATIC_MASK, ControlKind.DILATED_MASK, ControlKind.SHUFFLED_MASK)


@dataclass(frozen=True)
class MaskSpec:
    kind: ControlKind
    dilation_radius: int = 0
    shuffle_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ControlKind(self.kind))
        if self.kind not in MASK_KINDS:
            raise ValueError("Only static, dilated and shuffled masks are implemented")
        if isinstance(self.dilation_radius, bool) or not isinstance(self.dilation_radius, int):
            raise ValueError("Dilation radius must be an integer")
        if not 0 <= self.dilation_radius <= 128:
            raise ValueError("Dilation radius must be between 0 and 128 pixels")
        if (self.kind == ControlKind.DILATED_MASK) != (self.dilation_radius > 0):
            raise ValueError("Only Dilated Mask requires a positive dilation radius")
        if self.kind == ControlKind.SHUFFLED_MASK and self.shuffle_seed is None:
            raise ValueError("Shuffled Mask requires an explicit seed")
        if self.shuffle_seed is not None and (
            isinstance(self.shuffle_seed, bool) or not isinstance(self.shuffle_seed, int)
        ):
            raise ValueError("Shuffle seed must be an integer")


def binary_image(mask: SpatialMap) -> Image.Image:
    unit_map(mask, "mask", binary=True)
    values = np.asarray(mask)
    if isinstance(mask, Image.Image) and mask.mode == "L":
        return mask.copy()
    return Image.fromarray((values * 255).astype(np.uint8))


def transform_mask(mask: SpatialMap, spec: MaskSpec) -> Image.Image:
    image = binary_image(mask)
    if spec.kind == ControlKind.DILATED_MASK:
        image = image.filter(ImageFilter.MaxFilter(2 * spec.dilation_radius + 1))
    return image


def mask_rgb(mask: SpatialMap) -> Image.Image:
    """Three identical byte channels; VAE preprocessing alone maps them to [-1,1]."""
    return binary_image(mask).convert("RGB")


def validate_reference_size(size: tuple[int, int]) -> None:
    if min(size) < 64 or any(v % 16 for v in size) or size[0] * size[1] > 1024**2:
        raise ValueError("FLUX references need sides >=64, multiples of 16 and area <=1024²")
