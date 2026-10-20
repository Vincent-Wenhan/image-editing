"""Small model-independent editing contracts; supervision never enters EditInput."""

import math
from dataclasses import dataclass
from numbers import Real

from PIL import Image

from src.core.spatial import SpatialMap, array, rgb_size, roles, unit_map


@dataclass(frozen=True)
class EditInput:
    """Deployable inputs only; RGB source and optional same-size unit mask/strength."""

    source: Image.Image
    instruction: str
    mask: SpatialMap | None = None
    strength: float | None = None

    def __post_init__(self) -> None:
        size = rgb_size(self.source, "source")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Edit instruction must not be empty")
        if self.strength is not None and (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, Real)
            or not math.isfinite(self.strength)
            or not 0 <= self.strength <= 1
        ):
            raise ValueError("Strength must be in [0, 1]")
        if self.mask is not None and unit_map(self.mask, "mask") != size:
            raise ValueError("Input mask and source dimensions must match")


@dataclass(frozen=True)
class EditSupervision:
    """Training/assessment labels on target geometry; never passed to editing backends."""

    target: Image.Image
    role_labels: SpatialMap | None = None
    valid_region: SpatialMap | None = None

    def __post_init__(self) -> None:
        size = rgb_size(self.target, "target")
        if self.role_labels is not None and roles(self.role_labels) != size:
            raise ValueError("Role labels and target dimensions must match")
        if self.valid_region is not None:
            if unit_map(self.valid_region, "valid_region", binary=True) != size:
                raise ValueError("Valid region and target dimensions must match")
            if self.role_labels is not None:
                labels = array(self.role_labels, "role_labels")
                if labels.ndim == 2:
                    valid = array(self.valid_region, "valid_region") > 0
                    if (valid & (labels == 255)).any():
                        raise ValueError("Ignored role labels cannot be marked valid")


@dataclass(frozen=True)
class GenerationOptions:
    width: int
    height: int
    steps: int
    guidance: float
    seed: int


@dataclass(frozen=True)
class EditOutput:
    image: Image.Image
    seconds: float
    peak_allocated_gib: float
    peak_reserved_gib: float
