"""Small model-independent editing contracts; supervision never enters EditInput."""

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class EditInput:
    source: Image.Image
    instruction: str
    mask: Image.Image | None = None
    strength: float | None = None

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("Edit instruction must not be empty")
        if self.strength is not None and not 0 <= self.strength <= 1:
            raise ValueError("Strength must be in [0, 1]")
        if self.mask is not None and self.mask.size != self.source.size:
            raise ValueError("Input mask and source dimensions must match")


@dataclass(frozen=True)
class EditSupervision:
    target: Image.Image
    role_labels: Image.Image | None = None
    valid_region: Image.Image | None = None


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
