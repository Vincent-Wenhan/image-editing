"""Control data only: no spatial-control or role-prediction algorithm is implemented."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from PIL import Image


class ControlKind(StrEnum):
    NONE = "none"
    STATIC_MASK = "static_mask"
    DILATED_MASK = "dilated_mask"
    SOFT_MASK = "soft_mask"
    ORACLE_ROLE = "oracle_role"
    PREDICTED_ROLE = "predicted_role"


@dataclass(frozen=True)
class ControlBundle:
    kind: ControlKind = ControlKind.NONE
    maps: Mapping[str, Image.Image] = field(default_factory=dict)
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ControlKind(self.kind))
        if self.kind == ControlKind.NONE and self.maps:
            raise ValueError("Unconditional controls must not contain maps")
        if any(not isinstance(value, Image.Image) for value in self.maps.values()):
            raise TypeError("Control maps must be PIL images")
        if any(
            value is not None and not isinstance(value, str | int | float | bool)
            for value in self.metadata.values()
        ):
            raise TypeError("Control metadata accepts scalar values, never supervision objects")
        object.__setattr__(self, "maps", MappingProxyType(dict(self.maps)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def require_inference_safe(self) -> None:
        if self.kind == ControlKind.ORACLE_ROLE:
            raise ValueError("Oracle role labels are supervision; ordinary inference rejects them")
