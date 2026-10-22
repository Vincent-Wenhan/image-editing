"""Named spatial conditions; inference inputs remain separate from supervision."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from src.core.spatial import SpatialMap, roles, unit_map


class ControlKind(StrEnum):
    NONE = "none"
    CONSTANT_MASK = "constant_mask"
    STATIC_MASK = "static_mask"
    DILATED_MASK = "dilated_mask"
    SHUFFLED_MASK = "shuffled_mask"
    SOFT_MASK = "soft_mask"
    ORACLE_ROLE = "oracle_role"
    ORACLE_GEOMETRY = "oracle_geometry"
    DYNAMIC_BINARY = "dynamic_binary"
    PREDICTED_ROLE = "predicted_role"


@dataclass(frozen=True)
class ControlBundle:
    kind: ControlKind = ControlKind.NONE
    maps: Mapping[str, SpatialMap] = field(default_factory=dict)
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ControlKind(self.kind))
        if self.kind == ControlKind.NONE and self.maps:
            raise ValueError("Unconditional controls must not contain maps")
        if self.kind == ControlKind.ORACLE_GEOMETRY:
            from src.conditioning.geometry import geometry_rgb

            geometry_rgb(self.maps)
        elif self.kind != ControlKind.NONE:
            is_role = self.kind in {ControlKind.ORACLE_ROLE, ControlKind.PREDICTED_ROLE}
            key = "roles" if is_role else "mask"
            if set(self.maps) != {key}:
                raise ValueError(f"{self.kind} requires exactly one named {key} map")
            if is_role:
                roles(self.maps[key])
            else:
                unit_map(self.maps[key], "mask", binary=self.kind != ControlKind.SOFT_MASK)
        if any(
            value is not None and not isinstance(value, str | int | float | bool)
            for value in self.metadata.values()
        ):
            raise TypeError("Control metadata accepts scalar values, never supervision objects")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in self.metadata.values()
        ):
            raise ValueError("Control metadata numbers must be finite")
        if self.metadata.get("role_order", "T,D,P") != "T,D,P":
            raise ValueError("Role channel order must be T,D,P")
        object.__setattr__(self, "maps", MappingProxyType(dict(self.maps)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def require_inference_safe(self) -> None:
        if self.kind in {
            ControlKind.ORACLE_ROLE,
            ControlKind.DYNAMIC_BINARY,
            ControlKind.ORACLE_GEOMETRY,
        }:
            raise ValueError("Oracle role labels are supervision; ordinary inference rejects them")

    def validate_geometry(self, size: tuple[int, int]) -> None:
        for name, value in self.maps.items():
            actual = roles(value) if name == "roles" else unit_map(value, name)
            if actual != size:
                raise ValueError("Control map and source dimensions must match")
