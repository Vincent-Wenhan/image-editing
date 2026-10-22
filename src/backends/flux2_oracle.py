"""Opt-in oracle reference protocol, separate from ordinary editing dispatch."""

import re

from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.geometry import GEOMETRY_ENCODING, geometry_rgb
from src.conditioning.oracle import ROLE_ENCODING, role_rgb
from src.conditioning.static import mask_rgb, validate_reference_size
from src.core.types import EditInput, GenerationOptions


def oracle_references(
    edit: EditInput,
    controls: ControlBundle,
    options: GenerationOptions,
    *,
    allow_oracle: bool = False,
) -> list[Image.Image]:
    if not allow_oracle or controls.kind not in {
        ControlKind.ORACLE_ROLE,
        ControlKind.DYNAMIC_BINARY,
        ControlKind.ORACLE_GEOMETRY,
    }:
        raise ValueError("Use the explicit oracle upper-bound entry with allow_oracle=True")
    if (
        controls.metadata.get("purpose") != "oracle_upper_bound"
        or controls.metadata.get("role_encoding")
        != (
            {
                ControlKind.ORACLE_ROLE: ROLE_ENCODING,
                ControlKind.DYNAMIC_BINARY: "oracle_union_td_binary_v1",
                ControlKind.ORACLE_GEOMETRY: GEOMETRY_ENCODING,
            }[controls.kind]
        )
        or controls.metadata.get("role_order") != "T,D,P"
        or controls.metadata.get("image_order")
        != {
            ControlKind.ORACLE_ROLE: "source,role_rgb",
            ControlKind.DYNAMIC_BINARY: "source,mask",
            ControlKind.ORACLE_GEOMETRY: "source,geometry_rgb",
        }[controls.kind]
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(controls.metadata.get(key, "")))
            for key in ("role_sha256", "valid_sha256", "evidence_sha256", "source_sha256")
        )
    ):
        raise ValueError("Oracle encoding and supervision provenance must be explicit")
    if edit.mask is not None or edit.strength is not None:
        raise NotImplementedError("Extra mask/scalar-strength conditioning is not implemented")
    controls.validate_geometry(edit.source.size)
    validate_reference_size(edit.source.size)
    if edit.source.size != (options.width, options.height):
        raise ValueError("Oracle/source/output dimensions must match")
    image = (
        geometry_rgb(controls.maps)
        if controls.kind == ControlKind.ORACLE_GEOMETRY
        else (
            role_rgb(controls.maps["roles"])
            if controls.kind == ControlKind.ORACLE_ROLE
            else mask_rgb(controls.maps["mask"])
        )
    )
    return [edit.source, image]
