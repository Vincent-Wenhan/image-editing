"""Opt-in oracle reference protocol, separate from ordinary editing dispatch."""

import re

from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.oracle import ROLE_ENCODING, role_rgb
from src.conditioning.static import validate_reference_size
from src.core.types import EditInput, GenerationOptions


def oracle_references(
    edit: EditInput,
    controls: ControlBundle,
    options: GenerationOptions,
    *,
    allow_oracle: bool = False,
) -> list[Image.Image]:
    if not allow_oracle or controls.kind != ControlKind.ORACLE_ROLE:
        raise ValueError("Use the explicit oracle upper-bound entry with allow_oracle=True")
    if (
        controls.metadata.get("purpose") != "oracle_upper_bound"
        or controls.metadata.get("role_encoding") != ROLE_ENCODING
        or controls.metadata.get("role_order") != "T,D,P"
        or controls.metadata.get("image_order") != "source,role_rgb"
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
    return [edit.source, role_rgb(controls.maps["roles"])]
