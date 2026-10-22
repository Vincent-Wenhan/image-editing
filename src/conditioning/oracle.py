"""Explicit oracle upper-bound compilation; accepts labels, never target RGB."""

import re

import numpy as np
from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.core.spatial import SpatialMap, array, roles, unit_map

ROLE_ENCODING = "one_hot_rgb_tdp_v1"


def role_rgb(labels: SpatialMap) -> Image.Image:
    roles(labels)
    values = array(labels, "roles")
    if values.ndim != 2:
        raise NotImplementedError("The oracle model interface currently encodes hard T/D/P only")
    rgb = np.stack([values == label for label in (0, 1, 2)], axis=-1).astype(np.uint8) * 255
    return Image.fromarray(rgb)


def oracle_bundle(labels: SpatialMap, valid: SpatialMap, *, metadata: dict) -> ControlBundle:
    size = roles(labels)
    if unit_map(valid, "valid", binary=True) != size:
        raise ValueError("Oracle role/valid geometry must match")
    values = array(labels, "roles")
    if values.ndim != 2:
        raise NotImplementedError("Oracle compilation currently supports hard labels only")
    valid_values = array(valid, "valid") > 0
    if ((values == 255) & valid_values).any():
        raise ValueError("Ignored oracle labels cannot be valid")
    required = {"role_sha256", "valid_sha256", "evidence_sha256", "source_sha256"}
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))) for key in required):
        raise ValueError("Oracle compilation requires explicit SHA256 provenance")
    compiled = values.copy()
    compiled[~valid_values] = 255
    return ControlBundle(
        ControlKind.ORACLE_ROLE,
        {"roles": compiled},
        dict(
            metadata,
            purpose="oracle_upper_bound",
            role_order="T,D,P",
            role_encoding=ROLE_ENCODING,
            image_order="source,role_rgb",
        ),
    )
