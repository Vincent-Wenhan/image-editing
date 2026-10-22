"""Minimal oracle geometry encoding; no RGB target or appearance enters these maps."""

import numpy as np
from PIL import Image

from src.core.spatial import array, unit_map

GEOMETRY_ENCODING = "old64_new192_overlap128_dependency255_p0_v1"
GEOMETRY_KEYS = {"old", "new", "dependency"}


def geometry_rgb(maps: dict) -> Image.Image:
    if set(maps) != GEOMETRY_KEYS:
        raise ValueError("Geometry requires old/new/dependency maps")
    sizes = {unit_map(value, key, binary=True) for key, value in maps.items()}
    if len(sizes) != 1:
        raise ValueError("Geometry maps must share dimensions")
    old, new, dependency = [array(maps[k], k) > 0 for k in ("old", "new", "dependency")]
    if ((old | new) & dependency).any():
        raise ValueError("Object and dependency regions must be disjoint")
    values = np.zeros(old.shape, dtype=np.uint8)
    values[old & ~new], values[new & ~old] = 64, 192
    values[old & new], values[dependency] = 128, 255
    return Image.fromarray(values).convert("RGB")
