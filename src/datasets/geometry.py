"""Read hashed occupancy arrays only, never target/counterfactual RGB, for an oracle."""

from pathlib import Path

import numpy as np

from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.geometry import GEOMETRY_ENCODING
from src.core.io import sha256
from src.core.paths import data_path
from src.datasets.scene_manifest import load_oracle_condition


def load_geometry_condition(root: Path, row: dict) -> ControlBundle:
    oracle = load_oracle_condition(root, row)
    path = data_path(root, row["evidence_path"])
    if sha256(path) != row["evidence_sha256"]:
        raise ValueError("Geometry evidence SHA256 changed")
    # np.load is lazy: selecting these two keys does not decompress RGB arrays.
    with np.load(path, allow_pickle=False) as evidence:
        old, new = [evidence[key] for key in ("source_occupancy", "target_occupancy")]
    labels = np.asarray(oracle.maps["roles"])
    if any(
        value.shape != labels.shape
        or not np.isfinite(value).all()
        or value.min() < 0
        or value.max() > 1
        for value in (old, new)
    ):
        raise ValueError("Occupancy must match labels and be finite in [0,1]")
    return ControlBundle(
        ControlKind.ORACLE_GEOMETRY,
        {
            "old": (old > 0.5) & (labels == 0),
            "new": (new > 0.5) & (labels == 0),
            "dependency": labels == 1,
        },
        dict(oracle.metadata, role_encoding=GEOMETRY_ENCODING, image_order="source,geometry_rgb"),
    )
