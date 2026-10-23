"""Hashed occupancy/role loading and explicit feature-information ablations."""

from pathlib import Path

import numpy as np

from src.conditioning.features import FEATURE_MODES, FEATURE_SCHEMA, RoleFeatures
from src.core.io import sha256
from src.core.paths import data_path
from src.datasets.scene_manifest import load_oracle_condition


def load_features(root: Path, row: dict, mode: str) -> RoleFeatures:
    if mode not in FEATURE_MODES:
        raise ValueError("Unknown feature ablation")
    oracle = load_oracle_condition(root, row)
    path = data_path(root, row["evidence_path"])
    if sha256(path) != row["evidence_sha256"]:
        raise ValueError("Feature evidence SHA256 changed")
    with np.load(path, allow_pickle=False) as evidence:
        old, new = [evidence[k] for k in ("source_occupancy", "target_occupancy")]
    labels = np.asarray(oracle.maps["roles"])
    if any(
        a.shape != labels.shape or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1
        for a in (old, new)
    ):
        raise ValueError("Occupancy shape/range differs from role truth")
    valid = labels != 255
    old, new = (old > 0.5) & valid, (new > 0.5) & valid
    if mode == "constant":
        values = np.ones((5, *labels.shape), np.float32)
    elif mode == "binary":
        union = np.isin(labels, [0, 1])
        values = np.stack([union] * 4 + [valid]).astype(np.float32)
    elif mode == "old_new":
        values = np.stack([old, new, valid, valid, valid]).astype(np.float32)
    else:
        values = np.stack([old, new, labels == 1, labels == 2, valid]).astype(np.float32)
    return RoleFeatures(
        values,
        mode,
        dict(
            oracle.metadata,
            feature_schema=FEATURE_SCHEMA,
            feature_order="old,new,D,P,valid",
            image_order="source",
            role_encoding="independent_features_v1",
        ),
    )
