"""Verify bytes, physical grouping and recomputed causal labels without invoking Blender."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.conditioning.role_truth import derive_roles
from src.core.config import load_yaml
from src.core.io import read_rows, sha256
from src.core.metadata import config_digest
from src.core.paths import data_path
from src.core.types import EditSupervision
from src.datasets.scene_config import scene_group, validate_scene_config
from src.datasets.scene_pairs import instruction, rgb_image
from src.datasets.validation import validate

EVIDENCE_KEYS = {
    "source",
    "target",
    "source_no_shadow",
    "target_no_shadow",
    "background",
    "source_occupancy",
    "target_occupancy",
}


def checked_image(root: Path, row: dict, key: str) -> Image.Image:
    path = data_path(root, row[f"{key}_path"])
    if sha256(path) != row[f"{key}_sha256"]:
        raise ValueError(f"Scene {key} SHA256 changed")
    with Image.open(path) as image:
        image.load()
        if image.mode != "L" or image.size != (int(row["width"]), int(row["height"])):
            raise ValueError("Role/valid image mode or geometry changed")
        return image.copy()


def validate_scene_dataset(root: Path) -> dict:
    dataset = json.loads((root / "dataset.json").read_text())
    if (
        dataset["schema_version"] != 1
        or dataset["kind"] != "blender_translation_v1"
        or dataset["status"] != "complete"
    ):
        raise ValueError("Not a completed supported scene dataset")
    config = dataset["config"]
    validate_scene_config(config)
    if (
        config_digest(config) != dataset["config_sha256"]
        or sha256(root / "scene_config.yaml") != dataset["config_file_sha256"]
        or load_yaml(root / "scene_config.yaml") != config
    ):
        raise ValueError("Scene configuration SHA256 changed")
    raw_path = root / "raw/index.json"
    if sha256(raw_path) != dataset["raw_index_sha256"]:
        raise ValueError("Raw render index SHA256 changed")
    raw = json.loads(raw_path.read_text())
    if (
        raw["schema_version"] != 1
        or raw["config_sha256"] != dataset["config_sha256"]
        or raw["renderer"] != config["renderer"]["version"]
        or raw["device"] != "CPU"
        or [group["scene"] for group in raw["groups"]] != config["scenes"]
    ):
        raise ValueError("Renderer, scene order or physical configuration changed")
    result = validate(root)
    manifests = {}
    for split in ("train", "dev"):
        if sha256(root / split / "manifest.csv") != dataset["manifests"][split]:
            raise ValueError("Scene manifest SHA256 changed")
        manifests[split] = {row["id"]: row for row in read_rows(root / split / "manifest.csv")}
    seen = {"train": set(), "dev": set()}
    statistics = []
    for group in raw["groups"]:
        spec, entries = group["scene"], group["entries"]
        split = spec["split"]
        group_id = scene_group(config, spec)
        if (
            group["group"] != group_id
            or len(entries) != len(config["strengths"])
            or not group["raw_files"]
        ):
            raise ValueError("Scene group or strength pairing changed")
        for relative, digest in group["raw_files"].items():
            if sha256(data_path(root / "raw", relative)) != digest:
                raise ValueError("Raw EXR SHA256 changed")
        source_digest = None
        for index, entry in enumerate(entries):
            identifier = f"{spec['id']}_s{index:03d}"
            row = manifests[split][identifier]
            strength = config["strengths"][index]
            displacement = strength * config["max_translation_widths"] * spec["object_width"]
            if (
                row["scene_id"] != spec["id"]
                or row["group_id"] != group_id
                or row["source_session"] != group_id
                or row["instruction"] != instruction(strength, displacement)
                or float(row["strength"]) != strength
                or entry["strength"] != strength
                or float(row["object_width"]) != spec["object_width"]
                or not math.isclose(float(row["displacement"]), displacement, abs_tol=1e-10)
                or not math.isclose(entry["displacement"], displacement, abs_tol=1e-10)
                or row["evidence_path"] != f"raw/{entry['evidence_path']}"
                or row["evidence_sha256"] != entry["sha256"]
            ):
                raise ValueError("Physical strength, scene identity or evidence pairing changed")
            evidence_path = data_path(root, row["evidence_path"])
            if sha256(evidence_path) != row["evidence_sha256"]:
                raise ValueError("Counterfactual evidence SHA256 changed")
            with np.load(evidence_path, allow_pickle=False) as file:
                if set(file.files) != EVIDENCE_KEYS:
                    raise ValueError("Incomplete counterfactual evidence")
                evidence = dict(file)
            expected, valid = derive_roles(evidence, strength, config["roles"])
            labels = checked_image(root, row, "role")
            valid_image = checked_image(root, row, "valid")
            if not np.array_equal(labels, expected) or not np.array_equal(valid_image, valid):
                raise ValueError("Role labels differ from counterfactual evidence")
            images = {}
            for key in ("source", "target", "mask"):
                with Image.open(data_path(root, row[f"{key}_path"])) as image:
                    images[key] = image.copy()
                expected_image = (
                    rgb_image(evidence[key])
                    if key != "mask"
                    else Image.fromarray(
                        (evidence["source_occupancy"] > 0.5).astype(np.uint8) * 255
                    )
                )
                if not np.array_equal(images[key], expected_image):
                    raise ValueError("Image differs from declared linear render evidence")
            EditSupervision(images["target"], labels, valid_image)
            if source_digest is not None and row["source_sha256"] != source_digest:
                raise ValueError("Strengths in one scene must share the same source")
            source_digest = row["source_sha256"]
            exposed = (
                (evidence["source_occupancy"] > 0.99999)
                & (evidence["target_occupancy"] < 1e-5)
                & (valid > 0)
            )
            background = evidence["background"]
            if background.shape != evidence["source"].shape or not np.isfinite(background).all():
                raise ValueError("Invalid empty-background evidence")
            error = np.abs(evidence["target_no_shadow"] - background)[exposed]
            if strength and (
                not exposed.any()
                or not {0, 1, 2} <= set(np.unique(expected))
                or (error.size and error.max() > config["roles"]["unexplained_tolerance"])
            ):
                raise ValueError(
                    "Translation lacks verified T/D/P or exposed-background completion"
                )
            statistics.append(
                dict(
                    id=identifier,
                    strength=strength,
                    exposed_pixels=int(exposed.sum()),
                    background_max_error=float(error.max()) if error.size else 0,
                    role_pixels={
                        name: int((expected == label).sum())
                        for name, label in (("T", 0), ("D", 1), ("P", 2), ("ignore", 255))
                    },
                )
            )
            seen[split].add(identifier)
    if any(seen[split] != set(manifests[split]) for split in seen):
        raise ValueError("Scene manifests contain missing or extra strength samples")
    return dict(
        **result, scene_groups=len(raw["groups"]), scene_group_overlap=0, samples=statistics
    )
