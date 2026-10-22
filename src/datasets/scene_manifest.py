"""Separate synthetic supervision and explicit oracle labels from deployable inputs."""

import json
from pathlib import Path

from PIL import Image

from src.conditioning.controls import ControlBundle
from src.conditioning.oracle import oracle_bundle
from src.core.io import read_rows, sha256
from src.core.paths import data_path
from src.core.types import EditSupervision
from src.datasets.scene_validation import checked_image


def checked_row(root: Path, row: dict) -> None:
    generation = root / "generation.json"
    if generation.exists() and json.loads(generation.read_text())["status"] != "complete":
        raise ValueError("Oracle rejects failed or incomplete scene generation")
    dataset = json.loads((root / "dataset.json").read_text())
    if dataset.get("kind") != "blender_translation_v1" or dataset.get("status") != "complete":
        raise ValueError("Oracle loading requires a declared completed synthetic dataset")
    if row["split"] not in {"train", "dev", "test"} or row["split"] not in dataset["manifests"]:
        raise ValueError("Unknown scene split")
    manifest = root / row["split"] / "manifest.csv"
    if sha256(manifest) != dataset["manifests"][row["split"]]:
        raise ValueError("Oracle manifest SHA256 changed")
    canonical = [item for item in read_rows(manifest) if item["id"] == row["id"]]
    if canonical != [row]:
        raise ValueError("Oracle sample differs from its scene manifest")


def load_role_supervision(root: Path, row: dict) -> EditSupervision:
    checked_row(root, row)
    target_path = data_path(root, row["target_path"])
    if sha256(target_path) != row["target_sha256"]:
        raise ValueError("Target SHA256 changed")
    with Image.open(target_path) as image:
        target = image.copy()
    return EditSupervision(
        target, checked_image(root, row, "role"), checked_image(root, row, "valid")
    )


def load_oracle_condition(root: Path, row: dict) -> ControlBundle:
    """Read source/labels/valid only. Target and counterfactual RGB are never opened."""
    checked_row(root, row)
    if sha256(data_path(root, row["source_path"])) != row["source_sha256"]:
        raise ValueError("Oracle source SHA256 changed")
    return oracle_bundle(
        checked_image(root, row, "role"),
        checked_image(root, row, "valid"),
        metadata={
            key: row[key]
            for key in (
                "role_sha256",
                "valid_sha256",
                "evidence_sha256",
                "source_sha256",
                "scene_id",
                "group_id",
            )
        }
        | dict(sample_id=row["id"], strength=float(row["strength"])),
    )
