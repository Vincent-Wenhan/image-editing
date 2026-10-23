"""Validate saved-run identity and actual data before calculating any metrics."""

import math
import re
from pathlib import Path

from src.core.images import dimensions
from src.core.io import read_rows, sha256
from src.core.metadata import config_digest
from src.core.random import sample_seed
from src.core.references import reference_preprocessing
from src.datasets.role_controls import ORACLE_KINDS, load_controls
from src.datasets.validation import verify_images

COMPARISON_KEYS = ("side", "steps", "guidance", "seed")


def verify_run(root: Path, manifest: str | Path, record: dict) -> dict[str, dict]:
    if record["status"] != "complete":
        raise ValueError("Cannot evaluate an incomplete inference run")
    if sha256(manifest) != record.get("manifest_sha256"):
        raise ValueError("Manifest SHA256 differs from the inference run")
    config = record["config"]
    features = "feature_mode" in config or "role_features" in record
    if features and ("feature_mode" not in config or "role_features" not in record):
        raise ValueError("Feature inference metadata cannot be omitted")
    if (config.get("control") in ORACLE_KINDS or features) != bool(
        config.get("allow_oracle", False)
    ):
        raise ValueError("Inference oracle upper-bound permission differs from method")
    preprocessing = reference_preprocessing(
        config.get("control", "none"), config.get("reference_preprocessing", "auto")
    )
    if record.get("reference_preprocessing", preprocessing) != preprocessing:
        raise ValueError("Reference preprocessing differs from inference configuration")
    if "config_sha256" in record and config_digest(config) != record["config_sha256"]:
        raise ValueError("Inference configuration SHA256 changed")
    for key in ("limit", "side", "steps", "seed"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool):
            raise ValueError(f"Invalid inference configuration: {key}")
    if min(config["limit"], config["steps"]) < 1 or config["side"] < 16:
        raise ValueError("Invalid inference limit/steps/side")
    if not isinstance(config.get("guidance"), int | float) or not math.isfinite(config["guidance"]):
        raise ValueError("Invalid inference guidance")
    rows = read_rows(manifest)
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in identifiers
    ):
        raise ValueError("Invalid/duplicate manifest sample IDs")
    for row in rows:
        verify_images(root, row)
    condition_data = None
    if config.get("control", "none") != "none":
        condition_data = load_controls(Path(config["control_root"]), root, config["control"])
        if record.get("conditioning") != condition_data.metadata():
            raise ValueError("Inference control index SHA256 or method changed")
    elif record.get("conditioning"):
        raise ValueError("No-mask run contains undeclared conditions")
    selected = rows[: config["limit"]]
    if features:
        from src.evaluation.feature_provenance import verify_features

        verify_features(root, Path(manifest), record, selected)
    if [item["id"] for item in record["results"]] != [row["id"] for row in selected]:
        raise ValueError("Inference sample selection differs from configured manifest subset")
    for row, item in zip(selected, record["results"], strict=True):
        expected_size = dimensions((int(row["width"]), int(row["height"])), config["side"])
        if (item["width"], item["height"]) != expected_size:
            raise ValueError(f"Inference dimensions differ from configuration: {row['id']}")
        if item["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"Source changed: {row['id']}")
        if item["prompt"] != row["instruction"] or item["seed"] != sample_seed(
            config["seed"], row["id"]
        ):
            raise ValueError(f"Inference prompt/seed differs from configuration: {row['id']}")
        if condition_data and item.get("conditioning") != dict(
            condition_data.bundle(row, expected_size).metadata
        ):
            raise ValueError("Inference mask provenance differs from prepared controls")
    return dict(zip(identifiers, rows, strict=True))


def verify_comparison(reference: dict, record: dict) -> None:
    """Compare sampling budget and sample identity, allowing a different LoRA/output path."""
    if reference["status"] != "complete" or record["status"] != "complete":
        raise ValueError("Paired evaluation requires complete runs")
    if reference["manifest_sha256"] != record["manifest_sha256"]:
        raise ValueError("Paired runs use different manifests")
    for key in COMPARISON_KEYS:
        if reference["config"][key] != record["config"][key]:
            raise ValueError(f"Paired inference configuration differs: {key}")
    for key in ("repo_id", "revision"):
        if reference["model"].get(key) != record["model"].get(key):
            raise ValueError(f"Paired base model differs: {key}")
    identity_keys = ("id", "seed", "width", "height", "source_sha256", "prompt")

    def identities(value):
        return [tuple(item[key] for key in identity_keys) for item in value["results"]]

    if identities(reference) != identities(record):
        raise ValueError("Paired runs use different fixed samples")
