"""Bridge canonical project manifests to the official Inter-Edit JSON schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.config import load_assets
from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.validation import dataset_splits, validate, verify_images
from src.qwen.provenance import INTER_EDIT_REPOSITORY

SCHEMA = "qwen_cjt_manifest_v1"
SUPPORTED_SPLITS = ("train", "dev")
CONTROL_ORDER = ("original_image_path", "mask_image_path")


def _rows(root: Path, split: str, limit: int | None) -> list[dict[str, str]]:
    root = root.resolve()
    if split not in SUPPORTED_SPLITS:
        raise ValueError("Qwen training and validation manifests may only use train or dev")
    if "test" in dataset_splits(root):
        raise ValueError("Qwen pilot refuses a dataset that exposes the sealed Test split")
    validate(root)
    rows = read_rows(root / split / "manifest.csv")
    if limit is not None and (limit < 1 or limit > len(rows)):
        raise ValueError(f"limit must be in [1, {len(rows)}]")
    selected = rows[:limit] if limit is not None else rows
    for row in selected:
        if row["split"] != split:
            raise ValueError(f"Manifest row {row['id']} is assigned to {row['split']}")
        verify_images(root, row)
        caption = root / split / "target" / f"{row['id']}.txt"
        if caption.read_text(encoding="utf-8").strip() != row["instruction"].strip():
            raise ValueError(f"Caption mismatch: {caption}")
        _check_paths(root, row)
    return selected


def _check_paths(root: Path, row: dict[str, str]) -> None:
    paths = {key: data_path(root, row[f"{key}_path"]) for key in ("source", "target", "mask")}
    if len(set(paths.values())) != 3:
        raise ValueError(f"Source, target and mask paths must be distinct: {row['id']}")
    if paths["target"] in (paths["source"], paths["mask"]):
        raise ValueError(f"Target cannot be a Qwen control input: {row['id']}")


def _sample(row: dict[str, str]) -> dict[str, Any]:
    """Return only the fields consumed by Inter-Edit plus a stable sample ID."""
    return {
        "id": row["id"],
        "original_image_path": row["source_path"],
        "edited_image_path": row["target_path"],
        "mask_image_path": row["mask_path"],
        "instruction": row["instruction"],
    }


def provenance_path(manifest: str | Path) -> Path:
    manifest = Path(manifest)
    return manifest.with_name(f"{manifest.stem}.provenance.json")


def write_manifest(
    root: str | Path,
    split: str,
    output: str | Path,
    *,
    limit: int | None = None,
    lock_path: str | Path = "configs/assets.lock.json",
    external_revision: str | None = None,
) -> dict:
    """Write one train/dev JSON manifest and an immutable audit sidecar."""
    root = Path(root).resolve()
    output = Path(output).resolve()
    sidecar = provenance_path(output)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Qwen manifest already exists: {output}")
    rows = _rows(root, split, limit)
    samples = [_sample(row) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, samples)
    assets = load_assets(lock_path)
    revision = external_revision or assets["repositories"][INTER_EDIT_REPOSITORY]["revision"]
    record = {
        "schema_version": SCHEMA,
        "purpose": "qwen_inter_edit_cjt",
        "data_root": str(root),
        "split": split,
        "selection": "canonical manifest order" if limit is None else f"first {limit}",
        "ids": [row["id"] for row in rows],
        "dataset_json_sha256": sha256(root / "dataset.json"),
        "canonical_manifest_sha256": sha256(root / split / "manifest.csv"),
        "json_sha256": sha256(output),
        "control_order": list(CONTROL_ORDER),
        "target_field": "edited_image_path",
        "inter_edit_revision": revision,
        "qwen_model": assets["models"]["qwen_image_edit_2511"],
        "samples": [
            {
                "id": row["id"],
                "source_sha256": row["source_sha256"],
                "target_sha256": row["target_sha256"],
                "mask_sha256": row["mask_sha256"],
            }
            for row in rows
        ],
    }
    write_json(sidecar, record)
    verify_manifest(output, root=root, lock_path=lock_path)
    return record


def verify_manifest(
    manifest: str | Path,
    *,
    root: str | Path | None = None,
    lock_path: str | Path = "configs/assets.lock.json",
) -> dict:
    """Recheck the bridge, canonical files and target/control separation."""
    manifest = Path(manifest).resolve()
    sidecar = provenance_path(manifest)
    if not manifest.exists() or not sidecar.exists():
        raise FileNotFoundError(
            f"Qwen manifest and provenance sidecar are both required: {manifest}"
        )
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    data_root = Path(record["data_root"]).resolve()
    if root is not None and data_root != Path(root).resolve():
        raise ValueError("Qwen manifest data root changed")
    if (
        record.get("schema_version") != SCHEMA
        or record.get("split") not in SUPPORTED_SPLITS
        or record.get("json_sha256") != sha256(manifest)
        or record.get("dataset_json_sha256") != sha256(data_root / "dataset.json")
        or record.get("canonical_manifest_sha256")
        != sha256(data_root / record["split"] / "manifest.csv")
        or tuple(record.get("control_order", ())) != CONTROL_ORDER
        or record.get("target_field") != "edited_image_path"
    ):
        raise ValueError("Qwen manifest provenance changed")
    rows = _rows(data_root, record["split"], len(record["ids"]))
    if [row["id"] for row in rows] != record["ids"]:
        raise ValueError("Qwen manifest selection differs from the canonical manifest")
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or len(samples) != len(rows):
        raise ValueError("Qwen manifest must be a JSON list with the recorded sample count")
    for row, sample in zip(rows, samples, strict=True):
        expected = _sample(row)
        if sample != expected:
            raise ValueError(f"Qwen sample differs from the canonical row: {row['id']}")
        controls = [sample[key] for key in CONTROL_ORDER]
        if sample["edited_image_path"] in controls:
            raise ValueError(f"Target leaked into Qwen controls: {row['id']}")
    expected_samples = [
        {
            "id": row["id"],
            "source_sha256": row["source_sha256"],
            "target_sha256": row["target_sha256"],
            "mask_sha256": row["mask_sha256"],
        }
        for row in rows
    ]
    if record.get("samples") != expected_samples:
        raise ValueError("Qwen sample checksums differ from the canonical manifest")
    assets = load_assets(lock_path)
    if record.get("qwen_model") != assets["models"]["qwen_image_edit_2511"]:
        raise ValueError("Qwen model lock changed")
    expected_revision = assets["repositories"][INTER_EDIT_REPOSITORY]["revision"]
    if record.get("inter_edit_revision") != expected_revision:
        raise ValueError("Inter-Edit revision changed")
    return record
