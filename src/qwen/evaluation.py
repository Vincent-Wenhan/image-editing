"""Target-aware synthetic evaluation kept separate from Qwen inference inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.scene_pairs import rgb_image
from src.datasets.scene_validation import checked_image
from src.evaluation.role_failures import geometry_metrics
from src.evaluation.role_metrics import aggregate, region_metrics
from src.evaluation.scene_semantics import TranslationLocator, change_proxies
from src.qwen.data import verify_condition_manifest

SEMANTIC_METRICS = (
    "old_only_target_mae",
    "old_only_source_distance",
    "new_only_target_mae",
    "new_only_source_distance",
    "target_displacement_px",
    "position_error_px",
    "position_error_extents",
    "distance_to_old_px",
    "d_shadow_removal_mae",
    "d_shadow_creation_mae",
    "d_signed_change_response",
    "d_dark_region_iou",
    "p_edge_mae",
)


def _image(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != size:
            raise ValueError(f"Prediction geometry/mode differs: {path}")
        return np.asarray(image, dtype=np.float32) / 255.0


def _semantic_metrics(
    root: Path,
    row: dict,
    source: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Compute evaluation-only geometry and shadow proxies from frozen evidence."""
    evidence_path = data_path(root, row["evidence_path"])
    if sha256(evidence_path) != row["evidence_sha256"]:
        raise ValueError(f"Qwen evidence provenance changed: {row['id']}")
    with np.load(evidence_path, allow_pickle=False) as evidence:
        old = np.asarray(evidence["source_occupancy"]) > 0.5
        new = np.asarray(evidence["target_occupancy"]) > 0.5
        background = np.asarray(rgb_image(evidence["background"]), dtype=np.float32) / 255.0
    locator = TranslationLocator(source, target, background, old, new)
    located = locator.locate(prediction)
    return {
        **geometry_metrics(source, target, prediction, {"old": old & valid, "new": new & valid}),
        **change_proxies(source, target, prediction, background, labels, valid),
        "target_displacement_px": float(np.linalg.norm(locator.new - locator.old)),
        "position_error_px": located["position_error_px"],
        "position_error_extents": located["position_error_extents"],
        "distance_to_old_px": located["distance_to_old_px"],
        "localization_status": located["status"],
        "localization": located,
    }


def _scene_mean(rows: list[dict], key: str) -> float | None:
    groups = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            groups.setdefault(row["group_id"], []).append(value)
    return float(np.mean([np.mean(values) for values in groups.values()])) if groups else None


def _semantic_summary(rows: list[dict]) -> dict:
    return {
        **{key: _scene_mean(rows, key) for key in SEMANTIC_METRICS},
        "localized": sum(row["localization_status"] == "localized" for row in rows),
        "abstained": sum(row["localization_status"] == "abstain" for row in rows),
    }


def run(args) -> None:
    run_root = Path(args.run).resolve()
    predictions = Path(args.predictions).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new Qwen score directory: {output}")
    inference = json.loads((predictions / "inference.json").read_text(encoding="utf-8"))
    if inference.get("status") != "complete":
        raise ValueError("Qwen scoring requires a completed inference record")
    record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    root = Path(record["data_root"])
    manifest = Path(record["dev_manifest"])
    verify_condition_manifest(manifest, root=root, lock_path=args.lock)
    rows = {row["id"]: row for row in read_rows(root / "dev/manifest.csv")}
    items = inference.get("results")
    if items is None:
        items = [inference]
    if not items:
        raise ValueError("Qwen inference has no prediction results")
    scored = []
    seen = set()
    for item in items:
        sample_id = item["sample_id"]
        if sample_id in seen or sample_id not in rows:
            raise ValueError(f"Unknown or duplicate Qwen Dev prediction: {sample_id}")
        seen.add(sample_id)
        row = rows[sample_id]
        size = (int(row["width"]), int(row["height"]))
        source_path = data_path(root, row["source_path"])
        target_path = data_path(root, row["target_path"])
        if sha256(source_path) != item["source_sha256"]:
            raise ValueError(f"Qwen source provenance changed: {sample_id}")
        if sha256(target_path) != item["target_sha256"]:
            raise ValueError(f"Qwen target provenance changed: {sample_id}")
        prediction_path = predictions / f"{sample_id}.png"
        if sha256(prediction_path) != item["prediction_sha256"]:
            raise ValueError(f"Qwen prediction checksum changed: {sample_id}")
        source, target, prediction = (
            _image(source_path, size),
            _image(target_path, size),
            _image(prediction_path, size),
        )
        labels = np.asarray(checked_image(root, row, "role"))
        valid = np.asarray(checked_image(root, row, "valid")) > 0
        scored.append(
            {
                "id": sample_id,
                "group_id": row["group_id"],
                "strength": float(row["strength"]),
                **region_metrics(source, target, prediction, labels, valid),
                **_semantic_metrics(root, row, source, target, prediction, labels, valid),
            }
        )
    result = {
        "schema_version": "qwen_synthetic_metrics_v1",
        "run": str(run_root),
        "predictions": str(predictions),
        "condition": record.get("condition"),
        "records": scored,
        "all": aggregate(scored),
        "nonzero": aggregate([row for row in scored if row["strength"] > 0]),
        "by_strength": {
            str(strength): aggregate([row for row in scored if row["strength"] == strength])
            for strength in sorted({row["strength"] for row in scored})
        },
        "semantic": {
            "all": _semantic_summary(scored),
            "nonzero": _semantic_summary([row for row in scored if row["strength"] > 0]),
            "by_strength": {
                str(strength): _semantic_summary(
                    [row for row in scored if row["strength"] == strength]
                )
                for strength in sorted({row["strength"] for row in scored})
            },
        },
        "scope": "synthetic pixel diagnostics; no semantic success claim",
    }
    output.mkdir(parents=True)
    write_json(output / "metrics.json", result)
    print(result)
