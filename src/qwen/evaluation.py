"""Target-aware synthetic evaluation kept separate from Qwen inference inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.scene_validation import checked_image
from src.evaluation.role_metrics import aggregate, region_metrics
from src.qwen.data import verify_condition_manifest


def _image(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != size:
            raise ValueError(f"Prediction geometry/mode differs: {path}")
        return np.asarray(image, dtype=np.float32) / 255.0


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
        "scope": "synthetic pixel diagnostics; no semantic success claim",
    }
    output.mkdir(parents=True)
    write_json(output / "metrics.json", result)
    print(result)
