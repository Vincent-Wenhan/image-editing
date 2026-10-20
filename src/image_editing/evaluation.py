"""Pixel diagnostics only: these do not measure semantic success or necessity."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from image_editing.common import read_rows, write_json, write_rows


def load_image(path, size, mode="RGB"):
    with Image.open(path) as image:
        image = image.convert(mode)
        interpolation = Image.Resampling.NEAREST if mode == "L" else Image.Resampling.LANCZOS
        return np.asarray(image.resize(size, interpolation), dtype=np.float32) / 255


def metrics(source, target, prediction, mask, threshold):
    drift = np.abs(prediction - source).mean(axis=2)
    outside = mask < 0.5
    return dict(
        target_mae=float(np.abs(prediction - target).mean()),
        identity_target_mae=float(np.abs(source - target).mean()),
        changed_area_fraction=float((drift > threshold).mean()),
        source_drift_mae=float(drift.mean()),
        outside_mask_drift_proxy=float(drift[outside].mean()) if outside.any() else None,
    )


def run(args):
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    root, predictions = Path(args.data_root), Path(args.predictions)
    run_info = json.loads((predictions / "run.json").read_text())
    if run_info["status"] != "complete":
        raise ValueError("Cannot evaluate an incomplete inference run")
    rows = {row["id"]: row for row in read_rows(args.manifest)}
    collected = []
    for item in run_info["results"]:
        row = rows[item["id"]]
        if item["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"Source changed: {row['id']}")
        size = (item["width"], item["height"])
        path = predictions / f"{row['id']}.png"
        with Image.open(path) as prediction_image:
            if prediction_image.size != size:
                raise ValueError(f"Prediction dimensions changed: {path}")
        prediction = load_image(path, size)
        source, target, mask = [
            load_image(root / row[f"{key}_path"], size, mode)
            for key, mode in (("source", "RGB"), ("target", "RGB"), ("mask", "L"))
        ]
        collected.append(
            dict(id=row["id"], **metrics(source, target, prediction, mask, args.threshold))
        )
    if not collected:
        raise ValueError("No completed predictions")
    write_rows(predictions / "metrics.csv", collected)
    summary = dict(
        count=len(collected),
        manifest_count=len(rows),
        threshold=args.threshold,
        scope="exact completed run subset; pixel diagnostics, not edit success",
    )
    for key in collected[0]:
        if key != "id":
            values = [row[key] for row in collected if row[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
    write_json(predictions / "metrics.json", summary)
    print(summary)


def configure(parser):
    parser.add_argument("--data-root", default="data/magicbrush")
    parser.add_argument("--manifest", default="data/magicbrush/dev/manifest.csv")
    parser.add_argument("--predictions", default="outputs/base_inference")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.set_defaults(func=run)
