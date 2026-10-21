"""Evaluate the exact completed subset using the existing metrics file formats."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.config import ProjectSettings
from src.core.io import write_json, write_rows
from src.core.paths import data_path
from src.evaluation.metrics import load_image, metrics
from src.evaluation.provenance import verify_comparison, verify_run


def run(args):
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    root, predictions = Path(args.data_root), Path(args.predictions)
    run_info = json.loads((predictions / "run.json").read_text())
    rows = verify_run(root, args.manifest, run_info)
    if getattr(args, "reference_run", None):
        reference = json.loads((Path(args.reference_run) / "run.json").read_text())
        verify_run(root, args.manifest, reference)
        verify_comparison(reference, run_info)
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
            load_image(data_path(root, row[f"{key}_path"]), size, mode)
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
    settings = ProjectSettings.load()
    parser.add_argument("--data-root", default=settings.dataset["root"])
    parser.add_argument("--manifest", default=settings.manifest)
    parser.add_argument("--predictions", default="outputs/base_inference")
    parser.add_argument("--reference-run", help="Require identical Base/LoRA sampling and samples")
    parser.add_argument(
        "--threshold", type=float, default=settings.experiment["evaluation_threshold"]
    )
    parser.set_defaults(func=run)
