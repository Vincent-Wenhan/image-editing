"""CPU audit of effective latent supervision on prepared Train/Dev samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.io import read_rows, sha256, write_json
from src.datasets.scene_validation import checked_image
from src.qwen.role_data import _dimensions, _load_evidence
from src.qwen.role_loss import REGION_NAMES, build_region_masks, latent_masks


def run(args) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((Path(args.run) / "run.json").read_text())
    root = Path(record["data_root"])
    selected = json.loads(Path(record[f"{args.split}_manifest"]).read_text())
    canonical = {row["id"]: row for row in read_rows(root / args.split / "manifest.csv")}
    config = json.loads((root / "dataset.json").read_text())["config"]
    threshold = config["roles"]["shadow_high"]
    results = []
    for sample in selected:
        row = canonical[sample["id"]]
        labels = np.asarray(checked_image(root, row, "role"))
        valid = np.asarray(checked_image(root, row, "valid")) > 0
        evidence = _load_evidence(root, row)
        masks = build_region_masks(evidence, labels, valid, shadow_threshold=threshold)
        width, height = _dimensions(1024**2, int(row["width"]) / int(row["height"]))
        latent = latent_masks(masks, height // 8, width // 8)
        source_shadow = (
            np.abs(evidence["source"] - evidence["source_no_shadow"]).max(-1) >= threshold
        )
        target_shadow = (
            np.abs(evidence["target"] - evidence["target_no_shadow"]).max(-1) >= threshold
        )
        directed = masks["d_remove"] | masks["d_create"]
        results.append(
            {
                "id": row["id"],
                "strength": float(row["strength"]),
                "latent_size": [width // 8, height // 8],
                "pixels": {key: int(mask.sum()) for key, mask in masks.items()},
                "latent_areas": {
                    key: float(latent[index].sum()) for index, key in enumerate(REGION_NAMES)
                },
                "new_overlap_shadow_pixels": int((source_shadow & target_shadow & directed).sum()),
                "unassigned_valid_d_pixels": int(((labels == 1) & valid & ~directed).sum()),
                "evidence_sha256": row["evidence_sha256"],
            }
        )
    write_json(
        output / "audit.json",
        {
            "status": "complete",
            "split": args.split,
            "records": results,
            "manifest_sha256": sha256(record[f"{args.split}_manifest"]),
            "scope": "supervision coverage only; no training or edit-quality conclusion",
        },
    )
    print(json.dumps(results, indent=2))


def configure(parser):
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.set_defaults(func=run)
