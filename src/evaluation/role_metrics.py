"""Region-separated synthetic diagnostics; pixel errors are not semantic success rates."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.io import sha256, write_json, write_rows
from src.core.paths import data_path
from src.datasets.scene_manifest import load_role_supervision
from src.datasets.scene_validation import validate_scene_dataset
from src.evaluation.metrics import load_image
from src.evaluation.provenance import verify_comparison, verify_run

METRICS = (
    "t_target_mae",
    "d_target_mae",
    "p_source_drift",
    "identity_target_mae",
    "identity_t_target_mae",
    "identity_d_target_mae",
    "target_mae",
    "changed_fraction",
)


def region_metrics(source, target, prediction, labels, valid) -> dict:
    if (
        source.shape != target.shape
        or target.shape != prediction.shape
        or labels.shape != source.shape[:2]
        or valid.shape != labels.shape
    ):
        raise ValueError("Role diagnostic geometry differs")
    if not all(np.isfinite(x).all() for x in (source, target, prediction)):
        raise ValueError("Non-finite evaluation images")

    def mean(values, selected):
        return float(values[selected].mean()) if selected.any() else None

    result = {}
    for label, name in ((0, "t_target_mae"), (1, "d_target_mae"), (2, "p_source_drift")):
        selected = (labels == label) & valid
        reference = source if label == 2 else target
        result[name] = mean(np.abs(prediction - reference), selected)
        if label < 2:
            result[f"identity_{name}"] = mean(np.abs(source - target), selected)
    result.update(
        identity_target_mae=float(np.abs(source - target).mean()),
        target_mae=float(np.abs(prediction - target).mean()),
        changed_fraction=float((np.abs(prediction - source).mean(axis=-1) > 0.05).mean()),
    )
    return result


def aggregate(rows: list[dict]) -> dict:
    """Average strengths within scene first, then give each scene equal weight."""
    result = dict(samples=len(rows), scenes=len({row["group_id"] for row in rows}))
    for metric in METRICS:
        scenes = []
        for group in {row["group_id"] for row in rows}:
            values = [r[metric] for r in rows if r["group_id"] == group and r[metric] is not None]
            if values:
                scenes.append(float(np.mean(values)))
        result[metric] = float(np.mean(scenes)) if scenes else None
    return result


def score_run(root: Path, manifest: Path, predictions: Path) -> dict:
    record = json.loads((predictions / "run.json").read_text())
    canonical = verify_run(root, manifest, record)
    rows, hashes = [], {}
    for item in record["results"]:
        row = canonical[item["id"]]
        size = (item["width"], item["height"])
        if size != (int(row["width"]), int(row["height"])):
            raise ValueError("Synthetic role evaluation requires exact render size")
        with Image.open(predictions / f"{row['id']}.png") as image:
            if image.mode != "RGB" or image.size != size:
                raise ValueError("Actual prediction mode/geometry differs from saved run")
        supervision = load_role_supervision(root, row)
        source, target, prediction = (
            load_image(data_path(root, row["source_path"]), size),
            load_image(data_path(root, row["target_path"]), size),
            load_image(predictions / f"{row['id']}.png", size),
        )
        labels = np.asarray(supervision.role_labels)
        valid = np.asarray(supervision.valid_region) > 0
        rows.append(
            dict(
                id=row["id"],
                group_id=row["group_id"],
                strength=float(row["strength"]),
                **region_metrics(source, target, prediction, labels, valid),
            )
        )
        hashes[row["id"]] = sha256(predictions / f"{row['id']}.png")
    return dict(
        run_sha256=sha256(predictions / "run.json"),
        prediction_sha256=hashes,
        records=rows,
        all=aggregate(rows),
        nonzero=aggregate([r for r in rows if r["strength"] > 0]),
        by_strength={
            str(s): aggregate([r for r in rows if r["strength"] == s])
            for s in sorted({r["strength"] for r in rows})
        },
    )


def paired_scene_ci(
    left: list[dict], right: list[dict], metric: str, seed: int = 42, *, resamples: int = 2000
) -> dict:
    if type(resamples) is not int or resamples < 1:
        raise ValueError("Bootstrap resamples must be a positive integer")
    indexed = {row["id"]: row for row in right}
    if set(indexed) != {row["id"] for row in left}:
        raise ValueError("Paired role cohorts differ")
    groups = {}
    for row in left:
        other = indexed[row["id"]]
        if (row["group_id"], row["strength"]) != (other["group_id"], other["strength"]):
            raise ValueError("Paired scene identities differ")
        if row[metric] is not None and other[metric] is not None:
            groups.setdefault(row["group_id"], []).append(row[metric] - other[metric])
    values = np.array([np.mean(group) for _, group in sorted(groups.items())])
    rng = np.random.default_rng(seed)
    interval = None
    if len(values) >= 2:
        boot = values[rng.integers(len(values), size=(resamples, len(values)))].mean(axis=1)
        interval = np.quantile(boot, [0.025, 0.975]).tolist()
    return dict(
        mean_difference=float(values.mean()) if len(values) else None,
        ci95=interval,
        scenes=len(values),
        scope="within-training-seed scene bootstrap, not training-seed uncertainty",
    )


def summarize_study(study: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((study / "plan.json").read_text())
    validate_scene_dataset(Path(plan["data"]))
    result = dict(
        status="complete",
        human_status="pending",
        scope="synthetic pixel diagnostics; no semantic success claim",
        seeds={},
    )
    output.mkdir(parents=True)
    for seed in plan["protocol"]["train_seeds"]:
        for split in plan["protocol"]["inference"]["splits"]:
            methods, reference = {}, None
            for job in plan["jobs"]:
                if job["phase"] != "train" or job["seed"] != seed:
                    continue
                directory = Path(job["directory"])
                predictions = directory / f"{split}_inference"
                record = json.loads((predictions / "run.json").read_text())
                training = json.loads((directory / "train_report.json").read_text())
                if record.get("lora_sha256") != training["checkpoint_sha256"]:
                    raise ValueError("Evaluated adapter differs from the matched trained method")
                if reference:
                    verify_comparison(reference, record)
                reference = record
                scored = score_run(
                    directory / "data", directory / "data" / split / "manifest.csv", predictions
                )
                methods[job["method"]] = scored
                write_rows(output / f"{seed}_{split}_{job['method']}.csv", scored["records"])
            paired = []
            for left, right in combinations(sorted(methods), 2):
                pair = dict(left=left, right=right, direction="left minus right; lower is better")
                for metric in ("t_target_mae", "d_target_mae", "p_source_drift"):
                    cohorts = [
                        [r for r in methods[m]["records"] if r["strength"] > 0]
                        for m in (left, right)
                    ]
                    pair[metric] = paired_scene_ci(*cohorts, metric)
                paired.append(pair)
            result["seeds"].setdefault(str(seed), {})[split] = dict(
                methods=methods, paired_nonzero=paired
            )
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(summarize_study(args.study, args.output))


if __name__ == "__main__":
    main()
