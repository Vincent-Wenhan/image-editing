"""Separate palette sensitivity, old-region retention and new-region error diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.io import read_rows, sha256, write_json, write_rows
from src.datasets.geometry import load_geometry_condition
from src.datasets.holdout import require_dev
from src.datasets.scene_manifest import load_role_supervision
from src.datasets.validation import verify_images
from src.evaluation.role_metrics import region_metrics, score_run


def pixels(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != size:
            raise ValueError("Diagnostic image RGB/geometry changed")
        return np.asarray(image, dtype=np.float32) / 255


def geometry_metrics(source, target, prediction, maps: dict) -> dict:
    """Old/new errors are proxies; none of these values claims an object count."""
    old, new = maps["old"], maps["new"]
    result = {}
    for name, selected in (("old_only", old & ~new), ("new_only", new & ~old)):
        if not selected.any():
            result[name + "_target_mae"] = result[name + "_source_distance"] = None
            continue
        result[name + "_target_mae"] = float(np.abs(prediction - target)[selected].mean())
        result[name + "_source_distance"] = float(np.abs(prediction - source)[selected].mean())
    return result


def palette_response(original, changed, control, changed_control, selected) -> float | None:
    delta = (changed_control - control)[selected]
    denominator = float(np.square(delta).sum())
    if not selected.any() or denominator == 0:
        return None
    return float(((changed - original)[selected] * delta).sum() / denominator)


def analyze(root: Path, cohort: Path, diagnostics: dict[str, Path], output: Path) -> dict:
    require_dev(root, cohort, root / "dev/manifest.csv")
    if output.exists():
        raise FileExistsError(output)
    rows = {r["id"]: r for r in read_rows(root / "dev/manifest.csv")}
    result = dict(
        status="complete",
        scope="pixel proxies and OOD sensitivity; human pending",
        cohort_sha256=sha256(cohort),
        methods={},
    )
    for method, directory in diagnostics.items():
        report = json.loads((directory / "report.json").read_text())
        run = json.loads((directory / "baseline/run.json").read_text())
        if (
            report["status"] != "complete"
            or report["cohort_sha256"] != sha256(cohort)
            or sha256(run["config"]["lora"]) != report["lora_sha256"]
            or run["lora_sha256"] != report["lora_sha256"]
        ):
            raise ValueError("Diagnostic adapter/cohort provenance changed")
        scored, records = score_run(root, root / "dev/manifest.csv", directory / "baseline"), []
        for entry in report["results"]:
            row = rows[entry["id"]]
            verify_images(root, row)
            folder = directory / row["id"] / entry["variant"]
            for key in ("control", "prediction"):
                if sha256(folder / f"{key}.png") != entry[key + "_sha256"]:
                    raise ValueError("Diagnostic image SHA256 changed")
            size = int(row["width"]), int(row["height"])
            source, target, prediction = [
                pixels(path, size)
                for path in (
                    root / row["source_path"],
                    root / row["target_path"],
                    folder / "prediction.png",
                )
            ]
            supervision = load_role_supervision(root, row)
            labels, valid = (
                np.asarray(supervision.role_labels),
                np.asarray(supervision.valid_region) > 0,
            )
            geometry = load_geometry_condition(root, row)
            old, new = geometry.maps["old"], geometry.maps["new"]
            # A union is mathematically unable to identify which footprint was old.
            union_invariant = np.array_equal(old | new, new | old)
            original_folder = directory / row["id"] / "original"
            baseline, control, changed_control = [
                pixels(path, size)
                for path in (
                    original_folder / "prediction.png",
                    original_folder / "control.png",
                    folder / "control.png",
                )
            ]
            records.append(
                dict(
                    id=row["id"],
                    group_id=row["group_id"],
                    strength=float(row["strength"]),
                    variant=entry["variant"],
                    union_swap_invariant=union_invariant,
                    prediction_sha256=entry["prediction_sha256"],
                    palette_response={
                        name: palette_response(
                            baseline,
                            prediction,
                            control,
                            changed_control,
                            (labels == label) & valid,
                        )
                        for label, name in enumerate(("T", "D", "P"))
                    },
                    **region_metrics(source, target, prediction, labels, valid),
                    **geometry_metrics(source, target, prediction, geometry.maps),
                )
            )
        result["methods"][method] = dict(
            report_sha256=sha256(directory / "report.json"), baseline=scored, interventions=records
        )
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    for method, values in result["methods"].items():
        write_rows(
            output / f"{method}.csv",
            [
                dict(r, palette_response=json.dumps(r["palette_response"]))
                for r in values["interventions"]
            ],
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--diagnostic", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from src.evaluation.fairness import pairs

    analyze(args.data, args.cohort, pairs(args.diagnostic), args.output)


if __name__ == "__main__":
    main()
