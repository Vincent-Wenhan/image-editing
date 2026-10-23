"""Feature-study gates and old/new pixel diagnostics, with human outcomes kept pending."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.core.io import read_rows, sha256, write_json, write_rows
from src.datasets.geometry import load_geometry_condition
from src.datasets.holdout import require_dev
from src.evaluation.feature_fairness import audit_seed
from src.evaluation.provenance import verify_comparison
from src.evaluation.role_failures import geometry_metrics, pixels
from src.evaluation.role_metrics import paired_scene_ci, score_run


def numeric_gate(methods: dict) -> dict:
    """Necessary pixel gates; s=0 target MAE measures output identity, not semantic Go."""
    full = methods["full_role"]

    def better(left, right, *, tolerance=0, strict=True):
        if left is None or right is None or not np.isfinite([left, right]).all():
            return False
        return bool(left < right if strict else left <= right + tolerance)

    criteria = {}
    for baseline in ("constant", "binary", "old_new"):
        other = methods[baseline]
        criteria[baseline] = dict(
            t=better(full["nonzero"]["t_target_mae"], other["nonzero"]["t_target_mae"]),
            d=better(full["nonzero"]["d_target_mae"], other["nonzero"]["d_target_mae"]),
            p=better(
                full["nonzero"]["p_source_drift"], other["nonzero"]["p_source_drift"], strict=False
            ),
            identity=better(
                full["by_strength"]["0.0"]["target_mae"],
                other["by_strength"]["0.0"]["target_mae"],
                tolerance=1e-4,
                strict=False,
            ),
        )
    return dict(passed=all(all(c.values()) for c in criteria.values()), criteria=criteria)


def scene_mean(records: list[dict], key: str) -> float | None:
    groups = {}
    for row in records:
        if row[key] is not None:
            groups.setdefault(row["group_id"], []).append(row[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def summarize(study: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((study / "plan.json").read_text())
    root, cohort = Path(plan["data"]), Path(plan["cohort"]["path"])
    require_dev(root, cohort, root / "dev/manifest.csv")
    output.mkdir(parents=True)
    result = dict(
        status="running",
        plan_sha256=sha256(study / "plan.json"),
        seeds={},
        human_status="pending",
        semantic_success=None,
        scope="fixed Dev, scene-equal pixel errors; old/new proxies are not object counts",
    )
    for seed in plan["protocol"]["train_seeds"]:
        jobs = [j for j in plan["jobs"] if j["phase"] == "train" and j["seed"] == seed]
        fairness = audit_seed(jobs)
        methods, reference, geometry = {}, None, {}
        for job in jobs:
            directory = Path(job["directory"])
            predictions = directory / "dev_inference"
            record = json.loads((predictions / "run.json").read_text())
            checked = fairness["methods"][job["method"]]
            if (
                record["lora_sha256"] != checked["lora_sha256"]
                or record["role_features"]["sha256"] != checked["encoder_sha256"]
            ):
                raise ValueError("Evaluation differs from the matched paired checkpoint")
            if reference:
                verify_comparison(reference, record)
            reference = record
            data = directory / "data"
            score = score_run(data, data / "dev/manifest.csv", predictions)
            methods[job["method"]] = score
            records = []
            for row in read_rows(data / "dev/manifest.csv"):
                size = int(row["width"]), int(row["height"])
                source, target, prediction = [
                    pixels(p, size)
                    for p in (
                        data / row["source_path"],
                        data / row["target_path"],
                        predictions / f"{row['id']}.png",
                    )
                ]
                records.append(
                    dict(
                        id=row["id"],
                        group_id=row["group_id"],
                        strength=float(row["strength"]),
                        **geometry_metrics(
                            source, target, prediction, load_geometry_condition(data, row).maps
                        ),
                    )
                )
            nonzero = [r for r in records if r["strength"] > 0]
            geometry[job["method"]] = dict(
                records=records,
                nonzero={
                    key: scene_mean(nonzero, key)
                    for key in (
                        "old_only_target_mae",
                        "old_only_source_distance",
                        "new_only_target_mae",
                        "new_only_source_distance",
                    )
                },
            )
            write_rows(output / f"{seed}_{job['method']}_geometry.csv", records)
        paired = {
            baseline: {
                metric: paired_scene_ci(
                    [r for r in methods["full_role"]["records"] if r["strength"] > 0],
                    [r for r in methods[baseline]["records"] if r["strength"] > 0],
                    metric,
                )
                for metric in ("t_target_mae", "d_target_mae", "p_source_drift")
            }
            for baseline in ("constant", "binary", "old_new")
        }
        result["seeds"][str(seed)] = dict(
            fairness=fairness,
            methods=methods,
            geometry=geometry,
            paired_nonzero=paired,
            numeric_gate=numeric_gate(methods),
        )
    result["numeric_pass"] = all(s["numeric_gate"]["passed"] for s in result["seeds"].values())
    result["decision"] = (
        "pending_visual_and_human" if result["numeric_pass"] else "no_go_current_budget"
    )
    result["status"] = "complete"
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.study, args.output)


if __name__ == "__main__":
    main()
