"""Apply preregistered repair gates and paired scene intervals, keeping humans pending."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.core.io import sha256, write_json
from src.evaluation.role_metrics import paired_scene_ci
from src.inference.feature_diagnostic_study import phase_outputs
from src.inference.feature_repair_study import load_comparison
from src.infra.experiment import source_snapshot


def numeric_gate(methods: dict, gate: dict) -> dict:
    candidate = methods["centered"]
    fraction = gate["edit_relative_improvement"]
    tolerance = gate["preservation_tolerance"]

    def comparison(other, key, improvement=0, slack=0):
        left, right = candidate["nonzero"][key], other["nonzero"][key]
        return bool(
            left is not None
            and right is not None
            and np.isfinite([left, right]).all()
            and left <= right * (1 - improvement) + slack
        )

    original = methods["original"]
    identity = [m["by_strength"]["0.0"]["target_mae"] for m in [candidate, original]]
    criteria = dict(
        t=comparison(original, "t_target_mae", fraction),
        d=comparison(original, "d_target_mae", fraction),
        p=comparison(original, "p_source_drift", slack=tolerance),
        identity=bool(
            None not in identity
            and np.isfinite(identity).all()
            and identity[0] <= identity[1] + tolerance
        ),
    )
    for name in ("zero", "centered_shuffled"):
        other = methods[name]
        criteria[name] = all(
            comparison(other, k) for k in ("t_target_mae", "d_target_mae")
        ) and any(comparison(other, k, fraction) for k in ("t_target_mae", "d_target_mae"))
    return dict(passed=all(criteria.values()), criteria=criteria)


def evaluate(study: Path, summary: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    plan, _ = load_comparison(study)
    result = json.loads(summary.read_text())
    outputs = phase_outputs(study, "compare")
    if (
        result["status"] != "complete"
        or result["phase"] != "compare"
        or result["plan_sha256"] != sha256(study / "plan.json")
        or result["manifest_sha256"] != sha256(Path(plan["data"]) / "dev/manifest.csv")
        or set(result["seeds"]) != set(outputs)
    ):
        raise ValueError("Gate analysis must match the completed frozen comparison")
    record = dict(
        status="complete",
        summary_sha256=sha256(summary),
        source=source_snapshot(),
        plan_sha256=sha256(study / "plan.json"),
        human_status="pending",
        real_ratings=0,
        visual_status="pending_non_blinded_review",
        seeds={},
    )
    for seed, methods in result["seeds"].items():
        report_sha = sha256(outputs[seed] / "report.json")
        for method in methods.values():
            if any(
                r["run_sha256"] != report_sha
                or sha256(r["prediction_path"]) != r["prediction_sha256"]
                for r in method["records"]
            ):
                raise ValueError("Gate summary prediction or report provenance changed")
        intervals = {}
        gate = plan["protocol"]["gate"]
        for baseline in ("original", "zero", "centered_shuffled"):
            paired = {}
            for metric in (
                "t_target_mae",
                "d_target_mae",
                "p_source_drift",
                "target_mae",
                "d_shadow_creation_mae",
                "d_shadow_removal_mae",
                "position_error_extents",
            ):
                strength_zero = metric == "target_mae"
                rows = [
                    [r for r in methods[name]["records"] if (r["strength"] == 0) == strength_zero]
                    for name in ("centered", baseline)
                ]
                paired["identity_mae" if strength_zero else metric] = paired_scene_ci(
                    *rows,
                    metric,
                    seed=plan["protocol"]["inference"]["seed"],
                    resamples=gate["bootstrap_resamples"],
                )
            intervals[baseline] = paired
        record["seeds"][seed] = dict(numeric_gate=numeric_gate(methods, gate), paired=intervals)
    record["numeric_pass"] = all(s["numeric_gate"]["passed"] for s in record["seeds"].values())
    record["decision"] = (
        "pending_visual_and_human" if record["numeric_pass"] else "no_go_frozen_repair"
    )
    write_json(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.study, args.summary, args.output)


if __name__ == "__main__":
    main()
