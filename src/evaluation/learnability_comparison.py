"""Compare the one registered normalization change at equal actual training budgets."""

import argparse
import copy
import json
import math
from pathlib import Path

from src.core.io import sha256, write_json
from src.evaluation.role_metrics import paired_scene_ci
from src.infra.experiment import source_snapshot
from src.training.learnability_study import load_plan


def repair_gates(original, normalized, gate):
    a, b = original["nonzero"], normalized["nonzero"]

    def bounded(left, right, improvement=0, tolerance=0):
        return bool(
            left is not None
            and right is not None
            and math.isfinite(left)
            and math.isfinite(right)
            and left <= right * (1 - improvement) + tolerance
        )

    return dict(
        t=bounded(b["t_target_mae"], a["t_target_mae"], gate["role_improvement"]),
        d=bounded(b["d_target_mae"], a["d_target_mae"], gate["role_improvement"]),
        p=bounded(b["p_source_drift"], a["p_source_drift"], tolerance=gate["identity_tolerance"]),
        identity=bounded(
            normalized["by_strength"]["0.0"]["target_mae"],
            original["by_strength"]["0.0"]["target_mae"],
            tolerance=gate["identity_tolerance"],
        ),
    )


def compare(study: Path, baseline: Path, normalized_study: Path, normalized: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    plans = [load_plan(path) for path in [study, normalized_study]]
    studies = [json.loads(path.read_text()) for path in [baseline, normalized]]
    p, changed = copy.deepcopy(plans[0]["protocol"]), copy.deepcopy(plans[1]["protocol"])
    if (
        plans[0].get("normalization", "none") != "none"
        or plans[1].get("normalization") != "channel_rms"
        or changed["encoder"].pop("normalization", None) != "channel_rms"
        or p != changed
        or any(
            record["status"] != "complete" or record["plan_sha256"] != sha256(path / "plan.json")
            for record, path in zip(studies, [study, normalized_study], strict=True)
        )
    ):
        raise ValueError("Compare only the registered single normalization change")
    result = dict(
        status="running",
        source=source_snapshot(),
        baseline_sha256=sha256(baseline),
        normalized_sha256=sha256(normalized),
        human_status="pending",
        test_status="sealed",
        seeds={},
    )
    for seed in p["train_seeds"]:
        a, b = [record["seeds"][str(seed)] for record in studies]
        fingerprint = copy.deepcopy(b["fairness"]["effective_recipe"])
        if (
            fingerprint["encoder"].pop("normalization", None) != "channel_rms"
            or fingerprint != a["fairness"]["effective_recipe"]
        ):
            raise ValueError("Repair changed actual samples/noise/initialization/capacity/budget")
        original, fixed = [record["methods"]["train"]["full_role"] for record in [a, b]]
        gates = repair_gates(original, fixed, p["gate"])
        paired = {}
        for split in ["train", "dev"]:
            cohorts = [
                [r for r in record["methods"][split]["full_role"]["records"] if r["strength"] > 0]
                for record in [b, a]
            ]
            paired[split] = {
                metric: paired_scene_ci(
                    *cohorts,
                    metric,
                    seed=p["inference"]["seed"],
                    resamples=p["gate"]["bootstrap_resamples"],
                )
                for metric in [
                    "t_target_mae",
                    "d_target_mae",
                    "p_source_drift",
                    "t_contrast_mae",
                    "d_contrast_mae",
                ]
            }
        result["seeds"][str(seed)] = dict(
            capacity_and_actual_trace_exact=True,
            relative_gates=gates,
            learnability_gates=b["gates"],
            paired_normalized_minus_original=paired,
            passed=all(gates.values()) and b["gates"]["passed"],
        )
    result["passed"] = all(record["passed"] for record in result["seeds"].values())
    result.update(
        status="complete",
        decision="go_controlled_mechanism_only" if result["passed"] else "no_go_channel_rms",
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--normalized-study", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.study, args.baseline, args.normalized_study, args.normalized, args.output)


if __name__ == "__main__":
    main()
