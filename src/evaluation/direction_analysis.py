"""Pair frozen directions by scene; keep denoising, semantic increment and CFG separate."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.core.io import sha256, write_json, write_rows
from src.evaluation.direction_gates import average, collapse_times, slice_gates
from src.evaluation.direction_metrics import BRANCHES, bf16_float, cfg_effect, score, velocities
from src.evaluation.direction_provenance import arrays, verify_report
from src.evaluation.role_metrics import paired_scene_ci
from src.inference.direction_study import load_direction
from src.infra.experiment import source_snapshot


def score_report(report, plan):
    settings = plan["inherited"]["inference"]
    floor = plan["protocol"]["gate"]["energy_floor"]
    identity = {
        (r["group_id"], r["state"]["family"], r["state"]["time"]): r
        for r in report["results"]
        if r["strength"] == 0
    }
    rows = []
    measurements = []
    for case in report["results"]:
        group, state = report["groups"][case["group_id"]], case["state"]
        reference = identity[(case["group_id"], state["family"], state["time"])]
        inputs = arrays(group["input_path"])
        x = arrays(state["path"])["x"]
        supervision = arrays(group["supervision"][case["id"]]["path"])
        original = velocities(arrays(case["variants"]["original"]["path"]), settings["guidance"])
        zero = velocities(arrays(case["variants"]["zero"]["path"]), settings["guidance"])
        drop = velocities(arrays(case["variants"]["drop_d"]["path"]), settings["guidance"])
        raw_original = arrays(case["variants"]["original"]["path"])
        for variant, receipt in case["variants"].items():
            raw = arrays(receipt["path"])
            current = velocities(raw, settings["guidance"])
            baseline = velocities(
                arrays(reference["variants"][variant]["path"]), settings["guidance"]
            )
            support = np.abs(raw["residual"] - raw_original["residual"]) > floor
            lost = (
                float(
                    (
                        bf16_float(raw["fused"])[support]
                        == bf16_float(raw_original["fused"])[support]
                    ).mean()
                )
                if support.any()
                else None
            )
            delta = {b: current[b] - baseline[b] for b in BRANCHES}
            measurements.append(
                dict(
                    id=case["id"],
                    group_id=case["group_id"],
                    split=case["split"],
                    strength=case["strength"],
                    family=report["job"]["family"],
                    method=report["job"]["method"],
                    seed=report["job"]["seed"],
                    state_family=state["family"],
                    time=state["time"],
                    variant=variant,
                    role_difference_lost_fraction=lost,
                    **cfg_effect(
                        delta["positive"],
                        delta["negative"],
                        delta["cfg_native"],
                        delta["cfg_fp32"],
                        floor,
                    ),
                    fusion=receipt["fusion"],
                )
            )
            for branch in BRANCHES:
                result = score(
                    current[branch],
                    baseline[branch],
                    zero[branch],
                    original[branch],
                    drop[branch],
                    x,
                    inputs["anchor"],
                    supervision["target"],
                    supervision["masks"],
                    state["actual_time"],
                    floor,
                )
                record = dict(
                    id=case["id"],
                    group_id=case["group_id"],
                    split=case["split"],
                    strength=case["strength"],
                    family=report["job"]["family"],
                    method=report["job"]["method"],
                    seed=report["job"]["seed"],
                    state_family=state["family"],
                    time=state["time"],
                    actual_time=state["actual_time"],
                    variant=variant,
                    branch=branch,
                )
                for region, r in result.items():
                    record.update(
                        {
                            f"{region}_{k}": v
                            for k, v in r.items()
                            if k not in ["zero_direction", "drop_d_direction"]
                        }
                    )
                    record.update({f"{region}_zero_{k}": v for k, v in r["zero_direction"].items()})
                    record.update(
                        {
                            f"{region}_drop_d_{k}": v
                            for k, v in r["drop_d_direction"].items()
                            if k in ["cosine", "projection"]
                        }
                    )
                rows.append(record)
    return rows, measurements


def summarize(rows):
    keys = [k for k in rows[0] if k.startswith(("t_", "d_", "p_"))]
    summaries = []
    group_keys = ["family", "seed", "split", "method", "variant", "branch", "state_family", "time"]
    selected = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        selected.setdefault(key, []).append(row)
    for key, values in sorted(selected.items()):
        nonzero = [r for r in values if r["strength"] > 0]
        summaries.append(
            dict(
                zip(group_keys, key, strict=True),
                samples=len(values),
                scenes=len({r["group_id"] for r in values}),
                metrics={k: average(nonzero, k) for k in keys},
                coverage={k: sum(r[k] is not None for r in nonzero) for k in keys},
                identity_p_mse=average([r for r in values if r["strength"] == 0], "p_endpoint_mse"),
            )
        )
    return summaries


def analyze(study, attempt, output):
    if output.exists():
        raise FileExistsError(output)
    plan = load_direction(study)
    phase = json.loads((study / f"diagnose_phase_{attempt:02d}.json").read_text())
    if phase["status"] != "complete" or set(phase["jobs"]) != {j["name"] for j in plan["jobs"]}:
        raise ValueError("Require the complete paired frozen comparison")
    rows, mechanisms, reports = [], [], {}
    conditions = {}
    supervision = {}
    sources = []

    def score_job(job):
        receipt = phase["jobs"][job["name"]]
        path = Path(receipt["path"])
        if receipt["status"] != "complete" or sha256(path) != receipt["sha256"]:
            raise ValueError("Phase report identity changed")
        report, contexts, targets = verify_report(study, plan, job, path)
        scored, measured = score_report(report, plan)
        return job, path, report, contexts, targets, scored, measured

    with ThreadPoolExecutor(max_workers=plan["inherited"]["max_parallel"]) as pool:
        completed = list(pool.map(score_job, plan["jobs"]))
    for job, path, report, contexts, targets, scored, measured in completed:
        sources.append(report["source"])
        for key, value in contexts.items():
            if key in conditions and conditions[key] != value:
                raise ValueError(
                    "Methods/checkpoints have different source, prompt, noise or anchor"
                )
            conditions[key] = value
        for key, value in targets.items():
            if key in supervision and supervision[key] != value:
                raise ValueError("Supervision changes across frozen checkpoints")
            supervision[key] = value
        rows.extend(scored)
        mechanisms.extend(measured)
        reports[job["name"]] = dict(
            path=str(path),
            sha256=sha256(path),
            seconds=report["seconds"],
            peak_allocated_gib=report["peak_allocated_gib"],
            peak_reserved_gib=report["peak_reserved_gib"],
        )
    if any(s != sources[0] for s in sources) or phase["source"] != sources[0]:
        raise ValueError("Mixed GPU executable source snapshots")
    decisions = {}
    paired = []
    settings = plan["inherited"]["inference"]
    gate = plan["protocol"]["gate"]
    for family in plan["parents"]:
        slices = {}
        for seed in plan["inherited"]["train_seeds"]:
            for branch in ["cfg_native", "cfg_fp32"]:
                for split in ["train", "dev"]:
                    selected = [
                        r
                        for r in rows
                        if r["family"] == family and r["seed"] == seed and r["branch"] == branch
                    ]
                    slices[f"{seed}/{branch}/{split}"] = slice_gates(
                        selected, gate, split, settings["seed"]
                    )
                    for time in plan["protocol"]["times"]:
                        part = [
                            r
                            for r in selected
                            if r["state_family"] == "source_bridge"
                            and r["time"] == time
                            and r["split"] == split
                            and r["variant"] == "original"
                            and r["strength"] > 0
                        ]
                        for method in ["constant", "binary", "old_new"]:
                            for metric in [
                                "t_contrast_gain",
                                "d_contrast_gain",
                                "t_endpoint_mse",
                                "d_endpoint_mse",
                                "p_endpoint_mse",
                            ]:
                                result = paired_scene_ci(
                                    collapse_times(
                                        [r for r in part if r["method"] == "full_role"], metric
                                    ),
                                    collapse_times(
                                        [r for r in part if r["method"] == method], metric
                                    ),
                                    metric,
                                    seed=settings["seed"],
                                    resamples=gate["bootstrap_resamples"],
                                )
                                paired.append(
                                    dict(
                                        family=family,
                                        seed=seed,
                                        branch=branch,
                                        split=split,
                                        time=time,
                                        comparison=f"full_minus_{method}",
                                        metric=metric,
                                        **result,
                                    )
                                )
        decisions[family] = dict(slices=slices, passed=all(v["passed"] for v in slices.values()))
    output.mkdir(parents=True)
    write_rows(output / "metrics.csv", rows)
    result = dict(
        status="complete",
        source=source_snapshot(),
        gpu_source=sources[0],
        phase_sha256=sha256(study / f"diagnose_phase_{attempt:02d}.json"),
        plan_sha256=sha256(study / "plan.json"),
        reports=reports,
        training_updates=0,
        test_status="sealed",
        human_status="pending",
        real_ratings=0,
        conditions_exact_across_methods=True,
        records=rows,
        mechanisms=mechanisms,
        summaries=summarize(rows),
        paired=paired,
        decisions=decisions,
        go_timestep_only=any(v["passed"] for v in decisions.values()),
    )
    result["decision"] = (
        "go_preregister_timestep_only"
        if result["go_timestep_only"]
        else "no_go_timestep_training_without_direction_evidence"
    )
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    analyze(args.study, args.attempt, args.output)


if __name__ == "__main__":
    main()
