"""Verify matched overfit runs, score Train/Dev contrasts and apply the preregistered gates."""

import argparse
import json
from pathlib import Path

from src.core.io import sha256, write_json, write_rows
from src.evaluation.feature_analysis import scene_mean
from src.evaluation.feature_diagnostic_analysis import (
    scene_references,
    score_prediction,
    summarize_records,
)
from src.evaluation.feature_fairness import audit_seed
from src.evaluation.learnability_metrics import contrast_metrics, learnability_gates
from src.evaluation.learnability_provenance import verify_inference
from src.evaluation.learnability_training import training_diagnostics
from src.evaluation.role_failures import pixels
from src.evaluation.role_metrics import paired_scene_ci
from src.inference.learnability import evaluation_rows
from src.infra.experiment import source_snapshot
from src.training.learnability_study import load_plan

CONTRASTS = ("t_contrast_mae", "d_contrast_mae", "t_contrast_response", "d_contrast_response")


def mechanism_summary(observations, probes):
    """Keep time, branch and sampled-case scope explicit; gradient norms are not shares."""
    training = {}
    for method, measured in observations.items():
        training[method] = dict(
            loss_first20=measured["loss_first20"],
            loss_last20=measured["loss_last20"],
            category_to_position_mean=sum(
                s["content_to_position"] for s in measured["input_scales"]
            )
            / len(measured["input_scales"]),
            updates=[
                dict(
                    update=u["update"],
                    timestep=u["timestep"],
                    fusion=u["fusion"],
                    parameters=u["parameters"],
                    optimizer_groups=u["optimizer_groups"],
                    layers=u["layers"],
                )
                for u in measured["updates"]
            ],
        )
    states = []
    for case in probes["states"]:
        for state in case["probes"]:
            for branch in ["positive", "negative"]:
                states.append(
                    dict(
                        id=case["id"],
                        split=case["split"],
                        step=state["step"],
                        timestep=state["timestep"],
                        branch=branch,
                        variants={
                            name: dict(
                                role_difference_lost_fraction=r["role_difference_lost_fraction"],
                                residual_vs_original=r["residual_vs_original"],
                                fusion=r["fusion"],
                                velocity_vs_original=r["output_vs_original"],
                                layers=r["layers"],
                                cfg_vs_original=state["cfg"][name],
                            )
                            for name, r in state[branch]["records"].items()
                        },
                    )
                )
    return dict(
        training=training,
        states=states,
        gradients=probes["gradients"],
        scope="First nonzero Train/Dev case only; region gradient norms are not additive fractions",
    )


def summarize_teacher(rows):
    """Aggregate supervised probes independently of already-recorded pixel statistics."""
    result = {}
    for time in sorted({t["time"] for r in rows for t in r.get("teacher", [])}):
        records = [
            dict(
                group_id=r["group_id"],
                velocity_mse=t["velocity_mse"],
                **{f"{k}_velocity_mse": v for k, v in t["regions"].items()},
            )
            for r in rows
            for t in r.get("teacher", [])
            if t["time"] == time
        ]
        result[str(time)] = {
            key: scene_mean(records, key)
            for key in ["velocity_mse", "t_velocity_mse", "d_velocity_mse", "p_velocity_mse"]
        }
    return result


def summarize(rows):
    result = summarize_records(rows)
    result["teacher_preprocessing"] = sorted(
        {r.get("teacher_preprocessing", "unspecified") for r in rows if "teacher" in r}
    )
    selections = dict(all=rows, nonzero=[r for r in rows if r["strength"] > 0])
    for key, selected in selections.items():
        result[key].update({k: scene_mean(selected, k) for k in CONTRASTS})
        result[key]["contrast_samples"] = sum(r["t_contrast_mae"] is not None for r in selected)
    for strength, part in result["by_strength"].items():
        selected = [r for r in rows if r["strength"] == float(strength)]
        part.update({k: scene_mean(selected, k) for k in CONTRASTS})
    result["teacher_by_time"] = summarize_teacher(rows)
    return result


def score_report(report, references):
    identity = {
        (r["update"], r["group_id"], r["variant"]): r
        for r in report["results"]
        if r["strength"] == 0
    }
    groups = {}
    for item in report["results"]:
        row = references[item["id"]]
        prediction = pixels(
            Path(item["path"]), (int(row["row"]["width"]), int(row["row"]["height"]))
        )
        reference = identity.get((item["update"], item["group_id"], item["variant"]))
        contrasts = dict.fromkeys(CONTRASTS)
        if reference is not None and item["strength"] > 0:
            baseline = pixels(
                Path(reference["path"]), (int(row["row"]["width"]), int(row["row"]["height"]))
            )
            contrasts = contrast_metrics(row, prediction, baseline)
        record = dict(
            id=item["id"],
            group_id=item["group_id"],
            strength=item["strength"],
            update=item["update"],
            variant=item["variant"],
            prediction_path=item["path"],
            prediction_sha256=item["prediction_sha256"],
            **score_prediction(row, prediction),
            **contrasts,
        )
        if item["teacher"] is not None:
            record["teacher"] = item["teacher"]["records"]
            record["teacher_preprocessing"] = item["teacher"].get(
                "target_preprocessing", "legacy_reference_cast_order"
            )
        groups.setdefault((item["split"], item["update"], item["variant"]), []).append(record)
    return {key: summarize(records) for key, records in groups.items()}


def analyze(study: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    plan = load_plan(study)
    p, root = plan["protocol"], Path(plan["data"])
    jobs = [j for j in plan["jobs"] if j["phase"] == "train"]
    all_rows = evaluation_rows(plan, jobs[0])
    references = {}
    for split in ["train", "dev"]:
        references.update(
            scene_references(root, split, {r["id"] for r in all_rows if r["split"] == split})
        )
    output.mkdir(parents=True)
    result = dict(
        status="running",
        plan_sha256=sha256(study / "plan.json"),
        source=source_snapshot(),
        human_status="pending",
        real_ratings=0,
        test_status="sealed",
        seeds={},
    )
    views = {
        name: dict(
            status="complete",
            data=str(root),
            split=split,
            manifest_path=str(root / split / "manifest.csv"),
            manifest_sha256=sha256(root / split / "manifest.csv"),
            calibration={
                k: v["locator"].calibration
                for k, v in references.items()
                if v["row"]["split"] == split
            },
            seeds={},
            human_status="pending",
            scope="non-blinded synthetic proxies; overfit is not generalization",
        )
        for name, split in [
            ("train", "train"),
            ("dev", "dev"),
            ("learning", "train"),
            ("train_interventions", "train"),
            ("dev_interventions", "dev"),
        ]
    }
    noise_identity, losses, training_sources, inference_sources = {}, [], [], []
    for seed in p["train_seeds"]:
        selected = [j for j in jobs if j["seed"] == seed]
        fairness = audit_seed(selected)
        methods, interventions, learning, probes = {s: {} for s in ["train", "dev"]}, {}, {}, {}
        observations = {}
        for job in selected:
            report, conditions = verify_inference(study, plan, job)
            inference_sources.append(report["source"])
            for key, fingerprint in conditions.items():
                if key in noise_identity and noise_identity[key] != fingerprint:
                    raise ValueError(
                        "Matched methods/seeds have different actual source/text/noise"
                    )
                noise_identity[key] = fingerprint
            scores = score_report(report, references)
            for split in ["train", "dev"]:
                methods[split][job["method"]] = scores[(split, p["steps"], "original")]
            if job["method"] == "full_role":
                for (split, step, variant), scored in scores.items():
                    if step == p["steps"]:
                        interventions.setdefault(split, {})[variant] = scored
                    if split == "train" and variant == "original":
                        learning[str(step)] = scored
                probes = dict(states=report["probes"], gradients=report["gradient_probes"])
            trained = json.loads((Path(job["directory"]) / "train_report.json").read_text())
            training_sources.append(trained["source"])
            observations[job["method"]] = training_diagnostics(
                trained, root, {r["id"] for r in all_rows if r["split"] == "train"}
            )
            losses.extend(
                dict(seed=seed, method=job["method"], **r)
                for r in trained["feature_statistics"]["observation"]["loss"]
            )
        paired = {
            split: {
                baseline: {
                    metric: paired_scene_ci(
                        [r for r in methods[split]["full_role"]["records"] if r["strength"] > 0],
                        [r for r in methods[split][baseline]["records"] if r["strength"] > 0],
                        metric,
                        seed=p["inference"]["seed"],
                        resamples=p["gate"]["bootstrap_resamples"],
                    )
                    for metric in ["t_target_mae", "d_target_mae", "p_source_drift", *CONTRASTS]
                }
                for baseline in ["constant", "binary", "old_new"]
            }
            for split in ["train", "dev"]
        }
        gates = learnability_gates(methods["train"], interventions["train"], p["gate"])
        result["seeds"][str(seed)] = dict(
            fairness=fairness,
            methods=methods,
            interventions=interventions,
            learning=learning,
            paired=paired,
            gates=gates,
            probes=probes,
            training_observations=observations,
            mechanisms=mechanism_summary(observations, probes),
        )
        for split in ["train", "dev"]:
            views[split]["seeds"][str(seed)] = methods[split]
        views["learning"]["seeds"][str(seed)] = learning
        views["train_interventions"]["seeds"][str(seed)] = interventions["train"]
        common_ids = set.intersection(
            *[{r["id"] for r in v["records"]} for v in interventions["dev"].values()]
        )
        views["dev_interventions"]["seeds"][str(seed)] = {
            k: summarize([r for r in v["records"] if r["id"] in common_ids])
            for k, v in interventions["dev"].items()
        }
    if any(s != training_sources[0] for s in training_sources) or any(
        s != inference_sources[0] for s in inference_sources
    ):
        raise ValueError("Methods were executed with different source snapshots")
    result["numeric_pass"] = all(s["gates"]["passed"] for s in result["seeds"].values())
    result["decision"] = (
        "go_controlled_mechanism_only"
        if result["numeric_pass"]
        else "no_go_current_injection_budget"
    )
    result["status"] = "complete"
    write_rows(output / "training_loss.csv", losses)
    for name, view in views.items():
        write_json(output / f"{name}_summary.json", view)
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.study, args.output)


if __name__ == "__main__":
    main()
