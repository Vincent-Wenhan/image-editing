"""Overlay corrected supervision without changing primary pixel metrics or decisions."""

import argparse
import copy
import json
from pathlib import Path

from src.core.io import sha256, write_json
from src.evaluation.feature_teacher import TARGET_PREPROCESSING
from src.evaluation.learnability_analysis import summarize_teacher
from src.evaluation.learnability_provenance import verify_inference, verify_teacher
from src.inference.learnability import evaluation_rows
from src.infra.experiment import source_snapshot
from src.training.learnability_study import load_plan


def verified_correction(study, plan, job, corrections):
    primary, _ = verify_inference(study, plan, job)
    path = corrections / job["name"] / "report.json"
    corrected = json.loads(path.read_text())
    execution = json.loads((corrections / f"{job['name']}_execution/execution.json").read_text())
    if (
        corrected["status"] != "complete"
        or corrected["job"] != job
        or corrected["plan_sha256"] != sha256(study / "plan.json")
        or corrected["primary_inference_sha256"]
        != sha256(Path(job["directory"]) / "inference/report.json")
        or corrected["training_updates"] != 0
        or corrected["target_is_reference"]
        or corrected["target_preprocessing"] != TARGET_PREPROCESSING
        or corrected["test_status"] != "sealed"
        or execution["status"] != "complete"
        or execution["returncode"] != 0
        or execution["source"] != corrected["source"]
    ):
        raise ValueError("Corrected supervision provenance or target transform differs")
    originals = {
        (r["update"], r["id"]): r for r in primary["results"] if r["variant"] == "original"
    }
    lookup = {(r["update"], r["id"]): r for r in corrected["results"]}
    rows = {r["id"]: r for r in evaluation_rows(plan, job)}
    if lookup.keys() != originals.keys() or len(lookup) != len(corrected["results"]):
        raise ValueError("Missing or duplicate corrected teacher case")
    for key, item in lookup.items():
        before = originals[key]
        if (
            not item["source_only_capture"]
            or any(
                item[k] != before[k]
                for k in [
                    "split",
                    "group_id",
                    "strength",
                    "feature_sha256",
                    "checkpoint_sha256",
                    "encoder_sha256",
                    "source_tokens_sha256",
                    "initial_noise_sha256",
                    "text_sha256",
                ]
            )
            or item["teacher"]["target_preprocessing"] != TARGET_PREPROCESSING
        ):
            raise ValueError("Corrected teacher changed the actual original condition or weights")
        verify_teacher(dict(item, variant="original"), rows[item["id"]], plan["protocol"])
    expected = {r["id"] for r in primary["gradient_probes"]}
    if (
        len(corrected["gradients"]) != len(expected)
        or {r["id"] for r in corrected["gradients"]} != expected
    ):
        raise ValueError("Corrected fixed-gradient cohort changed")
    for row in corrected["gradients"]:
        gradient = row["gradient"]
        if (
            row["update"] != plan["protocol"]["steps"]
            or gradient["target_preprocessing"] != TARGET_PREPROCESSING
            or gradient["target_is_reference"]
            or gradient["target_sha256"] != rows[row["id"]]["target_sha256"]
            or [r["time"] for r in gradient["records"]] != plan["protocol"]["teacher_times"]
        ):
            raise ValueError("Corrected gradient supervision differs")
    return (
        lookup,
        corrected,
        dict(path=str(path.resolve()), sha256=sha256(path), source=corrected["source"]),
    )


def patch_scores(scored, corrections):
    """Keep verified pixel records and aggregates exactly, including legacy rounding."""
    result = copy.deepcopy(scored)
    rows = result["records"]
    for row in rows:
        if "teacher" in row:
            row["teacher"] = corrections[(row["update"], row["id"])]["teacher"]["records"]
            row["teacher_preprocessing"] = TARGET_PREPROCESSING
    result["teacher_by_time"] = summarize_teacher(rows)
    result["teacher_preprocessing"] = sorted(
        {r["teacher_preprocessing"] for r in rows if "teacher" in r}
    )
    return result


def correct(study, primary_analysis, corrections, output):
    if output.exists():
        raise FileExistsError(output)
    plan = load_plan(study)
    original = json.loads((primary_analysis / "summary.json").read_text())
    if original["status"] != "complete" or original["plan_sha256"] != sha256(study / "plan.json"):
        raise ValueError("Bind a completed original analysis")
    result = copy.deepcopy(original)
    result.update(
        source=source_snapshot(),
        primary_analysis_sha256=sha256(primary_analysis / "summary.json"),
        target_preprocessing=TARGET_PREPROCESSING,
        legacy_teacher_status="superseded_source_cast_order",
        corrections={},
    )
    mappings, sources, fingerprints = {}, [], {}
    for job in [j for j in plan["jobs"] if j["phase"] == "train"]:
        lookup, report, receipt = verified_correction(study, plan, job, corrections)
        mappings[(str(job["seed"]), job["method"])] = lookup
        result["corrections"][job["name"]] = receipt
        sources.append(report["source"])
        for item in report["results"]:
            for t in item["teacher"]["records"]:
                key = item["id"], t["time"]
                signature = tuple(
                    t[k]
                    for k in [
                        "target_latent_sha256",
                        "noise_sha256",
                        "source_tokens_sha256",
                        "text_sha256",
                    ]
                )
                if key in fingerprints and fingerprints[key] != signature:
                    raise ValueError("Corrected teachers differ across checkpoints/methods/seeds")
                fingerprints[key] = signature
        if job["method"] == "full_role":
            seed = result["seeds"][str(job["seed"])]
            seed["probes"]["gradients"] = report["gradients"]
            seed["mechanisms"]["gradients"] = report["gradients"]
    if any(source != sources[0] for source in sources):
        raise ValueError("Corrected probes used different source snapshots")
    for seed, record in result["seeds"].items():
        for split, methods in record["methods"].items():
            record["methods"][split] = {
                m: patch_scores(v, mappings[(seed, m)]) for m, v in methods.items()
            }
        for split, variants in record["interventions"].items():
            record["interventions"][split] = {
                n: patch_scores(v, mappings[(seed, "full_role")]) for n, v in variants.items()
            }
        record["learning"] = {
            u: patch_scores(v, mappings[(seed, "full_role")]) for u, v in record["learning"].items()
        }
        if record["gates"] != original["seeds"][seed]["gates"]:
            raise RuntimeError("Correction changed a preregistered gate")
    output.mkdir(parents=True)
    for name in ["train", "dev", "learning", "train_interventions", "dev_interventions"]:
        path = primary_analysis / f"{name}_summary.json"
        view = json.loads(path.read_text())
        for seed, methods in view["seeds"].items():
            view["seeds"][seed] = {
                m: patch_scores(v, mappings[(seed, m if name in {"train", "dev"} else "full_role")])
                for m, v in methods.items()
            }
        view.update(primary_summary_sha256=sha256(path), target_preprocessing=TARGET_PREPROCESSING)
        write_json(output / path.name, view)
    result["primary_metrics_and_decisions_unchanged"] = True
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--primary-analysis", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correct(args.study, args.primary_analysis, args.corrections, args.output)


if __name__ == "__main__":
    main()
