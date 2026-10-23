"""Prepare immutable references to existing assets; launch only bounded leased inference."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.conditioning.feature_interventions import VARIANTS
from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.datasets.holdout import require_dev
from src.evaluation.feature_fairness import audit_seed
from src.infra.experiment import execute


def verify_protocol(p: dict):
    if (
        p["schema_version"] != 1
        or p["study"] != "role_feature_failure_analysis_v1"
        or p["training_updates"] != 0
        or p["train_seeds"] != [211, 223]
        or p["methods"] != ["constant", "binary", "old_new", "full_role"]
        or p["variants"] != list(VARIANTS)
        or not 1 <= p["max_parallel"] <= 2
        or p["inference"] != dict(side=512, steps=20, guidance=4.0, seed=42)
        or p["probe_steps"] != [0, 4, 9, 14, 19]
        or p["diagnostic_method"] != "full_role"
    ):
        raise ValueError("Use the preregistered frozen-checkpoint Dev protocol")


def prepare(existing: Path, output: Path, protocol: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads((existing / "plan.json").read_text())
    p = load_yaml(protocol)
    verify_protocol(p)
    data, cohort = Path(previous["data"]), Path(previous["cohort"]["path"])
    seal = require_dev(data, cohort, data / "dev/manifest.csv")
    if not set(p["diagnostic_samples"]) <= set(seal["dev_ids"]):
        raise ValueError("Diagnostic samples must belong to sealed Dev")
    checkpoints, fairness = [], {}
    for seed in p["train_seeds"]:
        jobs = [j for j in previous["jobs"] if j["phase"] == "train" and j["seed"] == seed]
        fairness[str(seed)] = audit_seed(jobs)
        for job in jobs:
            report_path = Path(job["directory"]) / "train_report.json"
            report = json.loads(report_path.read_text())
            checkpoints.append(
                dict(
                    seed=seed,
                    method=job["method"],
                    lora=report["checkpoint"],
                    lora_sha256=report["checkpoint_sha256"],
                    encoder=report["feature_checkpoint"]["path"],
                    encoder_sha256=report["feature_checkpoint"]["sha256"],
                    training_report=str(report_path.resolve()),
                    training_report_sha256=sha256(report_path),
                    previous_predictions=str((Path(job["directory"]) / "dev_inference").resolve()),
                )
            )
    plan = dict(
        schema_version=1,
        status="prepared",
        protocol=p,
        protocol_path=str(protocol.resolve()),
        protocol_sha256=sha256(protocol),
        data=str(data.resolve()),
        cohort=str(cohort.resolve()),
        cohort_sha256=sha256(cohort),
        existing_study=str(existing.resolve()),
        existing_plan_sha256=sha256(existing / "plan.json"),
        checkpoints=checkpoints,
        fairness=fairness,
    )
    output.mkdir(parents=True)
    write_json(output / "protocol.json", p)
    write_json(output / "plan.json", plan)
    return plan


def load_plan(study: Path) -> dict:
    plan = json.loads((study / "plan.json").read_text())
    verify_protocol(plan["protocol"])
    if (
        plan["status"] != "prepared"
        or sha256(plan["protocol_path"]) != plan["protocol_sha256"]
        or plan["protocol"] != json.loads((study / "protocol.json").read_text())
        or sha256(plan["cohort"]) != plan["cohort_sha256"]
        or sha256(Path(plan["existing_study"]) / "plan.json") != plan["existing_plan_sha256"]
    ):
        raise ValueError("Prepared protocol, cohort or input study changed")
    require_dev(Path(plan["data"]), Path(plan["cohort"]))
    expected = {
        (s, m) for s in plan["protocol"]["train_seeds"] for m in plan["protocol"]["methods"]
    }
    declared = [(c["seed"], c["method"]) for c in plan["checkpoints"]]
    if set(declared) != expected or len(declared) != len(expected):
        raise ValueError("Incomplete or duplicate frozen checkpoint matrix")
    for ckpt in plan["checkpoints"]:
        if (
            sha256(ckpt["lora"]) != ckpt["lora_sha256"]
            or sha256(ckpt["encoder"]) != ckpt["encoder_sha256"]
            or sha256(ckpt["training_report"]) != ckpt["training_report_sha256"]
        ):
            raise ValueError("Frozen checkpoint or original training report changed")
        training = json.loads(Path(ckpt["training_report"]).read_text())
        if (
            training["status"] != "complete"
            or training["checkpoint_sha256"] != ckpt["lora_sha256"]
            or training["feature_checkpoint"]["sha256"] != ckpt["encoder_sha256"]
            or training["role_features"]["mode"] != ckpt["method"]
            or training["role_features"]["cohort_sha256"] != plan["cohort_sha256"]
            or training["feature_statistics"]["initial_seed"] != ckpt["seed"]
        ):
            raise ValueError("Declared diagnostic checkpoint differs from its actual training")
    return plan


def verified_receipt(study: Path, receipt: dict) -> Path:
    output = Path(receipt["output"])
    if sha256(output / "report.json") != receipt["report_sha256"]:
        raise ValueError("Completed attempt provenance changed")
    report = json.loads((output / "report.json").read_text())
    execution = receipt["execution"]
    if (
        report["status"] != "complete"
        or report["plan_sha256"] != sha256(study / "plan.json")
        or execution["status"] != "complete"
        or execution["returncode"] != 0
        or report["source"] != execution["source"]
    ):
        raise ValueError("Completed attempt provenance changed")
    return output


def phase_outputs(study: Path, phase: str) -> dict[str, Path]:
    records = json.loads((study / f"{phase}_phase.json").read_text())
    return {seed: verified_receipt(study, receipt) for seed, receipt in records.items()}


def run_phase(study: Path, phase: str, attempt: int = 1):
    if phase not in {"smoke", "diagnose"}:
        raise ValueError("Unsupported diagnostic phase")
    p = load_plan(study)["protocol"]
    if attempt < 1 or (study / f"{phase}_phase.json").exists():
        raise FileExistsError("Use a fresh attempt only for unfinished phases")
    if phase != "smoke":
        output = phase_outputs(study, "smoke")[str(p["train_seeds"][0])]
        smoke = json.loads((output / "report.json").read_text())
        if smoke["status"] != "complete" or not smoke["repeat_exact"]:
            raise ValueError("Real diagnostic GPU smoke has not passed")
    seeds = p["train_seeds"][:1] if phase == "smoke" else p["train_seeds"]
    run_jobs(study, phase, seeds, p["max_parallel"], attempt)


def run_jobs(
    study: Path,
    phase: str,
    seeds: list[int],
    parallel: int,
    attempt: int,
    worker_module="src.inference.feature_diagnostics",
):
    """Shared bounded UUID-lease launch; completed siblings are immutable and reusable."""
    if attempt < 1 or (study / f"{phase}_phase.json").exists():
        raise FileExistsError("Use a fresh attempt only for unfinished phases")

    def launch(seed):
        selected = study / f"{phase}_{seed}_receipt.json"
        if selected.exists():
            receipt = json.loads(selected.read_text())
            verified_receipt(study, receipt)
            return receipt
        output = study / f"{phase}_{seed}_attempt{attempt:02d}"
        execution = execute(
            [
                sys.executable,
                "-m",
                "src",
                "gpu",
                "--",
                sys.executable,
                "-m",
                worker_module,
                "--study",
                str(study.resolve()),
                "--phase",
                phase,
                "--train-seed",
                str(seed),
                "--output",
                str(output.resolve()),
            ],
            study / f"{phase}_{seed}_attempt{attempt:02d}_execution",
        )
        receipt = dict(
            output=str(output.resolve()),
            report_sha256=sha256(output / "report.json"),
            execution=execution,
        )
        verified_receipt(study, receipt)
        write_json(selected, receipt)
        return receipt

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(launch, seed): seed for seed in seeds}
        result = {str(futures[f]): f.result() for f in as_completed(futures)}
    write_json(study / f"{phase}_phase.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    prepare_parser.add_argument("--existing-study", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--protocol", type=Path, default=CONFIG_ROOT / "experiments/role_feature_diagnostics.yaml"
    )
    run = actions.add_parser("run")
    run.add_argument("--study", type=Path, required=True)
    run.add_argument("--phase", choices=["smoke", "diagnose"], required=True)
    run.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.existing_study, args.output, args.protocol)
    else:
        run_phase(args.study, args.phase, args.attempt)


if __name__ == "__main__":
    main()
