"""Bind frozen PR7 studies, then run zero-update direction smoke and paired diagnosis."""

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.inference.learnability import evaluation_rows
from src.infra.experiment import execute, source_snapshot
from src.training.learnability_study import completed, load_plan


def state_specs(protocol: dict) -> list[tuple[str, float]]:
    return [("source_bridge", t) for t in protocol["times"]] + [
        ("noise_only", t) for t in protocol["times"] if t != 1
    ]


def prepare(original: Path, normalized: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    protocol = load_yaml(CONFIG_ROOT / "experiments/role_direction.yaml")
    parents = {
        name: load_plan(path)
        for name, path in [("original", original), ("channel_rms", normalized)]
    }
    a, b = [copy.deepcopy(parents[n]["protocol"]) for n in parents]
    if b["encoder"].pop("normalization", None) != "channel_rms" or a != b:
        raise ValueError("Bind exactly the prior single-change, equal-budget studies")
    if parents["original"]["data"] != parents["channel_rms"]["data"]:
        raise ValueError("Use the identical canonical dataset")
    if protocol["training_updates"] != 0 or protocol["batch_size"] != 1:
        raise ValueError("Frozen direction studies cannot train or change batch arithmetic")
    jobs, cohort = [], None
    for family, parent in parents.items():
        for old in [j for j in parent["jobs"] if j["phase"] == "train"]:
            report = completed(old)
            rows = evaluation_rows(parent, old)
            if cohort is not None and cohort != rows:
                raise ValueError("Frozen comparison cohorts differ")
            cohort = rows
            pair = report["feature_statistics"]["observation"]["checkpoints"][-1]
            if pair["update"] != parent["protocol"]["steps"]:
                raise ValueError("Use only the preregistered final weights")
            jobs.append(
                dict(
                    name=f"{family}_{old['method']}_{old['seed']}",
                    family=family,
                    method=old["method"],
                    seed=old["seed"],
                    previous_job=old,
                    pair=pair,
                    train_report_sha256=sha256(Path(old["directory"]) / "train_report.json"),
                )
            )
    result = dict(
        status="ready",
        source=source_snapshot(),
        protocol=protocol,
        inherited=a,
        data=parents["original"]["data"],
        rows=rows,
        manifests={
            split: sha256(Path(parents["original"]["data"]) / split / "manifest.csv")
            for split in ["train", "dev", "test"]
        },
        parents={
            name: dict(path=str(path.resolve()), sha256=sha256(path / "plan.json"))
            for name, path in [("original", original), ("channel_rms", normalized)]
        },
        jobs=jobs,
        training_updates=0,
        test_status="sealed",
    )
    output.mkdir(parents=True)
    write_json(output / "plan.json", result)
    return result


def load_direction(study: Path) -> dict:
    plan = json.loads((study / "plan.json").read_text())
    if (
        plan["status"] != "ready"
        or plan["training_updates"] != 0
        or plan["test_status"] != "sealed"
    ):
        raise ValueError("Require the frozen zero-update plan")
    if plan["protocol"] != load_yaml(CONFIG_ROOT / "experiments/role_direction.yaml"):
        raise ValueError("Direction protocol changed since preparation")
    for parent in plan["parents"].values():
        if sha256(Path(parent["path"]) / "plan.json") != parent["sha256"]:
            raise ValueError("Parent study SHA256 changed")
    parents = {
        f: json.loads((Path(p["path"]) / "plan.json").read_text())
        for f, p in plan["parents"].items()
    }
    if (
        set(parents) != {"original", "channel_rms"}
        or plan["inherited"] != parents["original"]["protocol"]
    ):
        raise ValueError("Inherited prompt, seed or sampling budget changed")
    expected = {
        (f, j["method"], j["seed"]): j
        for f, p in parents.items()
        for j in p["jobs"]
        if j["phase"] == "train"
    }
    actual = [(j["family"], j["method"], j["seed"]) for j in plan["jobs"]]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Missing or duplicate frozen checkpoint matrix")
    for job in plan["jobs"]:
        if job["previous_job"] != expected[(job["family"], job["method"], job["seed"])]:
            raise ValueError("Frozen job differs from its parent study")
    canonical = next(j for j in parents["original"]["jobs"] if j["phase"] == "train")
    if plan["rows"] != evaluation_rows(parents["original"], canonical):
        raise ValueError("Direction rows differ from the fixed Train/Dev cohort")
    if any(
        sha256(Path(plan["data"]) / split / "manifest.csv") != digest
        for split, digest in plan["manifests"].items()
    ):
        raise ValueError("Canonical manifest bytes changed")
    if any(row["split"] not in {"train", "dev"} for row in plan["rows"]):
        raise ValueError("Sealed Test must not enter the direction matrix")
    return plan


def verify_job(job: dict) -> None:
    old = job["previous_job"]
    path = Path(old["directory"]) / "train_report.json"
    pair = job["pair"]
    report = json.loads(path.read_text())
    if (
        report["status"] != "complete"
        or sha256(path) != job["train_report_sha256"]
        or sha256(pair["lora"]) != pair["lora_sha256"]
        or sha256(pair["encoder"]["path"]) != pair["encoder"]["sha256"]
        or pair != report["feature_statistics"]["observation"]["checkpoints"][-1]
    ):
        raise ValueError("Frozen training report or paired checkpoint bytes changed")


def run_phase(study: Path, phase: str, attempt: int) -> None:
    plan = load_direction(study)
    if attempt < 1:
        raise ValueError("Use positive immutable attempt numbers")
    phase_path = study / f"{phase}_phase_{attempt:02d}.json"
    if phase_path.exists():
        raise FileExistsError(phase_path)
    if phase == "diagnose":
        smokes = sorted(study.glob("smoke_phase_*.json"))
        ready = False
        for path in smokes:
            record = json.loads(path.read_text())
            if record["status"] != "complete" or record["source"] != source_snapshot():
                continue
            for item in record["jobs"].values():
                report = json.loads(Path(item["path"]).read_text())
                ready |= (
                    sha256(item["path"]) == item["sha256"]
                    and report["status"] == "complete"
                    and report["smoke"]
                    and report["weights_unchanged"]
                    and report["plan_sha256"] == sha256(study / "plan.json")
                )
        if not ready:
            raise ValueError("Complete the real GPU smoke before the full matrix")
    jobs = (
        plan["jobs"]
        if phase == "diagnose"
        else [next(j for j in plan["jobs"] if j["name"] == "original_full_role_307")]
    )
    state = dict(
        status="running", phase=phase, source=source_snapshot(), jobs={}, training_updates=0
    )
    write_json(phase_path, state)

    def launch(job):
        directory = study / f"{phase}_{attempt:02d}" / job["name"]
        argv = [
            sys.executable,
            "-m",
            "src",
            "gpu",
            "--",
            sys.executable,
            "-m",
            "src.inference.direction_worker",
            "--study",
            str(study.resolve()),
            "--job",
            job["name"],
            "--output",
            str(directory.resolve()),
        ]
        if phase == "smoke":
            argv.append("--smoke")
        execute(argv, directory.with_name(directory.name + "_execution"))
        return dict(
            path=str((directory / "report.json").resolve()),
            sha256=sha256(directory / "report.json"),
        )

    try:
        with ThreadPoolExecutor(
            max_workers=1 if phase == "smoke" else plan["inherited"]["max_parallel"]
        ) as pool:
            pending = {pool.submit(launch, job): job for job in jobs}
            errors = []
            for future in as_completed(pending):
                job = pending[future]
                try:
                    state["jobs"][job["name"]] = launch_result = future.result()
                    launch_result["status"] = "complete"
                except BaseException as error:
                    state["jobs"][job["name"]] = dict(status="failed", error=repr(error))
                    errors.append(error)
                write_json(phase_path, state)
            if errors:
                raise RuntimeError(
                    "Direction jobs failed; preserve attempts and original logs"
                ) from errors[0]
        state["status"] = "complete"
    except BaseException:
        state["status"] = "failed"
        raise
    finally:
        write_json(phase_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    p = actions.add_parser("prepare")
    p.add_argument("--original-study", type=Path, required=True)
    p.add_argument("--normalized-study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = actions.add_parser("run")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--phase", choices=["smoke", "diagnose"], required=True)
    p.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.original_study, args.normalized_study, args.output)
    else:
        run_phase(args.study, args.phase, args.attempt)


if __name__ == "__main__":
    main()
