"""Bounded, reproducible upper-bound study using the existing leased trainer and CLI.

Prepare is CPU-only. Each run phase uses new outputs, preserves failures and
records executable source before/after every subprocess. No predictor is used.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.conditioning.controls import ControlKind
from src.conditioning.static import MaskSpec
from src.core.config import load_yaml
from src.core.io import read_rows, sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.datasets.control_export import export_controls
from src.datasets.role_controls import ORACLE_KINDS, export_role_controls
from src.datasets.scene_validation import validate_scene_dataset
from src.datasets.snapshot import snapshot_pairs
from src.evaluation.fairness import matched_training
from src.infra.experiment import execute
from src.training.config import build_config


def prepare(data: Path, output: Path, protocol_path: Path, cohort: Path | None = None) -> dict:
    validation = validate_scene_dataset(data)
    protocol = load_yaml(protocol_path)
    geometry = protocol.get("study") == "role_geometry_pilot_v1"
    expected = (
        ["dynamic_binary", "oracle_geometry"]
        if geometry
        else [
            "static_mask",
            "dilated_mask",
            "dynamic_binary",
            "oracle_role",
        ]
    )
    if protocol["schema_version"] != 1 or protocol["methods"] != expected:
        raise ValueError("Use the declared matched upper-bound protocol")
    if geometry:
        from src.datasets.holdout import require_dev

        if cohort is None or protocol["inference"]["splits"] != ["dev"]:
            raise ValueError("Geometry pilot requires a sealed cohort and Dev-only inference")
        require_dev(data, cohort)
    pilot = "oracle_geometry" if geometry else "oracle_role"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    template = load_yaml(CONFIG_ROOT / "flux2_edit_lora.yaml")
    resolution = int(read_rows(data / "train/manifest.csv")[0]["width"])
    template["config"]["process"][0]["datasets"][0]["resolution"] = [resolution]
    template_path = output / "template.yaml"
    template_path.write_text(yaml.safe_dump(template, sort_keys=False))
    plan = dict(
        status="preparing",
        data=str(data.resolve()),
        protocol=protocol,
        protocol_sha256=sha256(protocol_path),
        validation=validation,
        jobs=[],
    )
    if geometry:
        plan["cohort"] = dict(path=str(cohort.resolve()), sha256=sha256(cohort))
    jobs = [
        (
            "smoke",
            pilot,
            protocol["train_seeds"][0],
            protocol["smoke"]["steps"],
            protocol["smoke"]["batch_size"],
        )
    ]
    jobs.append(
        (
            "capacity",
            pilot,
            protocol["train_seeds"][0],
            protocol["smoke"]["steps"],
            protocol["batch_size"],
        )
    )
    jobs += [
        ("train", method, seed, protocol["steps"], protocol["batch_size"])
        for seed in protocol["train_seeds"]
        for method in protocol["methods"]
    ]
    for phase, method, seed, steps, batch in jobs:
        name = f"{phase}_{method}_{seed}"
        directory = output / name
        directory.mkdir()
        copied, controls = directory / "data", directory / "controls"
        snapshot_pairs(data, copied)
        if method in ORACLE_KINDS:
            export_role_controls(copied, controls, method)
        else:
            export_controls(
                copied,
                controls,
                MaskSpec(
                    ControlKind(method),
                    protocol["dilation_radius"] if method == "dilated_mask" else 0,
                ),
            )
        args = SimpleNamespace(
            steps=steps,
            batch_size=batch,
            rank=protocol["rank"],
            template=template_path,
            name=name,
            output_root=directory / "lora",
            data_root=copied,
            control=method,
            control_root=controls,
        )
        configuration = build_config(args)
        configuration["config"]["process"][0]["train"]["seed"] = seed
        path = directory / "train.yaml"
        path.write_text(yaml.safe_dump(configuration, sort_keys=False))
        plan["jobs"].append(
            dict(
                name=name,
                phase=phase,
                method=method,
                seed=seed,
                directory=str(directory.resolve()),
                config_sha256=sha256(path),
            )
        )
        write_json(output / "plan.json", plan)
    plan["status"] = "prepared"
    write_json(output / "plan.json", plan)
    return plan


def train_job(job: dict) -> None:
    directory = Path(job["directory"])
    config = directory / "train.yaml"
    if sha256(config) != job["config_sha256"]:
        raise ValueError("Prepared training configuration SHA256 changed")
    execute(
        [
            sys.executable,
            "-m",
            "src",
            "gpu",
            "--",
            sys.executable,
            "-m",
            "src.training",
            str(config),
            "--report",
            str(directory / "train_report.json"),
        ],
        directory / "training_execution",
    )


def infer_job(job: dict, protocol: dict) -> None:
    directory = Path(job["directory"])
    report = json.loads((directory / "train_report.json").read_text())
    if (
        report["config_sha256"] != job["config_sha256"]
        or sha256(directory / "train.yaml") != job["config_sha256"]
    ):
        raise ValueError("Training configuration drifted before matched inference")
    if (
        report["status"] != "complete"
        or sha256(report["checkpoint"]) != report["checkpoint_sha256"]
    ):
        raise ValueError("Inference requires the completed verified checkpoint")
    sampling = protocol["inference"]
    for split in sampling["splits"]:
        manifest = directory / "data" / split / "manifest.csv"
        argv = [
            sys.executable,
            "-m",
            "src",
            "gpu",
            "--",
            sys.executable,
            "-m",
            "src",
            "infer",
            "--data-root",
            str(directory / "data"),
            "--manifest",
            str(manifest),
            "--control",
            job["method"],
            "--control-root",
            str(directory / "controls"),
            "--lora",
            report["checkpoint"],
            "--limit",
            str(len(read_rows(manifest))),
            "--side",
            str(int(read_rows(manifest)[0]["width"])),
            "--steps",
            str(sampling["steps"]),
            "--guidance",
            str(sampling["guidance"]),
            "--seed",
            str(sampling["seed"]),
            "--output",
            str(directory / f"{split}_inference"),
        ]
        if job["method"] in ORACLE_KINDS:
            argv.append("--allow-oracle")
        execute(argv, directory / f"{split}_execution")


def verify_plan(plan: dict, phase: str) -> None:
    if phase not in {"smoke", "capacity", "train", "infer"}:
        raise ValueError("Unknown study phase")
    seeds, methods = plan["protocol"]["train_seeds"], plan["protocol"]["methods"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and distinct")
    pilot = "oracle_role"
    if plan["protocol"].get("study") == "role_geometry_pilot_v1":
        from src.datasets.holdout import require_dev

        if methods != ["dynamic_binary", "oracle_geometry"] or plan["protocol"]["inference"][
            "splits"
        ] != ["dev"]:
            raise ValueError("Geometry study permits the declared methods on Dev only")
        seal = Path(plan["cohort"]["path"])
        if sha256(seal) != plan["cohort"]["sha256"]:
            raise ValueError("Cohort seal SHA256 changed")
        require_dev(Path(plan["data"]), seal)
        pilot = "oracle_geometry"
    expected = {("smoke", pilot, seeds[0]), ("capacity", pilot, seeds[0])}
    expected |= {("train", method, seed) for method in methods for seed in seeds}
    actual = {(j["phase"], j["method"], j["seed"]) for j in plan["jobs"]}
    if len(actual) != len(plan["jobs"]) or not actual <= expected:
        raise ValueError("Study job identities differ from declared protocol")
    if phase in {"train", "infer"} and (actual != expected or plan["status"] != "prepared"):
        raise ValueError("Complete every planned configuration before matched study execution")
    if phase in {"smoke", "capacity"} and (phase, pilot, seeds[0]) not in actual:
        raise ValueError("Requested pilot configuration is not prepared")


def run_phase(output: Path, phase: str, parallel: int) -> dict:
    plan = json.loads((output / "plan.json").read_text())
    verify_plan(plan, phase)
    if not 1 <= parallel <= plan["protocol"]["max_parallel"]:
        raise ValueError("Parallelism exceeds the predeclared RAM/GPU limit")
    jobs = [
        job
        for job in plan["jobs"]
        if job["phase"] == (phase if phase in {"smoke", "capacity"} else "train")
    ]
    if phase in {"smoke", "capacity"} and parallel != 1:
        raise ValueError("Measure smoke on one GPU before parallel training")
    if phase != "smoke":
        smoke = next(job for job in plan["jobs"] if job["phase"] == "smoke")
        report = json.loads((Path(smoke["directory"]) / "train_report.json").read_text())
        if report["status"] != "complete":
            raise ValueError("Complete the minimum training smoke first")
    if phase in {"train", "infer"}:
        capacity = next(job for job in plan["jobs"] if job["phase"] == "capacity")
        report = json.loads((Path(capacity["directory"]) / "train_report.json").read_text())
        reload_report = json.loads((output / "oracle_reload_smoke/report.json").read_text())
        if report["status"] != "complete" or reload_report["status"] != "complete":
            raise ValueError(
                "Complete measured batch capacity and reloaded condition sensitivity first"
            )
    if (
        phase == "infer"
        and json.loads((output / "train_phase.json").read_text())["status"] != "complete"
    ):
        raise ValueError("Complete every matched training run before fixed evaluation")
    result_path = output / f"{phase}_phase.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result = dict(status="running", phase=phase, parallel=parallel, jobs={})
    write_json(result_path, result)
    action = (lambda j: infer_job(j, plan["protocol"])) if phase == "infer" else train_job
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(action, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                result["jobs"][job["name"]] = dict(status="complete")
            except BaseException as error:
                result["jobs"][job["name"]] = dict(status="failed", error=repr(error))
            write_json(result_path, result)
    result["status"] = (
        "complete" if all(j["status"] == "complete" for j in result["jobs"].values()) else "failed"
    )
    if phase == "train" and result["status"] == "complete":
        for seed in plan["protocol"]["train_seeds"]:
            reports = {
                job["method"]: Path(job["directory"]) / "train_report.json"
                for job in jobs
                if job["seed"] == seed
            }
            write_json(output / f"matched_budget_{seed}.json", matched_training(reports))
    write_json(result_path, result)
    if result["status"] != "complete":
        raise RuntimeError(f"Study phase failed; see {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    prepare_parser.add_argument("--data", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--cohort", type=Path, help="Required sealed Dev cohort for Geometry"
    )
    prepare_parser.add_argument(
        "--protocol", type=Path, default=CONFIG_ROOT / "experiments/role_validation.yaml"
    )
    run_parser = actions.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument(
        "--phase", choices=["smoke", "capacity", "train", "infer"], required=True
    )
    run_parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.data.resolve(), args.output.resolve(), args.protocol, args.cohort))
    else:
        print(run_phase(args.output.resolve(), args.phase, args.parallel))


if __name__ == "__main__":
    main()
