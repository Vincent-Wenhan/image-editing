"""Bounded matched feature experiments with sealed Test and fresh output directories."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.conditioning.features import FEATURE_MODES, EncoderConfig
from src.core.config import load_yaml
from src.core.io import read_rows, sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.datasets.holdout import require_dev
from src.datasets.scene_validation import validate_scene_dataset
from src.datasets.snapshot import snapshot_pairs
from src.infra.experiment import execute
from src.training.config import build_config
from src.training.feature_config import attach_features, verify_features
from src.training.role_study import train_job


def verify_protocol(protocol: dict):
    if (
        protocol["schema_version"] != 1
        or protocol["study"] != "independent_role_features_v1"
        or protocol["methods"] != list(FEATURE_MODES)
        or protocol["inference"]["splits"] != ["dev"]
        or len(protocol["train_seeds"]) != 2
        or len(set(protocol["train_seeds"])) != len(protocol["train_seeds"])
        or not 1 <= protocol["max_parallel"] <= 4
        or not 2 <= protocol["steps"] <= 20
        or protocol["smoke"]["steps"] != 2
        or protocol["smoke"]["batch_size"] != 1
        or protocol["batch_size"] != 4
    ):
        raise ValueError("Use the bounded preregistered four-method Dev protocol")
    EncoderConfig(**protocol["encoder"])


def prepare(data: Path, output: Path, protocol_path: Path, cohort: Path):
    require_dev(data, cohort)
    validation = validate_scene_dataset(data)
    protocol = load_yaml(protocol_path)
    verify_protocol(protocol)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    template = load_yaml(CONFIG_ROOT / "flux2_edit_lora.yaml")
    side = int(read_rows(data / "train/manifest.csv")[0]["width"])
    process = template["config"]["process"][0]
    process["datasets"][0]["resolution"] = [side]
    process["train"]["lr"] = protocol["lora_lr"]
    template_path = output / "template.yaml"
    template_path.write_text(yaml.safe_dump(template, sort_keys=False))
    plan = dict(
        status="preparing",
        data=str(data.resolve()),
        protocol=protocol,
        protocol_sha256=sha256(protocol_path),
        validation=validation,
        cohort=dict(path=str(cohort.resolve()), sha256=sha256(cohort)),
        jobs=[],
    )
    seed = protocol["train_seeds"][0]
    jobs = [
        ("smoke", "full_role", seed, protocol["smoke"]["steps"], 1),
        ("capacity", "full_role", seed, protocol["smoke"]["steps"], protocol["batch_size"]),
    ] + [
        ("train", mode, seed, protocol["steps"], protocol["batch_size"])
        for seed in protocol["train_seeds"]
        for mode in protocol["methods"]
    ]
    for phase, mode, seed, steps, batch in jobs:
        name = f"{phase}_{mode}_{seed}"
        directory = output / name
        directory.mkdir()
        copied = directory / "data"
        snapshot_pairs(data, copied)
        config = build_config(
            SimpleNamespace(
                steps=steps,
                batch_size=batch,
                rank=protocol["rank"],
                template=template_path,
                name=name,
                output_root=directory / "lora",
                data_root=copied,
                control="none",
                control_root=None,
            )
        )
        config["config"]["process"][0]["train"]["seed"] = seed
        attach_features(config, copied, cohort, mode, protocol["encoder"], protocol["feature_lr"])
        path = directory / "train.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        plan["jobs"].append(
            dict(
                name=name,
                phase=phase,
                method=mode,
                seed=seed,
                directory=str(directory.resolve()),
                config_sha256=sha256(path),
            )
        )
        write_json(output / "plan.json", plan)
    plan["status"] = "prepared"
    write_json(output / "plan.json", plan)
    return plan


def infer_job(job, protocol, cohort):
    directory = Path(job["directory"])
    report = json.loads((directory / "train_report.json").read_text())
    feature = report["feature_checkpoint"]
    if (
        report["status"] != "complete"
        or sha256(report["checkpoint"]) != report["checkpoint_sha256"]
        or report["config_sha256"] != job["config_sha256"]
        or sha256(directory / "train.yaml") != job["config_sha256"]
        or sha256(feature["path"]) != feature["sha256"]
    ):
        raise ValueError("Feature inference requires complete paired hashed training artifacts")
    manifest = directory / "data/dev/manifest.csv"
    rows, sampling = read_rows(manifest), protocol["inference"]
    execute(
        [
            sys.executable,
            "-m",
            "src",
            "gpu",
            "--",
            sys.executable,
            "-m",
            "src.inference.feature_runner",
            "--data-root",
            str(directory / "data"),
            "--manifest",
            str(manifest),
            "--cohort",
            cohort,
            "--lora",
            report["checkpoint"],
            "--encoder",
            feature["path"],
            "--feature-mode",
            job["method"],
            "--allow-oracle",
            "--limit",
            str(len(rows)),
            "--side",
            rows[0]["width"],
            "--steps",
            str(sampling["steps"]),
            "--guidance",
            str(sampling["guidance"]),
            "--seed",
            str(sampling["seed"]),
            "--output",
            str(directory / "dev_inference"),
        ],
        directory / "dev_execution",
    )


def completed(job):
    report = json.loads((Path(job["directory"]) / "train_report.json").read_text())
    if (
        report["status"] != "complete"
        or sha256(report["feature_checkpoint"]["path"]) != report["feature_checkpoint"]["sha256"]
    ):
        raise ValueError("Feature prerequisite training did not complete")
    return report


def run_phase(output: Path, phase: str):
    if phase not in {"smoke", "capacity", "train", "infer"}:
        raise ValueError("Unknown feature study phase")
    plan = json.loads((output / "plan.json").read_text())
    verify_protocol(plan["protocol"])
    if plan["protocol"] != json.loads((output / "protocol.json").read_text()):
        raise ValueError("Prepared feature protocol changed")
    if sha256(plan["cohort"]["path"]) != plan["cohort"]["sha256"]:
        raise ValueError("Feature cohort seal changed")
    require_dev(Path(plan["data"]), Path(plan["cohort"]["path"]))
    if plan["status"] != "prepared":
        raise ValueError("Feature study not fully prepared")
    identities = {(j["phase"], j["method"], j["seed"]) for j in plan["jobs"]}
    expected = {("train", m, s) for m in FEATURE_MODES for s in plan["protocol"]["train_seeds"]}
    first = plan["protocol"]["train_seeds"][0]
    expected |= {("smoke", "full_role", first), ("capacity", "full_role", first)}
    if identities != expected or len(identities) != len(plan["jobs"]):
        raise ValueError("Feature study matrix is incomplete or duplicated")
    for job in plan["jobs"]:
        path = Path(job["directory"]) / "train.yaml"
        config, protocol = load_yaml(path), plan["protocol"]
        meta = verify_features(config)
        process = config["config"]["process"][0]
        steps = protocol["steps"] if job["phase"] == "train" else protocol["smoke"]["steps"]
        batch = (
            protocol["smoke"]["batch_size"] if job["phase"] == "smoke" else protocol["batch_size"]
        )
        if (
            sha256(path) != job["config_sha256"]
            or meta["mode"] != job["method"]
            or meta["encoder"] != protocol["encoder"]
            or meta["lr"] != protocol["feature_lr"]
            or meta["cohort_sha256"] != plan["cohort"]["sha256"]
            or meta["cohort"] != plan["cohort"]["path"]
            or process["train"]["seed"] != job["seed"]
            or process["train"]["steps"] != steps
            or process["train"]["batch_size"] != batch
            or process["train"]["lr"] != protocol["lora_lr"]
            or process["network"]["linear"] != protocol["rank"]
            or process["network"]["linear_alpha"] != protocol["rank"]
        ):
            raise ValueError("Prepared feature job differs from the matched protocol")
    smoke = next(j for j in plan["jobs"] if j["phase"] == "smoke")
    if phase != "smoke":
        smoke_report = completed(smoke)
        probe = json.loads((output / "gpu_smoke/report.json").read_text())
        if (
            probe["status"] != "complete"
            or probe["lora_sha256"] != smoke_report["checkpoint_sha256"]
            or probe["encoder_sha256"] != smoke_report["feature_checkpoint"]["sha256"]
            or probe["cohort_sha256"] != plan["cohort"]["sha256"]
            or not probe["pretrained_zero_init_exact"]
            or not probe["checkpoint_reload_png_exact"]
            or not probe["probes"]["zero_condition_exact"]
        ):
            raise ValueError("Real GPU feature smoke gate has not passed")
    if phase in {"train", "infer"}:
        capacity = completed(next(j for j in plan["jobs"] if j["phase"] == "capacity"))
        if capacity["peak_reserved_gib"] > capacity["gpu_total_gib"] * 0.8:
            raise ValueError("Measured batch leaves insufficient OOM headroom")
    jobs = [j for j in plan["jobs"] if j["phase"] == ("train" if phase == "infer" else phase)]
    workers = 1 if phase in {"smoke", "capacity"} else plan["protocol"]["max_parallel"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(infer_job, j, plan["protocol"], plan["cohort"]["path"])
            if phase == "infer"
            else pool.submit(train_job, j): j["name"]
            for j in jobs
        }
        outcomes = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                outcomes[name] = dict(status="complete")
            except BaseException as error:
                outcomes[name] = dict(status="failed", error=repr(error))
            write_json(output / f"{phase}_phase.json", outcomes)
    if any(o["status"] != "complete" for o in outcomes.values()):
        raise RuntimeError(f"Feature {phase} phase failed; original logs retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    prepare_parser.add_argument("--cohort", type=Path, required=True)
    prepare_parser.add_argument(
        "--protocol", type=Path, default=CONFIG_ROOT / "experiments/role_features.yaml"
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("phase", choices=["smoke", "capacity", "train", "infer"])
    run_parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_root, args.output, args.protocol, args.cohort)
    else:
        run_phase(args.study, args.phase)


if __name__ == "__main__":
    main()
