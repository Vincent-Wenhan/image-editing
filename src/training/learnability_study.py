"""Bounded few-example overfit study, using the original trainer and sealed parent data."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.conditioning.features import FEATURE_MODES
from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.datasets.holdout import require_dev
from src.datasets.research_view import create_view, verify_view
from src.datasets.scene_validation import validate_scene_dataset
from src.infra.experiment import execute
from src.training.config import build_config
from src.training.feature_config import attach_features, verify_features
from src.training.role_study import train_job


def verify_protocol(p):
    if (
        p["schema_version"] != 1
        or p["study"] != "role_learnability_v1"
        or p["methods"] != list(FEATURE_MODES)
        or p["train_seeds"] != [307, 311]
        or p["train_groups"] != 2
        or p["strengths"] != [0, 0.5, 1]
        or p["steps"] != 120
        or p["batch_size"] != 6
        or p["observe_updates"] != [1, 20, 60, 120]
        or p["smoke"] != dict(steps=2, batch_size=1)
        or p["teacher_times"] != [0.1, 0.5, 0.9]
        or not 1 <= p["max_parallel"] <= 4
    ):
        raise ValueError("Use the preregistered bounded Train-overfit protocol")


def prepare(data: Path, cohort: Path, output: Path, protocol: Path):
    if output.exists():
        raise FileExistsError(output)
    require_dev(data, cohort)
    validation, p = validate_scene_dataset(data), load_yaml(protocol)
    verify_protocol(p)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", p)
    template = load_yaml(CONFIG_ROOT / "flux2_edit_lora.yaml")
    process = template["config"]["process"][0]
    process["datasets"][0]["resolution"] = [512]
    process["train"]["lr"] = p["lora_lr"]
    template_path = output / "template.yaml"
    template_path.write_text(yaml.safe_dump(template, sort_keys=False))
    plan = dict(
        status="preparing",
        data=str(data.resolve()),
        cohort=str(cohort.resolve()),
        cohort_sha256=sha256(cohort),
        protocol=p,
        protocol_path=str(protocol.resolve()),
        protocol_sha256=sha256(protocol),
        validation=validation,
        jobs=[],
    )
    first = p["train_seeds"][0]
    specs = [("smoke", "full_role", first, 2, 1), ("capacity", "full_role", first, 2, 6)]
    specs += [
        ("train", mode, seed, p["steps"], p["batch_size"])
        for seed in p["train_seeds"]
        for mode in p["methods"]
    ]
    for phase, mode, seed, steps, batch in specs:
        name = f"{phase}_{mode}_{seed}"
        directory = output / name
        directory.mkdir()
        view = directory / "view"
        create_view(data, cohort, view, p["prompt"], p["train_groups"])
        config = build_config(
            SimpleNamespace(
                steps=steps,
                batch_size=batch,
                rank=p["rank"],
                template=template_path,
                name=name,
                output_root=directory / "lora",
                data_root=data,
                control="none",
                control_root=None,
            )
        )
        attach_features(config, data, cohort, mode, p["encoder"], p["feature_lr"])
        meta = config["meta"]["role_features"]
        meta.update(
            training_view=str((view / "view.json").resolve()),
            training_view_sha256=sha256(view / "view.json"),
            observation=dict(
                updates=sorted({u for u in p["observe_updates"] if u <= steps} | {steps}),
                steps=steps,
                save_every=p["save_every"],
            ),
        )
        process = config["config"]["process"][0]
        process["train"]["seed"] = seed
        # Own end-step hook saves exact update numbers; upstream save_every is zero-based.
        process["save"].update(save_every=0, max_step_saves_to_keep=10)
        process["datasets"][0].update(
            folder_path=str((view / "target").resolve()),
            control_path=[str((view / "reference").resolve())],
        )
        verify_features(config)
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


def load_plan(study: Path):
    plan = json.loads((study / "plan.json").read_text())
    p = plan["protocol"]
    verify_protocol(p)
    if (
        plan["status"] != "prepared"
        or sha256(plan["protocol_path"]) != plan["protocol_sha256"]
        or p != json.loads((study / "protocol.json").read_text())
        or sha256(plan["cohort"]) != plan["cohort_sha256"]
    ):
        raise ValueError("Prepared overfit protocol or cohort changed")
    require_dev(Path(plan["data"]), Path(plan["cohort"]))
    expected = {("train", m, s) for m in p["methods"] for s in p["train_seeds"]}
    expected |= {(phase, "full_role", p["train_seeds"][0]) for phase in ["smoke", "capacity"]}
    declared = [(j["phase"], j["method"], j["seed"]) for j in plan["jobs"]]
    if set(declared) != expected or len(declared) != len(expected):
        raise ValueError("Incomplete or duplicate overfit matrix")
    for job in plan["jobs"]:
        path = Path(job["directory"]) / "train.yaml"
        if sha256(path) != job["config_sha256"]:
            raise ValueError("Prepared overfit config changed")
        config = load_yaml(path)
        meta = verify_features(config)
        view = verify_view(Path(meta["training_view"]), root=Path(plan["data"]))
        process = config["config"]["process"][0]
        steps = p["steps"] if job["phase"] == "train" else p["smoke"]["steps"]
        batch = 1 if job["phase"] == "smoke" else p["batch_size"]
        if (
            meta["mode"] != job["method"]
            or view["prompt"] != p["prompt"]
            or process["train"]["steps"] != steps
            or process["train"]["batch_size"] != batch
            or process["train"]["seed"] != job["seed"]
            or process["train"]["lr"] != p["lora_lr"]
            or meta["lr"] != p["feature_lr"]
            or meta["encoder"] != p["encoder"]
            or process["network"]["linear"] != p["rank"]
            or view["groups"] != p["train_groups"]
        ):
            raise ValueError("Overfit job differs from its fixed matched budget")
    return plan


def completed(job):
    path = Path(job["directory"]) / "train_report.json"
    report = json.loads(path.read_text())
    if (
        report["status"] != "complete"
        or report["config_sha256"] != job["config_sha256"]
        or sha256(report["checkpoint"]) != report["checkpoint_sha256"]
        or sha256(report["feature_checkpoint"]["path"]) != report["feature_checkpoint"]["sha256"]
        or len(report["feature_statistics"]["observation"]["loss"]) != report["steps"]
        or report["feature_statistics"]["batches"] != report["steps"]
    ):
        raise ValueError("Incomplete, skipped or changed overfit training")
    return report


def infer_job(job, study):
    completed(job)
    directory = Path(job["directory"])
    return execute(
        [
            sys.executable,
            "-m",
            "src",
            "gpu",
            "--",
            sys.executable,
            "-m",
            "src.inference.learnability",
            "--study",
            str(study.resolve()),
            "--job",
            job["name"],
            "--output",
            str(directory / "inference"),
        ],
        directory / "inference_execution",
    )


def run(study: Path, phase: str):
    plan = load_plan(study)
    if phase not in {"smoke", "capacity", "train", "infer"}:
        raise ValueError("Unsupported learnability phase")
    if (study / f"{phase}_phase.json").exists():
        raise FileExistsError("Preserve completed phases; prepare a fresh study for a retry")
    if phase != "smoke":
        report = completed(next(j for j in plan["jobs"] if j["phase"] == "smoke"))
        probe = json.loads((study / "gpu_smoke/report.json").read_text())
        if (
            probe["status"] != "complete"
            or probe["checkpoint_sha256"] != report["checkpoint_sha256"]
        ):
            raise ValueError("Real overfit forward/backward/reload smoke has not passed")
    if phase in {"train", "infer"}:
        capacity = completed(next(j for j in plan["jobs"] if j["phase"] == "capacity"))
        if capacity["peak_reserved_gib"] > capacity["gpu_total_gib"] * 0.8:
            raise ValueError("Measured batch6 leaves insufficient OOM headroom")
    jobs = [j for j in plan["jobs"] if j["phase"] == ("train" if phase == "infer" else phase)]
    outcomes = {}
    with ThreadPoolExecutor(
        max_workers=1 if phase in {"smoke", "capacity"} else plan["protocol"]["max_parallel"]
    ) as pool:
        futures = {
            pool.submit(infer_job, j, study) if phase == "infer" else pool.submit(train_job, j): j
            for j in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                outcomes[job["name"]] = dict(status="complete")
            except BaseException as error:
                outcomes[job["name"]] = dict(status="failed", error=repr(error))
            write_json(study / f"{phase}_phase.json", outcomes)
    if any(r["status"] != "complete" for r in outcomes.values()):
        raise RuntimeError("Overfit phase failed; preserve logs and outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    p = actions.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--protocol", type=Path, default=CONFIG_ROOT / "experiments/role_learnability.yaml"
    )
    p = actions.add_parser("run")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--phase", choices=["smoke", "capacity", "train", "infer"], required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.cohort, args.output, args.protocol)
    else:
        run(args.study, args.phase)


if __name__ == "__main__":
    main()
