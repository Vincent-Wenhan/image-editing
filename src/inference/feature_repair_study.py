"""Freeze the single post-diagnostic hypothesis before any repair GPU inference."""

import argparse
import json
from pathlib import Path

from src.conditioning.feature_residuals import RESIDUAL_VARIANTS
from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.inference.feature_diagnostic_study import load_plan, phase_outputs, run_jobs


def prepare(study: Path, config: Path):
    destination = study / "comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    plan, recipe = load_plan(study), load_yaml(config)
    expected = dict(
        schema_version=1,
        study="frozen_centered_role_residual_v1",
        training_updates=0,
        variants=list(RESIDUAL_VARIANTS),
        smoke_samples=["dev_2026102201_000_s000", "dev_2026102201_000_s001"],
    )
    if recipe != expected:
        raise ValueError("Use the single preregistered frozen residual intervention")
    completed = phase_outputs(study, "diagnose")
    if set(completed) != {str(s) for s in plan["protocol"]["train_seeds"]}:
        raise ValueError("Complete both diagnostic seeds before choosing the repair")
    write_json(
        destination,
        dict(
            recipe=recipe,
            config_path=str(config.resolve()),
            config_sha256=sha256(config),
            plan_sha256=sha256(study / "plan.json"),
            diagnostics={s: sha256(path / "report.json") for s, path in completed.items()},
        ),
    )


def load_comparison(study: Path):
    plan = load_plan(study)
    record = json.loads((study / "comparison.json").read_text())
    if (
        record["plan_sha256"] != sha256(study / "plan.json")
        or sha256(record["config_path"]) != record["config_sha256"]
        or load_yaml(record["config_path"]) != record["recipe"]
        or record["diagnostics"]
        != {s: sha256(p / "report.json") for s, p in phase_outputs(study, "diagnose").items()}
    ):
        raise ValueError("Repair recipe or completed diagnostic provenance changed")
    return plan, record


def run(study: Path, phase: str, attempt: int):
    plan, _record = load_comparison(study)
    p = plan["protocol"]
    if phase not in {"repair_smoke", "compare"}:
        raise ValueError("Unsupported repair phase")
    if phase == "compare":
        smoke_path = phase_outputs(study, "repair_smoke")[str(p["train_seeds"][0])]
        smoke = json.loads((smoke_path / "report.json").read_text())
        if not all(smoke["checks"].values()):
            raise ValueError("Real repair GPU smoke must pass before the comparison")
    seeds = p["train_seeds"][:1] if phase == "repair_smoke" else p["train_seeds"]
    run_jobs(study, phase, seeds, p["max_parallel"], attempt, "src.inference.feature_repair")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    prepare_parser.add_argument("--study", type=Path, required=True)
    prepare_parser.add_argument(
        "--config", type=Path, default=CONFIG_ROOT / "experiments/role_feature_repair.yaml"
    )
    launch = actions.add_parser("run")
    launch.add_argument("--study", type=Path, required=True)
    launch.add_argument("--phase", choices=["repair_smoke", "compare"], required=True)
    launch.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.study, args.config)
    else:
        run(args.study, args.phase, args.attempt)


if __name__ == "__main__":
    main()
