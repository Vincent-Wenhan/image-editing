"""Launch the pinned external trainer and wait for its cleanup before releasing a lease."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from src.core.config import ProjectSettings, load_assets, load_yaml
from src.core.io import sha256, write_json
from src.core.metadata import now
from src.core.paths import PROJECT_ROOT
from src.infra.gpu import inventory, require_lease
from src.infra.process import command
from src.training.memory import plan_batch


def wait_for_worker(argv: list[str], trainer: Path) -> None:
    process = subprocess.Popen(argv, cwd=trainer)

    def forward(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)

    previous = signal.signal(signal.SIGTERM, forward)
    try:
        try:
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    if returncode:
        raise subprocess.CalledProcessError(returncode, argv)


def launch(
    config: Path, report: Path, root: Path = PROJECT_ROOT, memory_reference: Path | None = None
) -> None:
    require_lease()
    config, report = config.resolve(), report.resolve()
    trainer = root / "external/ai-toolkit"
    revision = command(["git", "-C", str(trainer), "rev-parse", "HEAD"])
    if (
        revision
        != load_assets(root / "configs/assets.lock.json")["repositories"]["ai-toolkit"]["revision"]
    ):
        raise ValueError("ai-toolkit revision does not match assets.lock.json")
    if report.exists():
        raise FileExistsError(report)
    if not config.is_file():
        raise FileNotFoundError(config)
    extra = []
    if memory_reference:
        card = next(
            card for card in inventory() if card["uuid"] == os.environ["IMAGE_EDIT_GPU_UUID"]
        )
        configuration = load_yaml(config)
        reference = json.loads(memory_reference.read_text())
        settings = ProjectSettings.load()
        policy = settings.experiment["acceptance"]["memory_policy"] | {
            "minimum_reference_steps": settings.experiment["acceptance"]["debug_steps"]
        }
        started = time.perf_counter()
        try:
            plan = plan_batch(configuration, reference, card, policy)
        except BaseException as error:
            write_json(
                report,
                dict(
                    created=now(),
                    status="failed",
                    phase="memory_preflight",
                    config=str(config),
                    config_sha256=sha256(config),
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    gpu_uuid=card["uuid"],
                    gpu_total_gib=card["total"] / 1024,
                    out_of_memory=False,
                    seconds=time.perf_counter() - started,
                ),
            )
            raise
        plan.update(
            reference=str(memory_reference.resolve()),
            reference_sha256=sha256(memory_reference),
            requested_config=str(config),
            requested_config_sha256=sha256(config),
        )
        effective = report.with_suffix(".effective.yaml")
        plan_path = report.with_suffix(".memory.json")
        if effective.exists() or plan_path.exists():
            raise FileExistsError("Memory-plan artifacts already exist; use a new report")
        effective.parent.mkdir(parents=True, exist_ok=True)
        configuration["config"]["process"][0]["train"]["batch_size"] = plan["effective_batch_size"]
        effective.write_text(yaml.safe_dump(configuration, sort_keys=False))
        write_json(plan_path, plan)
        config = effective
        extra = ["--memory-plan", str(plan_path)]
    wait_for_worker(
        [
            sys.executable,
            "-m",
            "src.training.worker",
            str(config),
            "--report",
            str(report),
            "--project-root",
            str(root),
            *extra,
        ],
        trainer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--memory-reference", type=Path, help="Verified debug report for capacity-bound batch"
    )
    args = parser.parse_args()
    launch(args.config, args.report, memory_reference=args.memory_reference)
