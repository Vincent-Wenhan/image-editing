"""Isolated pinned trainer; failures keep their traceback through cleanup/reporting."""

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from src.core.config import ProjectSettings, load_yaml
from src.core.io import sha256, write_json
from src.core.metadata import now, versions
from src.infra.gpu import require_lease
from src.infra.process import command
from src.training.checkpoints import validate_checkpoint
from src.training.flux2_compat import configure_flux2_trainer
from src.training.lifecycle import attempt, gpu_statistics, release_job

logger = logging.getLogger(__name__)


def load_runtime(root: Path, lock: dict, revision: str):
    """Import only inside the leased worker, after checking the upstream revision."""
    import torch
    from accelerate.utils import set_seed
    from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein4BModel
    from toolkit.job import get_job

    configure_flux2_trainer(
        Flux2Klein4BModel, root / lock["models"]["training_text"]["path"], revision=revision
    )
    return SimpleNamespace(torch=torch, set_seed=set_seed, get_job=get_job)


def run_training(
    config: Path,
    report: Path,
    root: Path,
    *,
    runtime_loader=load_runtime,
    memory_plan: Path | None = None,
) -> None:
    require_lease()
    if report.exists():
        raise FileExistsError(report)
    result = dict(
        created=now(),
        status="running",
        config=str(config),
        gpu_uuid=os.environ.get("IMAGE_EDIT_GPU_UUID"),
    )
    write_json(report, result)
    started = time.perf_counter()
    job, runtime = None, None
    primary_failure = False
    try:
        lock = json.loads((root / "configs/assets.lock.json").read_text())
        trainer = root / "external/ai-toolkit"
        revision = command(["git", "-C", str(trainer), "rev-parse", "HEAD"])
        result["trainer_revision"] = revision
        if revision != lock["repositories"]["ai-toolkit"]["revision"]:
            raise ValueError("ai-toolkit revision does not match assets.lock.json")
        if Path.cwd().resolve() != trainer.resolve():
            raise RuntimeError("Worker cwd must be the locked ai-toolkit root")
        result.update(config_sha256=sha256(config), packages=versions())
        if memory_plan:
            result["memory_plan"] = json.loads(memory_plan.read_text())
        configuration = load_yaml(config)["config"]
        settings = configuration["process"][0]
        result.update(batch_size=settings["train"]["batch_size"], steps=settings["train"]["steps"])
        for key, value in {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
        }.items():
            os.environ[key] = value
        runtime = runtime_loader(root, lock, revision)
        runtime.set_seed(
            settings["train"].get("seed", ProjectSettings.load().process["train"]["seed"])
        )
        runtime.torch.cuda.reset_peak_memory_stats()
        write_json(report, result)
        job = runtime.get_job(str(config))
        job.run()
        checkpoint = Path(settings["training_folder"]) / configuration["name"]
        checkpoint /= f"{configuration['name']}.safetensors"
        result.update(validate_checkpoint(checkpoint))
        result["status"] = "complete"
    except BaseException as error:
        primary_failure = True
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        result["out_of_memory"] = bool(
            runtime is not None and isinstance(error, runtime.torch.cuda.OutOfMemoryError)
        )
        raise
    finally:
        telemetry_errors, cleanup_errors = {}, {}
        torch = runtime.torch if runtime else sys.modules.get("torch")
        if runtime is None and torch is not None:
            initialized = attempt(
                "cuda_initialized", lambda: torch.cuda.is_initialized(), telemetry_errors
            )
            if not initialized:
                torch = None
        if torch is not None:
            stats = attempt(
                "gpu_statistics", lambda: gpu_statistics(torch, telemetry_errors), telemetry_errors
            )
            result.update(stats or {})
        attempt("release_job", lambda: release_job(job, torch, cleanup_errors), cleanup_errors)
        job = None
        result["seconds"] = time.perf_counter() - started
        if telemetry_errors:
            result["telemetry_errors"] = telemetry_errors
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
        secondary = None
        if cleanup_errors and not primary_failure:
            secondary = RuntimeError(f"Training resource cleanup failed: {cleanup_errors}")
            result.update(status="failed", error=repr(secondary))
        try:
            write_json(report, result)
        except BaseException:
            logger.exception("Could not persist final training report: %s", report)
            if not primary_failure:
                raise
        if secondary:
            raise secondary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--memory-plan", type=Path)
    args = parser.parse_args()

    def terminate(signum, _frame):
        # Repeated group signals must not interrupt cleanup/report persistence.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        run_training(
            args.config.resolve(),
            args.report.resolve(),
            args.project_root.resolve(),
            memory_plan=args.memory_plan,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
