"""Run the locked upstream trainer with local text weights and hardware telemetry."""

import argparse
import json
import os
import time
from pathlib import Path

from src.core.io import sha256, write_json
from src.core.metadata import now, versions
from src.infra.gpu import require_lease
from src.infra.process import command
from src.training.checkpoints import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--report", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    require_lease()
    root = args.project_root.resolve()
    config, report = Path(args.config).resolve(), Path(args.report).resolve()
    lock = json.loads((root / "configs/assets.lock.json").read_text())
    trainer = root / "external/ai-toolkit"
    revision = command(["git", "-C", str(trainer), "rev-parse", "HEAD"])
    if revision != lock["repositories"]["ai-toolkit"]["revision"]:
        raise ValueError("ai-toolkit revision does not match assets.lock.json")
    if report.exists():
        raise FileExistsError(report)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    if Path.cwd().resolve() != trainer.resolve():
        raise RuntimeError("Start the trainer through src.training; worker cwd must be ai-toolkit")
    import torch
    import yaml
    from accelerate.utils import set_seed
    from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein4BModel
    from toolkit.job import get_job

    # The upstream architecture exposes this class attribute, not a YAML field.
    Flux2Klein4BModel.flux2_klein_te_path = str(root / "models/klein_text_encoder")
    configuration = yaml.safe_load(config.read_text())["config"]
    settings = configuration["process"][0]
    set_seed(settings["train"].get("seed", 42))
    torch.cuda.reset_peak_memory_stats()
    result = dict(
        created=now(),
        status="running",
        trainer_revision=revision,
        config=str(config),
        config_sha256=sha256(config),
        packages=versions(),
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
    )
    write_json(report, result)
    start = time.perf_counter()
    try:
        job = get_job(str(config))
        job.run()
        job.cleanup()
        checkpoint = Path(settings["training_folder"]) / configuration["name"]
        checkpoint = checkpoint / f"{configuration['name']}.safetensors"
        result.update(validate_checkpoint(checkpoint))
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        torch.cuda.synchronize()
        result.update(
            seconds=time.perf_counter() - start,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        write_json(report, result)


if __name__ == "__main__":
    main()
