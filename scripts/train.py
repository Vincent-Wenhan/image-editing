"""Run the locked upstream trainer with local text weights and hardware telemetry."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from image_editing.common import command, now, sha256, versions, write_json
from image_editing.gpu import require_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    require_lease()
    root = Path(__file__).resolve().parents[1]
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
    os.chdir(trainer)
    sys.path.insert(0, str(trainer))
    import torch
    from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein4BModel
    from toolkit.job import get_job

    # The upstream architecture exposes this class attribute, not a YAML field.
    Flux2Klein4BModel.flux2_klein_te_path = str(root / "models/Qwen3-4B")
    torch.manual_seed(42)
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
