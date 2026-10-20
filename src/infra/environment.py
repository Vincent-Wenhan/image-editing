"""Record software, storage, source revisions and optional leased CUDA smoke test."""

import platform
import shutil
import sys
from pathlib import Path

from src.core.io import write_json
from src.core.metadata import now, versions
from src.infra.gpu import inventory, require_lease
from src.infra.process import command


def run(args):
    result = dict(
        created=now(),
        python=sys.version,
        platform=platform.platform(),
        packages=versions(),
        gpus=inventory(),
        git={},
        disk_free_gib=round(shutil.disk_usage(".").free / 2**30, 1),
    )
    for path in (".", "external/ai-toolkit", "external/Inter-Edit"):
        if Path(path, ".git").exists():
            result["git"][path] = command(["git", "-C", path, "rev-parse", "HEAD"])
    if args.cuda:
        require_lease()
        import torch

        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        y = x @ x.T
        torch.cuda.synchronize()
        if not torch.isfinite(y).all():
            raise RuntimeError("CUDA BF16 matrix multiply produced non-finite results")
        result["cuda"] = dict(
            runtime=torch.version.cuda,
            name=torch.cuda.get_device_name(0),
            bf16=torch.cuda.is_bf16_supported(),
            matmul_finite=True,
        )
    write_json(args.output, result)
    print(result)


def configure(parser):
    parser.add_argument("--output", default="outputs/environment.json")
    parser.add_argument("--cuda", action="store_true")
    parser.set_defaults(func=run)
