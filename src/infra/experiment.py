"""Record a subprocess against a frozen executable source snapshot, without shell parsing."""

import os
import subprocess
import time
import traceback
from pathlib import Path

from src.core.io import sha256, write_json
from src.core.metadata import now, versions
from src.core.paths import PROJECT_ROOT
from src.infra.process import command


def source_snapshot(root: Path = PROJECT_ROOT) -> dict:
    files = [
        *root.glob("src/**/*.py"),
        *root.glob("configs/**/*.yaml"),
        root / "configs/assets.lock.json",
    ]
    return dict(
        revision=command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        files={str(path.relative_to(root)): sha256(path) for path in sorted(files)},
    )


def execute(argv: list[str], output: Path) -> dict:
    """Caller launches GPU work through the UUID-lease CLI; keep original failures."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    result = dict(
        status="running",
        created=now(),
        argv=argv,
        cwd=str(Path.cwd()),
        source=source_snapshot(),
        packages=versions(),
        parent_cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    write_json(output / "execution.json", result)
    try:
        with (output / "process.log").open("w") as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
        result["returncode"] = process.returncode
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv)
        if source_snapshot() != result["source"]:
            raise RuntimeError("Executable source changed while the experiment was running")
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.perf_counter() - started
        write_json(output / "execution.json", result)
    return result
