"""Launch the pinned external trainer using normal module imports in its cwd."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.core.config import load_assets
from src.core.paths import PROJECT_ROOT
from src.infra.gpu import require_lease
from src.infra.process import command


def launch(config: Path, report: Path, root: Path = PROJECT_ROOT) -> None:
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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.training.worker",
            str(config),
            "--report",
            str(report),
            "--project-root",
            str(root),
        ],
        cwd=trainer,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    launch(args.config, args.report)
