"""Explicit project and manifest path resolution without changing cwd."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"
if not CONFIG_ROOT.is_dir():
    CONFIG_ROOT = Path(sys.prefix) / "share/image-editing/configs"


def data_path(root: str | Path, relative: str) -> Path:
    """Manifest paths must remain inside their dataset root, including symlinks."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {path}")
    return path
