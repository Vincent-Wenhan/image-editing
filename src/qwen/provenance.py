"""Qwen asset and external-repository provenance checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.core.config import load_assets
from src.core.io import sha256

QWEN_ASSET = "qwen_image_edit_2511"
INTER_EDIT_REPOSITORY = "Inter-Edit"


def locked_qwen_asset(lock_path: str | Path = "configs/assets.lock.json") -> dict:
    assets = load_assets(lock_path)
    try:
        return dict(assets["models"][QWEN_ASSET])
    except KeyError as error:
        raise ValueError(f"Missing {QWEN_ASSET} in the asset lock") from error


def verify_external_repo(
    path: str | Path = "external/Inter-Edit",
    lock_path: str | Path = "configs/assets.lock.json",
) -> dict:
    """Require the pinned Inter-Edit revision and a clean tracked worktree."""
    path = Path(path).resolve()
    if not (path / ".git").exists():
        raise FileNotFoundError(f"Pinned external repository is missing: {path}")
    assets = load_assets(lock_path)
    expected = assets["repositories"][INTER_EDIT_REPOSITORY]["revision"]
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if actual != expected:
        raise ValueError(f"Inter-Edit revision differs: expected {expected}, got {actual}")
    if status:
        raise ValueError("Inter-Edit has tracked local changes; restore the pinned source first")
    return dict(path=str(path), revision=actual, clean=True)


def verify_model_download(
    path: str | Path | None = None,
    lock_path: str | Path = "configs/assets.lock.json",
) -> dict:
    """Require a completed local snapshot whose receipt matches the asset lock."""
    asset = locked_qwen_asset(lock_path)
    root = Path(path or asset["path"]).resolve()
    receipt_path = root / "download.json"
    if not receipt_path.exists():
        raise FileNotFoundError(
            f"Qwen model is not downloaded at {root}; run image-edit download --asset {QWEN_ASSET}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key in ("repo_id", "revision", "path"):
        expected = asset[key]
        actual = receipt.get(key)
        if key == "path":
            actual = str(Path(actual).resolve()) if actual else actual
            expected = str(Path(expected).resolve())
        if actual != expected:
            raise ValueError(f"Qwen download receipt differs for {key}: {actual!r}")
    return dict(asset=asset, path=str(root), receipt_sha256=sha256(receipt_path))
