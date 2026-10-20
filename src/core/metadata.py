"""Stable timestamps, package versions and CLI configuration snapshots."""

import importlib.metadata
from datetime import datetime
from zoneinfo import ZoneInfo


def config_snapshot(args) -> dict:
    return {key: value for key, value in vars(args).items() if key != "func"}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def versions() -> dict[str, str]:
    packages = ("torch", "diffusers", "transformers", "peft", "datasets", "huggingface-hub")
    return {name: importlib.metadata.version(name) for name in packages}
