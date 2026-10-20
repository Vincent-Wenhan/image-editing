"""Stable timestamps, package versions and CLI configuration snapshots."""

import hashlib
import importlib.metadata
import json
from datetime import datetime
from zoneinfo import ZoneInfo


def config_snapshot(args) -> dict:
    return {key: value for key, value in vars(args).items() if key != "func"}


def config_digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def versions() -> dict[str, str]:
    packages = ("torch", "diffusers", "transformers", "peft", "datasets", "huggingface-hub")
    return {name: importlib.metadata.version(name) for name in packages}
