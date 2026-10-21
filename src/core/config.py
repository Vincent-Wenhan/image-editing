"""Read canonical assets, data selection and experiment defaults once per command.

The full legacy ai-toolkit YAML remains a supported, directly runnable recipe.
Its values also provide CLI training and generation defaults instead of a second
set of Python constants. Explicit CLI overrides retain their existing meaning.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.paths import CONFIG_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_assets(path: str | Path = CONFIG_ROOT / "assets.lock.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ProjectSettings:
    assets: dict[str, Any]
    dataset: dict[str, Any]
    experiment: dict[str, Any]
    recipe: dict[str, Any]

    @classmethod
    def load(cls, root: Path = CONFIG_ROOT) -> "ProjectSettings":
        return cls(
            load_assets(root / "assets.lock.json"),
            load_yaml(root / "datasets/magicbrush.yaml"),
            load_yaml(root / "experiments/stage0.yaml"),
            load_yaml(root / "flux2_edit_lora.yaml"),
        )

    @property
    def process(self) -> dict[str, Any]:
        return self.recipe["config"]["process"][0]

    @property
    def manifest(self) -> str:
        return str(Path(self.dataset["root"]) / "dev/manifest.csv")
