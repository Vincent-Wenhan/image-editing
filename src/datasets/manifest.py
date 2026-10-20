"""Separate inference input loading from training/assessment supervision loading."""

from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from src.core.paths import data_path
from src.core.types import EditInput, EditSupervision


def load_edit_input(root: str | Path, row: Mapping[str, str]) -> EditInput:
    """Stage 0 uses source/instruction only, even if the manifest contains labels."""
    with Image.open(data_path(root, row["source_path"])) as image:
        source = image.convert("RGB")
    return EditInput(source=source, instruction=row["instruction"])


def load_supervision(root: str | Path, row: Mapping[str, str]) -> EditSupervision:
    with Image.open(data_path(root, row["target_path"])) as image:
        target = image.convert("RGB")
    return EditSupervision(target=target)
