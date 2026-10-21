"""Tiny local pairs exercise the same export/validation paths as real data."""

import pytest
from PIL import Image

from src.datasets.export import export_rows


@pytest.fixture
def paired_data(tmp_path):
    root = tmp_path / "pairs"
    for split, color in (("train", "blue"), ("dev", "green")):
        export_rows(
            [
                dict(
                    img_id=split,
                    turn_index=1,
                    instruction="Make <square> red & retain background",
                    source_img=Image.new("RGB", (32, 48), color),
                    target_img=Image.new("RGB", (32, 48), "red"),
                    mask_img=Image.new("L", (32, 48), 255),
                )
            ],
            root,
            split,
            1,
        )
    return root
