"""Tiny local pairs exercise the same export/validation paths as real data."""

import pytest
from PIL import Image, ImageDraw

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


@pytest.fixture
def spatial_data(tmp_path):
    root = tmp_path / "spatial_pairs"
    for split, color in (("train", "blue"), ("dev", "green")):
        rows = []
        for index in range(3):
            mask = Image.new("L", (64, 64))
            ImageDraw.Draw(mask).rectangle((4 + index * 16, 12, 12 + index * 16, 20), fill=255)
            rows.append(
                dict(
                    img_id=f"{split}_{index}",
                    turn_index=1,
                    instruction="Make square red",
                    source_img=Image.new("RGB", (64, 64), color),
                    target_img=Image.new("RGB", (64, 64), "red"),
                    mask_img=mask,
                )
            )
        export_rows(rows, root, split, len(rows))
    return root
