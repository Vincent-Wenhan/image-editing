"""Tiny local pairs exercise the same export/validation paths as real data."""

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.metadata import config_digest
from src.core.paths import CONFIG_ROOT
from src.datasets.export import export_rows
from src.datasets.scene_config import scene_group
from src.datasets.scene_pairs import compile_pairs


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


def evidence(color, strength=1):
    background = np.full((64, 64, 3), 0.4, dtype=np.float32)
    old = np.zeros((64, 64), dtype=np.float32)
    old[10:26, 10:26] = 1
    new = np.zeros_like(old)
    new[10:26, 34:50] = 1
    source_off, target_off = background.copy(), background.copy()
    source_off[old > 0] = color
    target_off[new > 0] = color
    source, target = source_off.copy(), target_off.copy()
    source[40:48, 10:20] -= 0.1
    target[40:48, 38:48] -= 0.1
    if strength == 0:
        target, target_off, new = source.copy(), source_off.copy(), old.copy()
    return dict(
        source=source,
        target=target,
        source_no_shadow=source_off,
        target_no_shadow=target_off,
        source_occupancy=old,
        target_occupancy=new,
        background=background,
    )


@pytest.fixture
def scene_data(tmp_path):
    config = load_yaml(CONFIG_ROOT / "datasets/blender_translation.yaml")
    config["renderer"].update(resolution=64, samples=1)
    config["scenes"] = [config["scenes"][0], config["scenes"][2]]
    config["strengths"] = [0.0, 1.0]
    root = tmp_path / "scenes"
    root.mkdir()
    (root / "scene_config.yaml").write_text(yaml.safe_dump(config))
    groups = []
    for spec in config["scenes"]:
        folder = root / "raw" / spec["id"]
        folder.mkdir(parents=True)
        entries, files = [], {}
        for index, strength in enumerate(config["strengths"]):
            path = folder / f"s{index:03d}.npz"
            np.savez_compressed(path, **evidence(spec["color"], strength))
            entries.append(
                dict(
                    strength=strength,
                    displacement=strength * config["max_translation_widths"] * spec["object_width"],
                    evidence_path=path.relative_to(root / "raw").as_posix(),
                    sha256=sha256(path),
                )
            )
        # Hash-only EXR fixture: unit tests do not claim real Blender execution.
        raw = folder / "background.exr"
        raw.write_bytes(b"unit-test fixture, not an EXR")
        files[raw.relative_to(root / "raw").as_posix()] = sha256(raw)
        groups.append(
            dict(scene=spec, group=scene_group(config, spec), entries=entries, raw_files=files)
        )
    write_json(
        root / "raw/index.json",
        dict(
            schema_version=1,
            renderer="4.5.3",
            device="CPU",
            config_sha256=config_digest(config),
            groups=groups,
        ),
    )
    compile_pairs(root, config)
    return root
