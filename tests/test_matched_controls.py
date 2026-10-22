"""Uninformative second references and opt-in casting preserve the old baseline."""

import json

import numpy as np
import pytest
from PIL import Image

from src.backends.flux2_conditions import reference_images
from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.static import MaskSpec, transform_mask
from src.core.io import read_rows
from src.core.references import reference_preprocessing
from src.core.types import EditInput, GenerationOptions
from src.datasets.control_export import export_controls
from src.datasets.controls import ControlDataset


def test_constant_control_is_content_independent_and_keeps_both_references(spatial_data, tmp_path):
    root = tmp_path / "constant"
    spec = MaskSpec(ControlKind.CONSTANT_MASK)
    export_controls(spatial_data, root, spec)
    data = ControlDataset(root, spatial_data, "constant_mask")
    values = []
    for split in ("train", "dev"):
        for row in read_rows(spatial_data / split / "manifest.csv"):
            image = data.image(row)
            values.append(np.asarray(image))
            source = Image.new("RGB", image.size, "blue")
            references = reference_images(
                EditInput(source, "edit"),
                data.bundle(row, image.size),
                GenerationOptions(*image.size, 2, 4, 42),
            )
            assert len(references) == 2 and references[0] is source
            assert np.asarray(references[1]).max() == 0
    assert all(np.array_equal(value, values[0]) for value in values)
    assert np.array_equal(transform_mask(Image.new("L", (64, 64), 255), spec), values[0][:, :, 0])
    # A modified constant must fail even with a refreshed file SHA in the index.
    from src.core.io import sha256

    index = json.loads((root / "controls.json").read_text())
    entry = next(iter(index["splits"]["dev"]["entries"].values()))
    Image.new("RGB", (64, 64), "white").save(root / entry["path"])
    entry["sha256"] = sha256(root / entry["path"])
    (root / "controls.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="declared mask transform"):
        ControlDataset(root, spatial_data, "constant_mask")


def test_casting_is_explicit_and_historical_default_is_unchanged():
    assert reference_preprocessing("none") == "diffusers"
    assert reference_preprocessing("none", "training") == "training"
    assert reference_preprocessing("static_mask") == reference_preprocessing("constant_mask")
    source = Image.new("RGB", (64, 64))
    edit = EditInput(source, "edit")
    options = GenerationOptions(64, 64, 2, 4, 42)
    assert reference_images(edit, ControlBundle(), options) is source
    explicit = ControlBundle(metadata={"reference_preprocessing": "training"})
    assert reference_images(edit, explicit, options) is source
    with pytest.raises(ValueError, match="dimensions"):
        reference_images(edit, explicit, GenerationOptions(128, 64, 2, 4, 42))
