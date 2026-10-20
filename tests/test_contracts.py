"""Inference has no target/role access and unsupported controls fail explicitly."""

from dataclasses import fields

import pytest
from PIL import Image

from src.backends.base import EditingBackend
from src.backends.flux2 import Flux2Backend
from src.conditioning.controls import ControlBundle, ControlKind
from src.core.io import read_rows
from src.core.types import EditInput, EditSupervision, GenerationOptions
from src.datasets.manifest import load_edit_input, load_supervision


def test_inference_loading_never_opens_supervision(paired_data):
    row = read_rows(paired_data / "dev/manifest.csv")[0]
    target = load_supervision(paired_data, row)
    assert isinstance(target, EditSupervision)
    (paired_data / row["target_path"]).unlink()
    (paired_data / row["mask_path"]).unlink()
    row.update(role_labels="missing-oracle.png", valid_region="missing-valid.png")
    edit_input = load_edit_input(paired_data, row)
    assert edit_input.source.mode == "RGB"
    assert edit_input.mask is edit_input.strength is None
    assert {field.name for field in fields(EditInput)} == {
        "source",
        "instruction",
        "mask",
        "strength",
    }


def test_oracle_and_metadata_cannot_silently_enter_inference():
    oracle = ControlBundle(ControlKind.ORACLE_ROLE, {"roles": Image.new("RGB", (2, 2))})
    with pytest.raises(ValueError, match="Oracle"):
        oracle.require_inference_safe()
    with pytest.raises(TypeError, match="supervision"):
        ControlBundle(metadata={"target": Image.new("RGB", (2, 2))})
    with pytest.raises(ValueError, match="Unconditional"):
        ControlBundle(maps={"mask": Image.new("L", (2, 2))})


@pytest.mark.parametrize("kind", [kind for kind in ControlKind if kind != ControlKind.NONE])
def test_unimplemented_conditions_are_rejected(kind):
    # No model loading or CUDA import is needed to check the protocol boundary.
    backend = object.__new__(Flux2Backend)
    assert isinstance(backend, EditingBackend)
    error = ValueError if kind == ControlKind.ORACLE_ROLE else NotImplementedError
    with pytest.raises(error):
        backend.edit(
            EditInput(Image.new("RGB", (32, 32)), "edit"),
            ControlBundle(kind, {"control": Image.new("L", (32, 32))}),
            GenerationOptions(32, 32, 1, 4.0, 42),
        )


def test_input_strength_and_mask_geometry_are_explicit():
    with pytest.raises(ValueError, match="Strength"):
        EditInput(Image.new("RGB", (32, 32)), "edit", strength=1.1)
    with pytest.raises(ValueError, match="dimensions"):
        EditInput(Image.new("RGB", (32, 32)), "edit", mask=Image.new("L", (16, 16)))
