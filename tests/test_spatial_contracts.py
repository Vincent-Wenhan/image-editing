"""Geometry, finite-value and probability constraints without implementing controls."""

import numpy as np
import pytest
from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.core.types import EditInput, EditSupervision


def test_soft_masks_and_role_fields_preserve_values_and_geometry():
    image = Image.new("RGB", (3, 2))
    mask = np.full((2, 3), 0.25, dtype=np.float32)
    roles = np.full((2, 3, 3), 1 / 3, dtype=np.float32)
    edit = EditInput(image, "move", mask=mask, strength=0.5)
    assert edit.mask is mask
    control = ControlBundle(ControlKind.SOFT_MASK, {"mask": mask})
    control.validate_geometry(image.size)
    role = ControlBundle(ControlKind.PREDICTED_ROLE, {"roles": roles})
    role.validate_geometry(image.size)
    EditSupervision(image, role_labels=roles, valid_region=np.ones((2, 3), dtype=bool))
    assert np.array_equal(control.maps["mask"], mask)
    with pytest.raises(ValueError, match="dimensions"):
        role.validate_geometry((2, 3))
    with pytest.raises(ValueError, match="binary"):
        ControlBundle(ControlKind.STATIC_MASK, {"mask": mask})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_strength_values_are_finite_unit_scalars(value):
    with pytest.raises(ValueError, match="Strength"):
        EditInput(Image.new("RGB", (3, 2)), "move", strength=value)


@pytest.mark.parametrize("change", ["nan", "range", "sum", "shape", "integer_rgb"])
def test_invalid_soft_role_fields_are_rejected(change):
    roles = np.full((2, 3, 3), 1 / 3, dtype=np.float32)
    if change == "nan":
        roles[0, 0, 0] = np.nan
    elif change == "range":
        roles[0, 0] = [-0.1, 0.5, 0.6]
    elif change == "sum":
        roles[0, 0] = [0.2, 0.2, 0.2]
    elif change == "shape":
        roles = roles[:, :, :2]
    else:
        roles = Image.new("RGB", (3, 2))
    with pytest.raises(ValueError):
        EditSupervision(Image.new("RGB", (3, 2)), role_labels=roles)


def test_hard_roles_and_validity_do_not_mark_ignore_pixels_valid():
    labels = np.array([[0, 1, 2], [255, 0, 2]], dtype=np.uint8)
    valid = labels != 255
    target = Image.new("RGB", (3, 2))
    EditSupervision(target, role_labels=labels, valid_region=valid)
    with pytest.raises(ValueError, match="Ignored"):
        EditSupervision(target, role_labels=labels, valid_region=np.ones_like(valid))
    with pytest.raises(ValueError, match="dimensions"):
        EditSupervision(target, role_labels=labels[:1])
    with pytest.raises(ValueError, match="binary"):
        EditSupervision(target, valid_region=np.full((2, 3), 0.25))
    with pytest.raises(TypeError, match="RGB"):
        EditInput(Image.new("RGBA", (3, 2)), "edit")
