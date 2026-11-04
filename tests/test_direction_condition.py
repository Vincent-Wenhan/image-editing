"""Geometry-only direction fields stay target-free and spatially aligned."""

import numpy as np
import pytest

from src.conditioning.direction import CHANNELS, build_direction_condition


def maps(shift: int = 0):
    old = np.zeros((32, 64), np.float32)
    new = np.zeros_like(old)
    old[8:16, 8:16] = 1
    new[8:16, 8 + shift : 16 + shift] = 1
    dependency = np.zeros_like(old)
    dependency[20:24, 20:28] = 1
    valid = np.ones_like(old)
    return old, new, dependency, valid


def test_direction_condition_encodes_rightward_centroid_shift_and_tokens():
    condition = build_direction_condition(*maps(16))
    assert condition.values.shape == (len(CHANNELS), 32, 64)
    assert condition.metadata["displacement"][0] > 0
    assert condition.metadata["displacement"][1] == 0
    assert condition.values[3, 0, 0] == condition.metadata["displacement"][0]
    assert condition.tokens().shape == (8, 7)
    assert len(condition.digest) == 64


def test_identity_geometry_has_zero_direction_and_is_finite():
    old, _, dependency, valid = maps()
    condition = build_direction_condition(old, old, dependency, valid)
    assert np.all(condition.values[3:5] == 0)
    assert np.isfinite(condition.values).all()


@pytest.mark.parametrize("change", ["shape", "range", "normalization"])
def test_direction_condition_rejects_invalid_contract(change):
    old, new, dependency, valid = maps(8)
    if change == "shape":
        new = new[:-1]
    elif change == "range":
        dependency[0, 0] = 2
    else:
        with pytest.raises(ValueError, match="normalization"):
            build_direction_condition(old, new, dependency, valid, normalize_by="pixels")
        return
    with pytest.raises(ValueError):
        build_direction_condition(old, new, dependency, valid)
