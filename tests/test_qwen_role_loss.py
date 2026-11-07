"""Contracts for Qwen latent-grid role supervision."""

import numpy as np
import pytest

from src.qwen.role_loss import build_region_masks, latent_masks


def evidence():
    shape = (8, 8)
    source = np.full((*shape, 3), 0.2, dtype=np.float32)
    target = source.copy()
    source_no_shadow = source.copy()
    target_no_shadow = target.copy()
    source[2:4, 1:3] = 0.1
    target[4:6, 4:6] = 0.1
    source_no_shadow[2:4, 1:3] = 0.2
    target_no_shadow[4:6, 4:6] = 0.2
    old = np.zeros(shape, dtype=np.float32)
    new = np.zeros(shape, dtype=np.float32)
    old[1:4, 1:4] = 1
    new[3:6, 3:6] = 1
    return {
        "source": source,
        "target": target,
        "source_no_shadow": source_no_shadow,
        "target_no_shadow": target_no_shadow,
        "source_occupancy": old,
        "target_occupancy": new,
    }


def test_role_regions_are_disjoint_and_latent_masks_preserve_roles():
    labels = np.full((8, 8), 2, dtype=np.uint8)
    labels[1:4, 1:4] = 0
    labels[3:6, 3:6] = 0
    valid = np.ones((8, 8), dtype=bool)
    regions = build_region_masks(evidence(), labels, valid, shadow_threshold=0.02)
    stacked = np.stack(list(regions.values()))
    assert np.all(stacked.sum(axis=0) <= 1)
    resized = latent_masks(regions, 4, 4)
    assert resized.shape == (5, 4, 4)
    assert np.all((resized >= 0) & (resized <= 1))
    assert resized[0].sum() > 0 and resized[1].sum() > 0


def test_weighted_region_loss_normalizes_area_and_records_empty_regions():
    torch = pytest.importorskip("torch")
    from src.qwen.role_loss import weighted_region_loss

    error = torch.ones(1, 1, 2, 4, 4)
    masks = torch.zeros(1, 5, 4, 4)
    masks[:, 0, :2] = 1
    masks[:, 1, 2:] = 1
    role_loss, report = weighted_region_loss(error, masks, min_pixels=2)
    assert torch.isclose(role_loss, torch.tensor(1.0))
    assert report["active_regions"] == pytest.approx(0.4)
    assert report["region_loss"]["d_remove"] is None
