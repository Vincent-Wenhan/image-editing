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


def test_overlapping_shadow_strength_changes_are_supervised_and_ignore_is_excluded():
    values = evidence()
    labels = np.full((8, 8), 2, dtype=np.uint8)
    labels[1:4, 1:4] = labels[3:6, 3:6] = 0
    labels[6, :4] = 1
    valid = np.ones((8, 8), dtype=bool)
    # Both source/target remain dark: the previous appearance-only definition missed these.
    values["source"][6, :2] = 0.05
    values["target"][6, :2] = 0.12
    values["source"][6, 2:4] = 0.12
    values["target"][6, 2:4] = 0.05
    valid[6, 0] = False
    masks = build_region_masks(values, labels, valid)
    assert not masks["d_remove"][6, 0]
    assert masks["d_remove"][6, 1] and masks["d_create"][6, 2:4].all()
    assert np.count_nonzero(masks["d_remove"]) == 1


def test_latent_coverage_preserves_subpixel_area_without_uint8_rounding():
    masks = {
        key: np.zeros((16, 16), dtype=bool)
        for key in ("t_old", "t_new", "d_remove", "d_create", "p")
    }
    masks["d_remove"][0, 0] = True
    resized = latent_masks(masks, 1, 1)
    assert resized[2, 0, 0] == 1 / 256


def test_area_balance_gives_small_and_large_regions_equal_contribution_and_gradient():
    torch = pytest.importorskip("torch")
    from src.qwen.role_loss import weighted_region_loss

    error = torch.ones(1, 1, 1, 4, 4, requires_grad=True)
    error.data[..., 0, 0] = 9
    masks = torch.zeros(1, 5, 4, 4)
    masks[:, 0, 0, 0], masks[:, 4] = 1, 1
    masks[:, 4, 0, 0] = 0
    loss, _ = weighted_region_loss(error, masks, min_pixels=1)
    assert loss.item() == 5
    loss.backward()
    assert error.grad[..., 0, 0].item() == 0.5
    assert error.grad[..., 1, 1].item() == pytest.approx(0.5 / 15)
