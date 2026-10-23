"""Known synthetic references calibrate proxies; uncertainty must remain visible."""

import numpy as np
import pytest

from src.evaluation.scene_semantics import TranslationLocator, change_proxies, correlation


def scene():
    background = np.full((64, 96, 3), 0.5)
    source, target = background.copy(), background.copy()
    old, new = np.zeros((64, 96), bool), np.zeros((64, 96), bool)
    old[20:32, 12:24], new[20:32, 54:66] = True, True
    template = np.full((12, 12, 3), [0.1, 0.6, 0.8])
    template[:, :5] = [0.05, 0.2, 0.3]
    source[20:32, 12:24], target[20:32, 54:66] = template, template
    return source, target, background, old, new


def test_fft_valid_correlation_agrees_with_direct_arithmetic():
    image = np.arange(35).reshape(5, 7)
    kernel = np.array([[1, 2], [3, 4]])
    actual = correlation(image, kernel)
    expected = np.array(
        [[np.sum(image[y : y + 2, x : x + 2] * kernel) for x in range(6)] for y in range(4)]
    )
    assert np.allclose(actual, expected, atol=1e-10)


def test_locator_calibrates_identity_translation_and_duplicate_candidates():
    source, target, background, old, new = scene()
    locator = TranslationLocator(source, target, background, old, new)
    assert locator.calibration["status"] == "calibrated"
    exact, retained = locator.locate(target), locator.locate(source)
    assert exact["position_error_px"] == 0 and not exact["template_duplicate_candidate"]
    assert retained["position_error_px"] == 42 and retained["distance_to_old_px"] == 0
    duplicated = target.copy()
    duplicated[old] = source[old]
    assert locator.locate(duplicated)["template_duplicate_candidate"]
    assert locator.locate(background)["status"] == "abstain"
    identity = TranslationLocator(source, source, background, old, old)
    assert identity.locate(source)["position_error_px"] == 0


def test_low_contrast_or_changed_geometry_abstains_instead_of_claiming_success():
    source, target, bg, old, new = scene()
    locator = TranslationLocator(bg, bg, bg, old, new)
    assert locator.calibration["status"] == "abstain"
    assert locator.locate(target)["position_error_px"] is None
    with pytest.raises(ValueError, match="geometry"):
        TranslationLocator(source, target[:2], bg, old, new)


@pytest.mark.parametrize("invalid", ["nan", "range", "occupancy"])
def test_locator_rejects_invalid_numeric_or_mask_contracts(invalid):
    source, target, bg, old, new = scene()
    if invalid == "nan":
        source[0, 0] = np.nan
    elif invalid == "range":
        source[0, 0] = 2
    else:
        old = old[:2]
    with pytest.raises(ValueError, match="aligned boolean"):
        TranslationLocator(source, target, bg, old, new)


def test_dependency_creation_and_removal_are_distinct_from_preservation():
    source, target, bg, old, new = scene()
    source[40:45, 12:24] = 0
    target[40:45, 54:66] = 0
    labels = np.full(old.shape, 2)
    labels[old | new] = 0
    labels[40:45, 12:24] = labels[40:45, 54:66] = 1
    valid = np.ones(old.shape, bool)
    exact = change_proxies(source, target, target, bg, labels, valid)
    identity = change_proxies(source, target, source, bg, labels, valid)
    assert exact["d_signed_change_response"] == 1
    assert exact["d_shadow_creation_mae"] == exact["d_shadow_removal_mae"] == 0
    assert identity["d_signed_change_response"] == 0
    assert identity["d_shadow_creation_mae"] == identity["d_shadow_removal_mae"] == 0.5
    assert exact["d_dark_region_iou"] == 1 and identity["d_dark_region_iou"] == 0
    assert identity["p_edge_mae"] == 0
    invalid = np.zeros_like(valid)
    assert (
        change_proxies(source, target, source, bg, labels, invalid)["d_signed_change_response"]
        is None
    )
