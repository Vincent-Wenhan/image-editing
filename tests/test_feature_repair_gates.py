"""Necessary repair gates distinguish edit, preservation, identity and spatial-sham benefit."""

import copy

from src.evaluation.feature_repair_analysis import numeric_gate


def methods():
    baseline = dict(
        nonzero=dict(t_target_mae=0.1, d_target_mae=0.2, p_source_drift=0.01),
        by_strength={"0.0": dict(target_mae=0.01)},
    )
    result = {
        name: copy.deepcopy(baseline)
        for name in ["original", "zero", "centered", "centered_shuffled"]
    }
    result["centered"]["nonzero"].update(t_target_mae=0.098, d_target_mae=0.196)
    return result


def test_repair_requires_both_edit_regions_and_preservation_not_only_pixel_sensitivity():
    gate = dict(edit_relative_improvement=0.01, preservation_tolerance=1e-4)
    m = methods()
    assert numeric_gate(m, gate)["passed"]
    m["centered"]["nonzero"]["d_target_mae"] = 0.2
    assert not numeric_gate(m, gate)["criteria"]["d"]
    m = methods()
    m["centered"]["nonzero"]["p_source_drift"] += 0.00011
    assert not numeric_gate(m, gate)["criteria"]["p"]
    m = methods()
    m["centered"]["by_strength"]["0.0"]["target_mae"] += 0.00011
    assert not numeric_gate(m, gate)["criteria"]["identity"]


def test_repair_requires_benefit_over_zero_and_validity_preserving_sham():
    gate = dict(edit_relative_improvement=0.01, preservation_tolerance=1e-4)
    for name in ["zero", "centered_shuffled"]:
        m = methods()
        m[name] = copy.deepcopy(m["centered"])
        assert not numeric_gate(m, gate)["criteria"][name]
    m = methods()
    m["centered"]["nonzero"]["t_target_mae"] = float("nan")
    assert not numeric_gate(m, gate)["passed"]
