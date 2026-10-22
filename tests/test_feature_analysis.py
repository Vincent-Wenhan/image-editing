"""Synthetic numeric fixtures test decision rules, never fabricated human/model results."""

import copy

from src.conditioning.features import FEATURE_MODES
from src.evaluation.feature_analysis import numeric_gate, scene_mean


def test_feature_gate_requires_all_regions_identity_and_old_new_improvement():
    baseline = dict(
        nonzero=dict(t_target_mae=0.2, d_target_mae=0.2, p_source_drift=0.1),
        by_strength={"0.0": dict(identity_mae=0.1)},
    )
    methods = {mode: copy.deepcopy(baseline) for mode in FEATURE_MODES}
    assert not numeric_gate(methods)["passed"]
    methods["full_role"]["nonzero"].update(t_target_mae=0.1, d_target_mae=0.1)
    assert numeric_gate(methods)["passed"]
    methods["full_role"]["by_strength"]["0.0"]["identity_mae"] += 0.001
    assert not numeric_gate(methods)["passed"]
    methods["full_role"]["by_strength"]["0.0"]["identity_mae"] = 0.1
    methods["old_new"]["nonzero"]["d_target_mae"] = 0.05
    assert not numeric_gate(methods)["passed"]
    methods["old_new"]["nonzero"]["d_target_mae"] = None
    assert not numeric_gate(methods)["passed"]


def test_old_new_geometry_aggregation_is_scene_equal_and_keeps_missing_undefined():
    rows = [
        dict(group_id="a", error=0),
        dict(group_id="a", error=0),
        dict(group_id="b", error=1),
        dict(group_id="c", error=None),
    ]
    assert scene_mean(rows, "error") == 0.5
    assert scene_mean([rows[-1]], "error") is None
