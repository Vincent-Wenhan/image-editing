"""Synthetic numeric fixtures test decision rules, never fabricated human/model results."""

import copy

import numpy as np

from src.conditioning.features import FEATURE_MODES
from src.evaluation.feature_analysis import numeric_gate, scene_mean
from src.evaluation.role_metrics import aggregate, region_metrics


def test_feature_gate_requires_all_regions_identity_and_old_new_improvement():
    baseline = dict(
        nonzero=dict(t_target_mae=0.2, d_target_mae=0.2, p_source_drift=0.1),
        by_strength={"0.0": dict(target_mae=0.1)},
    )
    methods = {mode: copy.deepcopy(baseline) for mode in FEATURE_MODES}
    assert not numeric_gate(methods)["passed"]
    methods["full_role"]["nonzero"].update(t_target_mae=0.1, d_target_mae=0.1)
    assert numeric_gate(methods)["passed"]
    methods["full_role"]["by_strength"]["0.0"]["target_mae"] += 0.001
    assert not numeric_gate(methods)["passed"]
    methods["full_role"]["by_strength"]["0.0"]["target_mae"] = 0.1
    methods["old_new"]["nonzero"]["d_target_mae"] = 0.05
    assert not numeric_gate(methods)["passed"]
    methods["old_new"]["nonzero"]["d_target_mae"] = None
    assert not numeric_gate(methods)["passed"]


def test_gate_uses_actual_prediction_error_for_identity_metric_schema():
    source = np.zeros((4, 4, 3), dtype=np.float32)
    target = source.copy()
    target[:2] = 1
    labels = np.full((4, 4), 2, dtype=np.uint8)
    labels[0], labels[1] = 0, 1
    valid = np.ones((4, 4), dtype=bool)

    def scored(edit, identity):
        return dict(
            nonzero=aggregate(
                [dict(group_id="a", **region_metrics(source, target, edit, labels, valid))]
            ),
            by_strength={
                "0.0": aggregate(
                    [
                        dict(
                            group_id="a",
                            **region_metrics(
                                source, source, identity, np.full_like(labels, 2), valid
                            ),
                        )
                    ]
                )
            },
        )

    methods = {mode: scored(source, source) for mode in FEATURE_MODES}
    methods["full_role"] = scored(target, source)
    assert numeric_gate(methods)["passed"]
    methods["full_role"] = scored(target, source + 5e-5)
    assert numeric_gate(methods)["passed"]
    methods["full_role"] = scored(target, source + 0.001)
    identity = methods["full_role"]["by_strength"]["0.0"]
    assert identity["identity_target_mae"] == 0  # Dataset identity is still exact.
    assert identity["target_mae"] > 1e-4  # Prediction nevertheless breaks identity.
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
