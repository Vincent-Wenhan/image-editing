"""Separate target editing and protected background errors, with scene-level pairing."""

import numpy as np
import pytest

from src.evaluation.role_metrics import aggregate, paired_scene_ci, region_metrics


def test_role_metrics_do_not_reward_identity_as_edit_success():
    source = np.zeros((4, 4, 3), dtype=np.float32)
    target = source.copy()
    target[:2] = 1
    labels = np.full((4, 4), 2, dtype=np.uint8)
    labels[0], labels[1] = 0, 1
    valid = np.ones((4, 4), dtype=bool)
    identity = region_metrics(source, target, source, labels, valid)
    exact = region_metrics(source, target, target, labels, valid)
    assert identity["t_target_mae"] == identity["d_target_mae"] == 1
    assert identity["p_source_drift"] == exact["p_source_drift"] == 0
    assert exact["t_target_mae"] == exact["d_target_mae"] == 0
    assert exact["identity_t_target_mae"] == exact["identity_d_target_mae"] == 1
    valid[0] = False
    assert region_metrics(source, target, target, labels, valid)["t_target_mae"] is None


def test_scene_aggregation_and_pairing_are_equal_group_weight():
    rows = [
        dict(
            id=str(i),
            group_id="a" if i < 3 else "b",
            strength=i,
            **{
                k: 1 if i < 3 else 0
                for k in (
                    "t_target_mae",
                    "d_target_mae",
                    "p_source_drift",
                    "identity_target_mae",
                    "identity_t_target_mae",
                    "identity_d_target_mae",
                    "target_mae",
                    "changed_fraction",
                )
            },
        )
        for i in range(4)
    ]
    other = [dict(row, t_target_mae=0) for row in rows]
    assert aggregate(rows)["t_target_mae"] == 0.5
    result = paired_scene_ci(rows, other, "t_target_mae")
    assert result["mean_difference"] == 0.5 and result["scenes"] == 2
    assert result["ci95"] == [0, 1]
    with pytest.raises(ValueError):
        paired_scene_ci(rows, other[:1], "t_target_mae")
