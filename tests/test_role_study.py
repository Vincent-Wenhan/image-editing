"""Reject partial/mislabeled studies before a GPU lease or result can be created."""

import copy

import pytest

from src.training.role_study import verify_plan


def test_partial_preparation_allows_only_ready_pilot_not_an_incomplete_comparison():
    plan = dict(
        status="preparing",
        protocol=dict(train_seeds=[42, 43], methods=["static_mask", "oracle_role"]),
        jobs=[dict(phase="smoke", method="oracle_role", seed=42)],
    )
    verify_plan(plan, "smoke")
    with pytest.raises(ValueError, match="every planned"):
        verify_plan(plan, "train")
    with pytest.raises(ValueError, match="not prepared"):
        verify_plan(plan, "capacity")
    plan["jobs"] += [dict(phase="capacity", method="oracle_role", seed=42)]
    plan["jobs"] += [
        dict(phase="train", method=m, seed=s)
        for m in plan["protocol"]["methods"]
        for s in plan["protocol"]["train_seeds"]
    ]
    plan["status"] = "prepared"
    verify_plan(plan, "train")
    bad = copy.deepcopy(plan)
    bad["jobs"].append(bad["jobs"][0])
    with pytest.raises(ValueError, match="identities"):
        verify_plan(bad, "infer")
    with pytest.raises(ValueError, match="Unknown"):
        verify_plan(plan, "unknown")
