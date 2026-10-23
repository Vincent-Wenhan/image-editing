"""Archived directions distinguish perfect conditional editing from generic source denoising."""

import numpy as np
import pytest

from src.evaluation.direction_analysis import score_report, summarize
from src.evaluation.direction_gates import collapse_times


def bits(value):
    return (np.asarray(value, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_archived_strength_contrasts_and_d_knockout_are_scored_separately(tmp_path):
    anchor = np.zeros((6, 2), np.float32)
    epsilon = np.ones_like(anchor)
    target = anchor.copy()
    target[:2] = 0.5
    target[2:4] = -0.5
    masks = np.repeat(np.eye(3), 2, axis=1).astype(np.float32)
    source = tmp_path / "source.npz"
    np.savez(source, anchor=anchor, epsilon=epsilon)
    state = tmp_path / "state.npz"
    np.savez(state, x=epsilon)
    group = dict(input_path=str(source), supervision={})
    report = dict(
        job=dict(family="original", method="full_role", seed=307), groups={"g": group}, results=[]
    )
    for strength in [0.0, 1.0]:
        truth = anchor if strength == 0 else target
        supervision = tmp_path / f"target{strength}.npz"
        np.savez(supervision, target=truth, masks=masks)
        identifier = f"id{strength}"
        group["supervision"][identifier] = dict(path=str(supervision))
        variants = {}
        for variant in ["original", "zero", "shuffle", "swap_old_new", "swap_dp", "drop_d"]:
            velocity = epsilon - truth if variant == "original" else epsilon
            path = tmp_path / f"{identifier}_{variant}.npz"
            np.savez(
                path,
                positive=bits(velocity),
                negative=bits(velocity),
                cfg_native=bits(velocity),
                fused=bits(epsilon),
                residual=np.zeros_like(epsilon),
            )
            variants[variant] = dict(path=str(path), fusion={})
        report["results"].append(
            dict(
                id=identifier,
                group_id="g",
                split="train",
                strength=strength,
                state=dict(family="source_bridge", time=1.0, actual_time=1.0, path=str(state)),
                variants=variants,
            )
        )
    rows, mechanisms = score_report(
        report,
        dict(
            inherited=dict(inference=dict(guidance=4)), protocol=dict(gate=dict(energy_floor=1e-12))
        ),
    )
    assert len(rows) == 48 and len(mechanisms) == 12
    full = next(
        r
        for r in rows
        if r["strength"] == 1 and r["variant"] == "original" and r["branch"] == "cfg_native"
    )
    assert full["t_contrast_gain"] == pytest.approx(1) and full["d_drop_d_gain"] == pytest.approx(1)
    assert full["t_endpoint_mse"] == pytest.approx(0) and full["p_endpoint_mse"] == pytest.approx(0)
    zero = next(
        r
        for r in rows
        if r["strength"] == 1 and r["variant"] == "zero" and r["branch"] == "cfg_native"
    )
    assert zero["t_contrast_gain"] == 0 and zero["t_endpoint_mse"] > 0
    summary = next(
        r for r in summarize(rows) if r["variant"] == "original" and r["branch"] == "cfg_native"
    )
    assert summary["metrics"]["t_contrast_gain"] == 1 and summary["identity_p_mse"] == 0


def test_time_collapse_preserves_paired_case_identity_before_scene_bootstrap():
    rows = [
        dict(id="x", group_id="g", strength=1.0, score=0.2),
        dict(id="x", group_id="g", strength=1.0, score=0.4),
        dict(id="y", group_id="h", strength=1.0, score=0.8),
    ]
    result = collapse_times(rows, "score")
    assert (
        len(result) == 2 and result[0]["score"] == pytest.approx(0.3) and result[1]["score"] == 0.8
    )
