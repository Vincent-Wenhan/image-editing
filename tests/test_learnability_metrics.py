"""Same-scene contrasts distinguish role use from learning an average edit."""

import copy

import numpy as np
import pytest

from src.core.config import load_yaml
from src.core.paths import CONFIG_ROOT
from src.evaluation.learnability_metrics import contrast_metrics, learnability_gates


def test_constant_perfect_and_reversed_edit_contrasts():
    source = np.zeros((4, 4, 3), np.float32)
    target = source.copy()
    target[:2] = 0.3
    target[2:] = -0.1
    labels = np.repeat([[0], [0], [1], [1]], 4, axis=1)
    reference = dict(source=source, target=target, labels=labels, valid=np.ones((4, 4), bool))
    # An arbitrary appearance bias cancels when both strengths share it.
    identity = source + 0.05
    for prediction, error, response in [
        (identity, 1, 0),
        (target + 0.05, 0, 1),
        (identity - target, 2, -1),
    ]:
        result = contrast_metrics(reference, prediction, identity)
        assert [result[f"{r}_contrast_mae"] for r in ["t", "d"]] == pytest.approx(
            [error] * 2, abs=1e-6
        )
        assert [result[f"{r}_contrast_response"] for r in ["t", "d"]] == pytest.approx(
            [response] * 2, abs=1e-6
        )
    reference["valid"][:] = False
    assert all(v is None for v in contrast_metrics(reference, target, identity).values())


def gate_fixture():
    region = dict(
        t_target_mae=0.03,
        d_target_mae=0.06,
        p_source_drift=0.003,
        identity_t_target_mae=0.08,
        identity_d_target_mae=0.12,
        t_contrast_mae=0.5,
        d_contrast_mae=0.5,
    )
    full = dict(nonzero=region, by_strength={"0.0": dict(target_mae=0.004)})
    worse = copy.deepcopy(full)
    worse["nonzero"].update(t_target_mae=0.04, d_target_mae=0.08)
    return {"full_role": full, "constant": worse, "old_new": copy.deepcopy(worse)}, {
        "zero": copy.deepcopy(worse),
        "shuffle": copy.deepcopy(worse),
    }


def test_fit_does_not_pass_when_role_permutations_do_nothing():
    gate = load_yaml(CONFIG_ROOT / "experiments/role_learnability.yaml")["gate"]
    methods, interventions = gate_fixture()
    assert learnability_gates(methods, interventions, gate)["passed"]
    interventions["shuffle"] = copy.deepcopy(methods["full_role"])
    result = learnability_gates(methods, interventions, gate)
    assert result["fit_pass"] and not result["role_usage_pass"] and not result["passed"]
    methods["full_role"]["nonzero"]["identity_d_target_mae"] = None
    methods["full_role"]["nonzero"]["t_contrast_mae"] = np.nan
    result = learnability_gates(methods, interventions, gate)
    assert not result["fit"]["d"] and not result["fit"]["t_contrast"]


def test_fixed_inference_matrix_has_no_test_or_extra_capacity():
    from src.inference.learnability import case_policy

    p = load_yaml(CONFIG_ROOT / "experiments/role_learnability.yaml")
    rows = [
        dict(id=f"{split}_{g}_{s}", group_id=f"{split}_{g}", strength=str(s), split=split)
        for split, n in [("train", 2), ("dev", 4)]
        for g in range(n)
        for s in [0, 0.5, 1]
    ]
    count, probes = 0, 0
    for _seed in p["train_seeds"]:
        for mode in p["methods"]:
            for update in [0, 20, 60, 120] if mode == "full_role" else [120]:
                for row in rows:
                    if update in {20, 60} and row["split"] != "train":
                        continue
                    names, probe = case_policy(mode, update, row, rows, p)
                    count += len(names)
                    probes += probe
    assert count == 356 and probes == 4


def test_semantic_references_refuse_sealed_test_before_loading(tmp_path):
    from src.evaluation.feature_diagnostic_analysis import scene_references

    with pytest.raises(ValueError, match="sealed Test"):
        scene_references(tmp_path, "test")


def test_full_batch_trace_rejects_same_capacity_but_duplicate_examples():
    from src.evaluation.learnability_training import verify_full_batch_trace

    trained = dict(
        steps=2, feature_statistics=dict(input_trace=[dict(ids=["a", "b"]), dict(ids=["b", "a"])])
    )
    verify_full_batch_trace(trained, {"a", "b"})
    trained["feature_statistics"]["input_trace"][1]["ids"] = ["a", "a"]
    with pytest.raises(ValueError, match="exactly once"):
        verify_full_batch_trace(trained, {"a", "b"})


def test_supervised_overlay_keeps_primary_scores_and_legacy_records(scene_data):
    from src.evaluation.feature_diagnostic_analysis import scene_references, score_prediction
    from src.evaluation.learnability_analysis import CONTRASTS, summarize
    from src.evaluation.learnability_supervised import patch_scores

    records, corrected = [], {}
    for identifier, reference in scene_references(scene_data, "train").items():
        row = reference["row"]
        teacher = [dict(time=0.5, velocity_mse=1.0, regions=dict(t=1.0, d=1.0, p=1.0))]
        records.append(
            dict(
                id=identifier,
                group_id=row["group_id"],
                strength=float(row["strength"]),
                update=120,
                variant="original",
                teacher=teacher,
                **score_prediction(reference, reference["source"]),
                **dict.fromkeys(CONTRASTS),
            )
        )
        corrected[(120, identifier)] = dict(
            teacher=dict(
                records=[dict(time=0.5, velocity_mse=0.5, regions=dict(t=0.5, d=0.5, p=0.5))]
            )
        )
    original = summarize(records)
    patched = patch_scores(original, corrected)
    assert all(original[k] == patched[k] for k in ["all", "nonzero", "by_strength"])
    assert original["teacher_by_time"]["0.5"]["velocity_mse"] == 1.0
    assert patched["teacher_by_time"]["0.5"]["velocity_mse"] == 0.5
    assert original["records"][0]["teacher"][0]["velocity_mse"] == 1.0
