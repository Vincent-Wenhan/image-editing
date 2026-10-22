"""Analytic interventions and evaluation proxies, never fabricated human ratings."""

import numpy as np
import pytest
from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.core.io import sha256, write_json
from src.datasets.holdout import require_dev
from src.evaluation.review_page import FAILURES, page
from src.evaluation.role_failures import geometry_metrics, palette_response
from src.inference.role_diagnostics import interventions


def test_interventions_keep_geometry_and_isolate_preservation_color():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]]], dtype=np.uint8)
    result = interventions("oracle_role", Image.fromarray(rgb), None)
    assert np.array_equal(np.array(result["neutral_p"])[..., :2], rgb[..., :2])
    assert not np.array(result["neutral_p"])[..., 2].any()
    assert np.array_equal(np.array(result["cycle_palette"]), np.roll(rgb, 1, axis=-1))
    assert all(x.size == (4, 1) for x in result.values())


def test_old_new_binary_interventions_are_not_presented_as_deployable():
    old = np.array([[1, 0]], dtype=bool)
    new = ~old
    bundle = ControlBundle(
        ControlKind.ORACLE_GEOMETRY, dict(old=old, new=new, dependency=old & new)
    )
    result = interventions("dynamic_binary", Image.new("RGB", (2, 1)), bundle)
    assert np.asarray(result["old_only"])[0, 0].tolist() == [255] * 3
    assert np.asarray(result["new_only"])[0, 1].tolist() == [255] * 3
    with pytest.raises(ValueError):
        interventions("static_mask", Image.new("RGB", (2, 1)), bundle)


def test_palette_alignment_and_old_new_errors_have_known_values():
    z = np.zeros((1, 2, 3))
    one = np.ones_like(z)
    selected = np.ones((1, 2), bool)
    assert palette_response(z, one, z, one, selected) == 1
    assert palette_response(z, z, z, one, selected) == 0
    assert palette_response(z, one, z, z, selected) is None
    maps = dict(old=np.array([[1, 0]], bool), new=np.array([[0, 1]], bool))
    scores = geometry_metrics(z, one, z, maps)
    assert scores["old_only_target_mae"] == scores["new_only_target_mae"] == 1
    assert scores["old_only_source_distance"] == 0


def test_sealed_test_and_manifest_drift_fail_closed(scene_data, tmp_path):
    (scene_data / "test").mkdir()
    (scene_data / "test/manifest.csv").write_bytes((scene_data / "dev/manifest.csv").read_bytes())
    seal = tmp_path / "cohort.json"
    write_json(
        seal,
        dict(
            schema_version=1,
            purpose="dev_only_failure_analysis",
            test_status="sealed",
            dataset_sha256=sha256(scene_data / "dataset.json"),
            manifests={
                s: sha256(scene_data / s / "manifest.csv") for s in ("train", "dev", "test")
            },
        ),
    )
    require_dev(scene_data, seal, scene_data / "dev/manifest.csv")
    with pytest.raises(ValueError, match="sealed Dev"):
        require_dev(scene_data, seal, scene_data / "test/manifest.csv")
    (scene_data / "test/manifest.csv").write_text("changed")
    with pytest.raises(ValueError, match="sealed Dev"):
        require_dev(scene_data, seal)


def test_human_form_explains_failure_categories_without_prefilled_scores():
    assert {
        "duplicate_object",
        "wrong_target_position",
        "role_color_copy",
        "identity_break",
    } <= set(FAILURES)
    html = page([dict(task_id="fixture", instruction="Move right")], reviewer="rater_a")
    assert "duplicate_object=旧位置残留" in html
    assert '<option value=""></option>' in html


def test_geometry_protocol_cannot_select_test(scene_data, tmp_path):
    from src.training.role_study import verify_plan

    plan = dict(
        status="prepared",
        data=str(scene_data),
        jobs=[],
        protocol=dict(
            study="role_geometry_pilot_v1",
            methods=["dynamic_binary", "oracle_geometry"],
            train_seeds=[101, 103],
            inference=dict(splits=["test"]),
        ),
    )
    with pytest.raises(ValueError, match="Dev only"):
        verify_plan(plan, "train")
