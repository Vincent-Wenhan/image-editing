"""Geometry is an explicit oracle; fixtures are not experiment outcomes."""

import numpy as np
import pytest

from src.backends.flux2_oracle import oracle_references
from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.geometry import geometry_rgb
from src.core.io import read_rows, sha256
from src.core.types import GenerationOptions
from src.datasets.geometry import load_geometry_condition
from src.datasets.manifest import load_edit_input
from src.datasets.role_controls import RoleControlDataset, export_role_controls


def test_geometry_separates_direction_and_keeps_identity_black(scene_data):
    identity, edited = read_rows(scene_data / "dev/manifest.csv")
    assert not np.asarray(geometry_rgb(load_geometry_condition(scene_data, identity).maps)).any()
    bundle = load_geometry_condition(scene_data, edited)
    rgb = np.asarray(geometry_rgb(bundle.maps))
    assert set(np.unique(rgb)) <= {0, 64, 128, 192, 255}
    assert (rgb == rgb[..., :1]).all()
    swapped = dict(bundle.maps, old=bundle.maps["new"], new=bundle.maps["old"])
    assert not np.array_equal(rgb, geometry_rgb(swapped))
    assert np.array_equal(bundle.maps["old"] | bundle.maps["new"], swapped["old"] | swapped["new"])
    assert np.array_equal(rgb > 0, np.asarray(geometry_rgb(swapped)) > 0)
    with pytest.raises(ValueError, match="ordinary inference"):
        bundle.require_inference_safe()


def test_geometry_inference_opens_occupancy_not_target_rgb(scene_data, monkeypatch):
    row = read_rows(scene_data / "dev/manifest.csv")[1]
    from numpy.lib.npyio import NpzFile

    original = NpzFile.__getitem__
    seen = []

    def guarded(self, key):
        assert key in {"source_occupancy", "target_occupancy"}
        seen.append(key)
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", guarded)
    (scene_data / row["target_path"]).unlink()
    bundle = load_geometry_condition(scene_data, row)
    assert seen == ["source_occupancy", "target_occupancy"]
    edit = load_edit_input(scene_data, row)
    options = GenerationOptions(64, 64, 2, 4, 42)
    with pytest.raises(ValueError, match="allow_oracle"):
        oracle_references(edit, bundle, options)
    refs = oracle_references(edit, bundle, options, allow_oracle=True)
    assert len(refs) == 2 and np.array_equal(refs[1], geometry_rgb(bundle.maps))


def test_geometry_export_detects_actual_evidence_and_reference_changes(scene_data, tmp_path):
    output = tmp_path / "controls"
    export_role_controls(scene_data, output, "oracle_geometry")
    dataset = RoleControlDataset(output, scene_data, "oracle_geometry")
    row = read_rows(scene_data / "dev/manifest.csv")[1]
    assert dataset.folder == "geometry_rgb"
    assert dataset.bundle(row, (64, 64)).kind == ControlKind.ORACLE_GEOMETRY
    path = scene_data / row["evidence_path"]
    path.write_bytes(path.read_bytes() + b"modified")
    assert sha256(path) != row["evidence_sha256"]
    with pytest.raises(ValueError, match="SHA256"):
        dataset.image(row)


def test_geometry_rejects_overlap_and_arbitrary_values():
    z = np.zeros((4, 4))
    with pytest.raises(ValueError, match="disjoint"):
        ControlBundle(ControlKind.ORACLE_GEOMETRY, {"old": z + 1, "new": z, "dependency": z + 1})
    with pytest.raises(ValueError, match="binary"):
        geometry_rgb({"old": z + 0.5, "new": z, "dependency": z})


def test_geometry_actual_training_batch_has_only_declared_codes():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from src.training.spatial import verify_joint_batch

    mask = torch.zeros(3, 64, 64)
    mask[:, :16] = 64 / 255
    batch = SimpleNamespace(
        control_tensor_list=[[torch.zeros_like(mask), mask]], control_tensor=None
    )
    latent = torch.zeros(1, 128, 4, 4)
    verify_joint_batch(batch, latent, kind="oracle_geometry")
    with pytest.raises(ValueError, match="binary"):
        verify_joint_batch(batch, latent, kind="dynamic_binary")
    mask[:, 0] = 0.5
    with pytest.raises(ValueError, match="grayscale"):
        verify_joint_batch(batch, latent, kind="oracle_geometry")


def test_geometry_configuration_keeps_objective_and_reference_order(scene_data, tmp_path):
    import copy

    from src.core.config import ProjectSettings
    from src.training.spatial import attach_controls, verify_training_controls

    output = tmp_path / "controls"
    export_role_controls(scene_data, output, "oracle_geometry")
    config = ProjectSettings.load().recipe
    process = config["config"]["process"][0]
    before = copy.deepcopy(process["train"])
    process["datasets"][0].update(folder_path=str(scene_data / "train/target"), resolution=[64])
    attach_controls(config, scene_data, output, "oracle_geometry")
    assert process["train"] == before
    assert process["datasets"][0]["control_path"] == [
        str(scene_data / "train/reference"),
        str(output / "train/geometry_rgb"),
    ]
    assert verify_training_controls(config)["purpose"] == "oracle_upper_bound"
