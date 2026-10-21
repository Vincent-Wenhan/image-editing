"""Upper-bound LoRA inputs keep labels separate from target and preserve the objective."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.backends.flux2_oracle import oracle_references
from src.cli import build_parser
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.types import GenerationOptions
from src.datasets.manifest import load_edit_input
from src.datasets.role_controls import RoleControlDataset, export_role_controls
from src.training.spatial import attach_controls, verify_joint_batch, verify_training_controls

# Reuse clearly synthetic evidence; no fabricated renders or human results.


@pytest.mark.parametrize("kind", ["oracle_role", "dynamic_binary"])
def test_role_export_training_and_inference_use_identical_two_references(
    scene_data, tmp_path, kind
):
    controls = tmp_path / "controls"
    export_role_controls(scene_data, controls, kind)
    data = RoleControlDataset(controls, scene_data, kind)
    recipe = ProjectSettings.load().recipe
    process = recipe["config"]["process"][0]
    original_train = copy.deepcopy(process["train"])
    process["datasets"][0].update(folder_path=str(scene_data / "train/target"), resolution=[64])
    attach_controls(recipe, scene_data, controls, kind)
    assert verify_training_controls(recipe)["purpose"] == "oracle_upper_bound"
    assert process["train"] == original_train
    assert process["datasets"][0]["control_path"] == [
        str(scene_data / "train/reference"),
        str(controls / "train" / data.folder),
    ]
    row = read_rows(scene_data / "dev/manifest.csv")[1]
    edit = load_edit_input(scene_data, row)
    bundle = data.bundle(row, (64, 64))
    with pytest.raises(ValueError, match="ordinary inference"):
        bundle.require_inference_safe()
    references = oracle_references(
        edit, bundle, GenerationOptions(64, 64, 2, 4, 42), allow_oracle=True
    )
    assert np.array_equal(references[1], data.image(row))
    for path in (scene_data / "dev/target").glob("*.png"):
        path.unlink()
    for path in (scene_data / "raw").glob("*/*.npz"):
        path.unlink()
    # Consumption, unlike export and training, never needs these supervision images.
    assert np.array_equal(data.image(row), references[1])
    entry = data.index["splits"]["dev"]["entries"][row["id"]]
    path = controls / entry["path"]
    Image.new("RGB", (64, 64), "white").save(path)
    entry["sha256"] = sha256(path)
    write_json(controls / "controls.json", data.index)
    with pytest.raises(ValueError, match="encoding"):
        RoleControlDataset(controls, scene_data, kind)


def test_actual_role_batch_checks_one_hot_and_does_not_change_loss_or_gradients():
    torch = pytest.importorskip("torch")
    from src.training.flux2_compat import SUPPORTED_TRAINER_REVISION, configure_flux2_trainer

    class Model(torch.nn.Module):
        _image_edit_joint_controls = True
        _image_edit_joint_batches = 0
        _image_edit_condition_kind = "oracle_role"

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(3, 3, 1)
            self.vae = SimpleNamespace(device=torch.device("cpu"))

        def get_noise_prediction(self, latent, *args, batch=None, **kwargs):
            return self.layer(batch.control_tensor_list[0][0])

    torch.manual_seed(2)
    baseline = Model()
    adapted = copy.deepcopy(baseline)
    original = Model.get_noise_prediction
    configure_flux2_trainer(
        Model, __import__("pathlib").Path("."), revision=SUPPORTED_TRAINER_REVISION
    )
    source = torch.rand(3, 64, 64)
    roles = torch.zeros_like(source)
    roles[2] = 1
    roles[0, :10], roles[2, :10] = 1, 0
    batch = SimpleNamespace(control_tensor_list=[[source, roles]], control_tensor=None)
    latent = torch.zeros(1, 128, 4, 4)
    left = original(baseline, latent, 1, None, 4, batch=batch).square().mean()
    right = adapted.get_noise_prediction(latent, 1, None, 4, batch=batch).square().mean()
    assert torch.equal(left, right)
    left.backward()
    right.backward()
    assert torch.equal(baseline.layer.weight.grad, adapted.layer.weight.grad)
    with pytest.raises(ValueError, match="binary mask"):
        verify_joint_batch(batch, latent)
    roles[1] = 1
    with pytest.raises(ValueError, match="one-hot"):
        verify_joint_batch(batch, latent, kind="oracle_role")
    roles[:] = 0.5
    with pytest.raises(ValueError, match="one-hot"):
        verify_joint_batch(batch, latent, kind="oracle_role")


def test_oracle_cli_permission_is_explicit():
    args = build_parser().parse_args(["infer", "--control", "oracle_role", "--allow-oracle"])
    assert args.allow_oracle and args.control == "oracle_role"


def test_matched_training_rejects_recipe_or_asset_drift(scene_data, tmp_path):
    import json

    import yaml

    from src.evaluation.fairness import matched_training
    from src.training.flux2_compat import SUPPORTED_TRAINER_REVISION

    reports = {}
    for method in ("oracle_role", "dynamic_binary"):
        controls = tmp_path / method
        export_role_controls(scene_data, controls, method)
        recipe = ProjectSettings.load().recipe
        process = recipe["config"]["process"][0]
        process["datasets"][0].update(folder_path=str(scene_data / "train/target"), resolution=[64])
        attach_controls(recipe, scene_data, controls, method)
        config = tmp_path / f"{method}.yaml"
        config.write_text(yaml.safe_dump(recipe))
        checkpoint = tmp_path / f"{method}.bin"
        checkpoint.write_bytes(b"CPU fixture checkpoint bytes")
        path = tmp_path / f"{method}.json"
        write_json(
            path,
            dict(
                status="complete",
                trainer_revision=SUPPORTED_TRAINER_REVISION,
                config=str(config),
                config_sha256=sha256(config),
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),
                steps=150,
                batch_size=1,
                joint_control_batches=150,
                conditioning=recipe["meta"]["spatial_control"],
            ),
        )
        reports[method] = path
    assert matched_training(reports)["status"] == "verified"
    config = tmp_path / "dynamic_binary.yaml"
    recipe = yaml.safe_load(config.read_text())
    recipe["config"]["process"][0]["train"]["lr"] *= 2
    config.write_text(yaml.safe_dump(recipe))
    report = json.loads(reports["dynamic_binary"].read_text())
    report["config_sha256"] = sha256(config)
    write_json(reports["dynamic_binary"], report)
    with pytest.raises(ValueError, match="budget differs"):
        matched_training(reports)


@pytest.mark.parametrize("kind", ["oracle_role", "dynamic_binary"])
def test_upper_bound_runner_records_and_verifies_but_requires_permission(
    scene_data, tmp_path, monkeypatch, kind
):
    import json

    from src.core.types import EditOutput
    from src.evaluation.provenance import verify_run
    from src.inference import runner

    monkeypatch.setattr(runner, "require_lease", lambda: None)
    model = tmp_path / "model"
    model.mkdir()
    (model / "download.json").write_text('{"revision":"fixture"}')
    controls = tmp_path / "controls"
    export_role_controls(scene_data, controls, kind)
    calls = []

    class Backend:
        def __init__(self, *args):
            pass

        def edit(self, *args):
            raise AssertionError("Ordinary edit must not receive oracle labels")

        def edit_oracle(self, edit, control, options, *, allow_oracle):
            assert allow_oracle and not hasattr(edit, "target")
            calls.append(control.kind)
            return EditOutput(edit.source, 0, 0, 0)

    argv = [
        "infer",
        "--model",
        str(model),
        "--data-root",
        str(scene_data),
        "--manifest",
        str(scene_data / "dev/manifest.csv"),
        "--control",
        kind,
        "--control-root",
        str(controls),
        "--side",
        "64",
        "--output",
        str(tmp_path / "denied"),
    ]
    with pytest.raises(ValueError, match="allow-oracle"):
        runner.run(build_parser().parse_args(argv), backend_factory=Backend)
    argv[-1] = str(tmp_path / "upper_bound")
    runner.run(build_parser().parse_args(argv + ["--allow-oracle"]), backend_factory=Backend)
    record = json.loads((tmp_path / "upper_bound/run.json").read_text())
    assert record["status"] == "complete" and len(calls) == 2
    verify_run(scene_data, scene_data / "dev/manifest.csv", record)
