"""Joint-control recipes and actual raw batches fail instead of dropping masks."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from src.cli import build_parser
from src.conditioning.controls import ControlKind
from src.conditioning.static import MaskSpec
from src.core.config import ProjectSettings
from src.core.io import read_rows
from src.datasets.control_export import export_controls
from src.training.config import build_config
from src.training.flux2_compat import SUPPORTED_TRAINER_REVISION, configure_flux2_trainer
from src.training.spatial import verify_joint_batch, verify_training_controls


@pytest.fixture(params=[ControlKind.STATIC_MASK, ControlKind.CONSTANT_MASK])
def joint_recipe(spatial_data, tmp_path, request):
    model = tmp_path / "model"
    model.mkdir()
    (model / "ae.safetensors").touch()
    recipe = ProjectSettings.load().recipe
    process = recipe["config"]["process"][0]
    process["model"].update(name_or_path=str(model), vae_path=str(model / "ae.safetensors"))
    process["datasets"][0]["resolution"] = [64]
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(recipe))
    root = tmp_path / "controls"
    export_controls(spatial_data, root, MaskSpec(request.param))
    args = build_parser().parse_args(
        [
            "train-config",
            "--data-root",
            str(spatial_data),
            "--template",
            str(template),
            "--control",
            str(request.param),
            "--control-root",
            str(root),
        ]
    )
    return build_config(args)


def test_joint_recipe_preserves_original_objective(joint_recipe, spatial_data):
    process = joint_recipe["config"]["process"][0]
    baseline = ProjectSettings.load().process
    assert process["train"] == baseline["train"]
    assert process["network"] == baseline["network"]
    paths = process["datasets"][0]["control_path"]
    assert paths[0] == str(spatial_data / "train/reference")
    assert paths[1].endswith("controls/train/mask_rgb")
    assert all("target" not in path for path in paths)
    assert verify_training_controls(joint_recipe)["order"] == ["source", "mask"]
    # Exercise the actual disk protocol, not just the in-memory recipe.
    reloaded = yaml.safe_load(yaml.safe_dump(joint_recipe))
    assert reloaded == joint_recipe
    assert verify_training_controls(reloaded) == joint_recipe["meta"]["spatial_control"]


@pytest.mark.parametrize(
    "change", ["order", "missing", "resize", "override", "augment", "index", "target"]
)
def test_training_preflight_rejects_inconsistent_conditions(joint_recipe, change):
    config = copy.deepcopy(joint_recipe)
    dataset = config["config"]["process"][0]["datasets"][0]
    if change == "order":
        dataset["control_path"].reverse()
    elif change == "missing":
        Path(dataset["control_path"][1]).joinpath(
            next(Path(dataset["control_path"][1]).iterdir()).name
        ).unlink()
    elif change == "resize":
        dataset["full_size_control_images"] = False
    elif change == "override":
        dataset["control_path_2"] = dataset["folder_path"]
    elif change == "augment":
        dataset["flip_x"] = True
    elif change == "index":
        path = Path(config["meta"]["spatial_control"]["root"]) / "controls.json"
        path.write_text(json.dumps(json.loads(path.read_text()), indent=3))
    else:
        root = Path(config["meta"]["spatial_control"]["data_root"])
        row = read_rows(root / "train/manifest.csv")[0]
        Image.new("RGB", (64, 64)).save(root / row["target_path"])
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_training_controls(config)


def test_device_patch_checks_actual_two_control_batches(tmp_path):
    torch = pytest.importorskip("torch")
    calls = []

    class Model:
        _image_edit_joint_controls = True
        _image_edit_joint_batches = 0
        vae = SimpleNamespace(device=torch.device("cpu"))

        def get_noise_prediction(self, *args, **kwargs):
            calls.append(kwargs["batch"])
            return "prediction"

    configure_flux2_trainer(Model, tmp_path, revision=SUPPORTED_TRAINER_REVISION)
    source, mask = torch.rand(3, 64, 64), torch.zeros(3, 64, 64)
    latent = torch.zeros(1, 128, 4, 4)
    batch = SimpleNamespace(control_tensor_list=[[source, mask]], control_tensor=None)
    assert Model().get_noise_prediction(latent, 1, None, 4, batch=batch) == "prediction"
    assert Model._image_edit_joint_batches == 1 and calls == [batch]
    for images in ([source], [mask, source], [source, torch.ones(3, 64, 64) * 0.5]):
        bad = SimpleNamespace(control_tensor_list=[images], control_tensor=None)
        with pytest.raises(ValueError):
            Model().get_noise_prediction(latent, 1, None, 4, batch=bad)
    assert len(calls) == 1
    with pytest.raises(ValueError):
        verify_joint_batch(SimpleNamespace(control_tensor_list=None), latent)


@pytest.mark.parametrize("change", [None, "optimizer", "rank", "config_hash", "checkpoint"])
def test_budget_reuse_checks_recipe_and_real_reference_assets(joint_recipe, tmp_path, change):
    from src.core.io import sha256
    from src.evaluation.budget import verify_training_budget

    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(joint_recipe))
    baseline = copy.deepcopy(joint_recipe)
    baseline["meta"].pop("spatial_control")
    dataset = baseline["config"]["process"][0]["datasets"][0]
    dataset["control_path"] = dataset["control_path"][0]
    dataset.pop("full_size_control_images")
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(baseline))
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"test checkpoint bytes")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            dict(
                status="complete",
                trainer_revision=SUPPORTED_TRAINER_REVISION,
                config=str(path),
                config_sha256=sha256(path),
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),
                steps=150,
                batch_size=1,
            )
        )
    )
    if change == "config_hash":
        path.write_text(path.read_text() + "\n")
    elif change == "checkpoint":
        checkpoint.write_bytes(b"changed")
    elif change in {"optimizer", "rank"}:
        process = joint_recipe["config"]["process"][0]
        if change == "optimizer":
            process["train"]["optimizer"] = "sgd"
        else:
            process["network"]["linear"] = 32
        candidate.write_text(yaml.safe_dump(joint_recipe))
    if change:
        with pytest.raises(ValueError):
            verify_training_budget(report, candidate)
    else:
        assert verify_training_budget(report, candidate)["train_count"] == 3
