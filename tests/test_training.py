"""Recipe semantics, external worker isolation and adapter validation."""

import json
from pathlib import Path

import pytest
import yaml

from src.cli import build_parser
from src.core.config import ProjectSettings
from src.training import launcher
from src.training.config import build_config, run


@pytest.fixture
def training_args(paired_data, tmp_path):
    settings = ProjectSettings.load()
    recipe = settings.recipe
    model = tmp_path / "model"
    model.mkdir()
    vae = model / "ae.safetensors"
    vae.touch()
    recipe["config"]["process"][0]["model"].update(name_or_path=str(model), vae_path=str(vae))
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(recipe))
    return build_parser().parse_args(
        [
            "train-config",
            "--template",
            str(template),
            "--data-root",
            str(paired_data),
            "--steps",
            "20",
            "--batch-size",
            "2",
            "--rank",
            "32",
            "--name",
            "debug",
            "--output-root",
            str(tmp_path / "lora"),
            "--output",
            str(tmp_path / "debug.yaml"),
        ]
    )


def test_recipe_preserves_training_objective_and_pairing(training_args, paired_data):
    generated = build_config(training_args)
    process = generated["config"]["process"][0]
    original = ProjectSettings.load().process
    expected_train = original["train"] | {"steps": 20, "batch_size": 2}
    assert process["train"] == expected_train
    assert process["model"]["arch"] == "flux2_klein_4b"
    assert process["network"]["linear"] == process["network"]["linear_alpha"] == 32
    assert process["save"]["save_every"] == 20
    assert process["datasets"][0] == original["datasets"][0] | {
        "folder_path": str((paired_data / "train/target").resolve()),
        "control_path": str((paired_data / "train/reference").resolve()),
    }
    assert process["sample"] == original["sample"]
    assert process["train"]["seed"] == process["sample"]["seed"]
    assert process["datasets"][0]["resolution"] == [process["sample"]["width"]]
    run(training_args)
    assert yaml.safe_load(Path(training_args.output).read_text()) == generated
    with pytest.raises(FileExistsError):
        run(training_args)


def test_custom_template_fields_and_legacy_save_cadence(training_args):
    template = Path(training_args.template)
    config = yaml.safe_load(template.read_text())
    config["config"]["process"][0]["train"]["lr"] = 0.00003
    config["config"]["process"][0]["save"]["save_every"] = 80
    template.write_text(yaml.safe_dump(config))
    training_args.steps = 100
    process = build_config(training_args)["config"]["process"][0]
    assert process["train"]["lr"] == 0.00003
    assert process["save"]["save_every"] == 50


def test_launcher_uses_worker_cwd_and_checks_revision(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/assets.lock.json").write_text(
        json.dumps({"repositories": {"ai-toolkit": {"revision": "locked"}}})
    )
    trainer = tmp_path / "external/ai-toolkit"
    trainer.mkdir(parents=True)
    config = tmp_path / "debug.yaml"
    config.write_text("job: extension")
    report = tmp_path / "report.json"
    monkeypatch.setattr(launcher, "require_lease", lambda: None)
    monkeypatch.setattr(launcher, "command", lambda _: "locked")
    calls = []
    monkeypatch.setattr(
        launcher, "wait_for_worker", lambda args, trainer: calls.append((args, trainer))
    )
    previous_cwd = Path.cwd()
    launcher.launch(config, report, tmp_path)
    argv, kwargs = calls[0]
    assert argv[1:3] == ["-m", "src.training.worker"]
    assert str(config.resolve()) in argv and str(report.resolve()) in argv
    assert kwargs == trainer
    assert Path.cwd() == previous_cwd
    monkeypatch.setattr(launcher, "command", lambda _: "different")
    with pytest.raises(ValueError, match="revision"):
        launcher.launch(config, report, tmp_path)
    assert len(calls) == 1


def test_saved_adapter_must_have_finite_lora_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from src.training.checkpoints import validate_checkpoint

    checkpoint = tmp_path / "adapter.safetensors"
    safetensors.save_file({"lora_A.weight": torch.ones(2, 2)}, checkpoint)
    assert validate_checkpoint(checkpoint)["checkpoint_finite"]
    safetensors.save_file({"lora_A.weight": torch.tensor([float("nan")])}, checkpoint)
    with pytest.raises(ValueError, match="non-finite"):
        validate_checkpoint(checkpoint)
    safetensors.save_file({"unrelated.weight": torch.ones(1)}, checkpoint)
    with pytest.raises(ValueError, match="no LoRA"):
        validate_checkpoint(checkpoint)
