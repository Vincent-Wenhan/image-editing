"""The optional runtime adapter preserves the trainer loss and releases its hooks."""

import copy
from types import SimpleNamespace

import pytest

from src.conditioning.features import EncoderConfig
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.training.feature_config import attach_features, verify_features


def seal_fixture(data, tmp_path):
    seal = tmp_path / "seal.json"
    write_json(
        seal,
        dict(
            schema_version=1,
            purpose="dev_only_failure_analysis",
            test_status="sealed",
            dataset_sha256=sha256(data / "dataset.json"),
            manifests={s: sha256(data / s / "manifest.csv") for s in ("train", "dev")},
        ),
    )
    return seal


def metadata_fixture(data):
    return dict(data_root=str(data), mode="full_role", encoder=EncoderConfig().as_dict(), lr=1e-3)


def test_extra_optimizer_and_forward_keep_flow_target_and_gradients(scene_data, tmp_path):
    torch = pytest.importorskip("torch")
    from src.training.feature_adapter import install_features
    from src.training.flux2_compat import SUPPORTED_TRAINER_REVISION

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(128, 128)

        def forward(self, *, x, x_ids):
            return self.layer(x)

    class Model:
        def __init__(self):
            self.transformer = Transformer()

        def get_noise_prediction(self, latent, *args, batch, **kwargs):
            ids = torch.cartesian_prod(
                torch.tensor([0, 10]), torch.arange(4), torch.arange(4), torch.tensor([0])
            )[None]
            packed = latent.flatten(2).transpose(1, 2)
            return self.transformer(x=torch.cat([packed, torch.zeros_like(packed)], 1), x_ids=ids)[
                :, :16
            ]

    class Trainer:
        def hook_add_extra_train_params(self, params):
            return params

    original = Model.get_noise_prediction
    extra = Trainer.hook_add_extra_train_params
    model = Model()
    baseline = copy.deepcopy(model)
    state = install_features(
        Model, metadata_fixture(scene_data), SUPPORTED_TRAINER_REVISION, trainer_type=Trainer
    )
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            install_features(
                Model,
                metadata_fixture(scene_data),
                SUPPORTED_TRAINER_REVISION,
                trainer_type=Trainer,
            )
        trainer = Trainer()
        trainer.train_config = SimpleNamespace(seed=211)
        trainer.device_torch = "cpu"
        rng = torch.random.get_rng_state()
        params = trainer.hook_add_extra_train_params(
            [dict(params=model.transformer.parameters(), lr=1e-4)]
        )
        assert len(params) == 2 and params[1]["lr"] == 1e-3
        assert len(params[0]["params"]) == 2
        assert torch.equal(rng, torch.random.get_rng_state())
        optimizer = torch.optim.AdamW(params)
        row = read_rows(scene_data / "train/manifest.csv")[1]
        batch = SimpleNamespace(
            file_items=[
                SimpleNamespace(
                    path=str(scene_data / row["target_path"]),
                    flip_x=False,
                    flip_y=False,
                    crop_x=0,
                    crop_y=0,
                )
            ],
            control_tensor_list=[[torch.rand(3, 64, 64)]],
            control_tensor=None,
        )
        latent, target = torch.randn(1, 128, 4, 4), torch.randn(1, 16, 128)
        left = original(baseline, latent, batch=batch)
        right = model.get_noise_prediction(latent, batch=batch)
        assert torch.equal(left, right)
        loss = (right - target).square().mean()
        reference = (left - target).square().mean()
        loss.backward()
        reference.backward()
        assert torch.equal(
            model.transformer.layer.weight.grad, baseline.transformer.layer.weight.grad
        )
        optimizer.step()
        optimizer.zero_grad()
        (model.get_noise_prediction(latent, batch=batch) - target).square().mean().backward()
        optimizer.step()
        assert all(g["calls"] == 2 for g in state.stats["gradients"].values())
        assert state.stats["gradients"]["categories"]["max_abs"] > 0
        lora = tmp_path / "lora.safetensors"
        lora.write_bytes(b"fixture")
        state.save(tmp_path / "encoder.safetensors", lora)
        batch.file_items[0].flip_x = True
        with pytest.raises(ValueError, match="flipped"):
            model.get_noise_prediction(latent, batch=batch)
    finally:
        state.close()
    assert Model.get_noise_prediction is original
    assert Trainer.hook_add_extra_train_params is extra
    assert not model.transformer._forward_pre_hooks
    with pytest.raises(ValueError, match="pinned"):
        install_features(Model, metadata_fixture(scene_data), "wrong", trainer_type=Trainer)


def test_feature_config_rejects_loss_or_geometry_changes(scene_data, tmp_path, monkeypatch):
    from src.training import feature_config

    seal = seal_fixture(scene_data, tmp_path)
    # The small fixture has train/dev only, unlike the required real three-split data.
    real_sha = feature_config.sha256
    monkeypatch.setattr(
        feature_config,
        "sha256",
        lambda p: "test-unused" if str(p).endswith("test/manifest.csv") else real_sha(p),
    )
    config = ProjectSettings.load().recipe
    process = config["config"]["process"][0]
    process["datasets"][0].update(folder_path=str(scene_data / "train/target"), resolution=[64])
    loss_recipe = copy.deepcopy(process["train"])
    attach_features(config, scene_data, seal, "full_role", EncoderConfig().as_dict(), 1e-3)
    assert process["train"] == loss_recipe
    assert verify_features(config)["mode"] == "full_role"
    assert process["datasets"][0]["control_path"] == [str(scene_data / "train/reference")]
    for key, value in (("flip_x", True), ("mask_path", "target")):
        bad = copy.deepcopy(config)
        bad["config"]["process"][0]["datasets"][0][key] = value
        with pytest.raises(ValueError, match="source-only"):
            verify_features(bad)
    bad = copy.deepcopy(config)
    bad["config"]["process"][0]["train"]["noise_scheduler"] = "different"
    with pytest.raises(ValueError, match="flowmatch"):
        verify_features(bad)
