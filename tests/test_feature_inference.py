"""Hash-bound extended records and explicit upper-bound inference permission."""

import copy
import json
from types import SimpleNamespace

import pytest

from src.conditioning.features import EncoderConfig
from src.core.config import load_assets
from src.core.io import sha256
from src.core.metadata import config_digest
from src.core.types import EditOutput
from src.evaluation.provenance import verify_run
from tests.test_feature_training import seal_fixture


def test_feature_runner_record_verification_and_tamper_rejection(scene_data, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from src.backends.feature_checkpoint import load_encoder, save_encoder
    from src.conditioning.feature_encoder import build_encoder
    from src.inference import feature_runner

    seal = seal_fixture(scene_data, tmp_path)
    lora, encoder_path = tmp_path / "lora.safetensors", tmp_path / "features.safetensors"
    lora.write_bytes(b"CPU paired adapter fixture")
    save_encoder(encoder_path, build_encoder(EncoderConfig()), lora, "full_role")
    model = tmp_path / "model"
    model.mkdir()
    (model / "download.json").write_text(json.dumps(load_assets()["models"]["base"]))
    monkeypatch.setattr(feature_runner, "require_lease", lambda: None)

    class Backend:
        def __init__(self, model, lora, path):
            _, self.metadata = load_encoder(path, lora)
            self.last_fusion = dict(calls=2, tokens=16, source_tokens=[16], reference_times=[10])

        def edit_features(self, edit, features, options, *, allow_oracle):
            assert allow_oracle and not hasattr(edit, "target")
            assert features.size == edit.source.size
            return EditOutput(edit.source, 0, 0, 0)

    args = SimpleNamespace(
        model=model,
        lora=lora,
        encoder=encoder_path,
        data_root=scene_data,
        manifest=scene_data / "dev/manifest.csv",
        cohort=seal,
        output=tmp_path / "run",
        feature_mode="full_role",
        allow_oracle=True,
        limit=2,
        steps=2,
        guidance=4.0,
        seed=42,
        side=64,
    )
    feature_runner.run(args, backend_factory=Backend)
    record = json.loads((args.output / "run.json").read_text())
    assert record["status"] == "complete"
    assert record["role_features"]["sha256"] == sha256(encoder_path)
    assert len(verify_run(scene_data, args.manifest, record)) == 2
    for key in ("role_features",):
        bad = copy.deepcopy(record)
        bad.pop(key)
        with pytest.raises(ValueError, match="omitted"):
            verify_run(scene_data, args.manifest, bad)
    bad = copy.deepcopy(record)
    bad["results"][0]["role_features"]["feature_sha256"] = "tampered"
    with pytest.raises(ValueError, match="Actual oracle"):
        verify_run(scene_data, args.manifest, bad)
    bad = copy.deepcopy(record)
    bad["config"]["feature_mode"] = "constant"
    bad["config_sha256"] = config_digest(bad["config"])
    with pytest.raises(ValueError, match="provenance"):
        verify_run(scene_data, args.manifest, bad)
    args.manifest = scene_data / "train/manifest.csv"
    args.output = tmp_path / "denied_test"
    with pytest.raises(ValueError, match="sealed Dev"):
        feature_runner.run(args, backend_factory=Backend)
    args.manifest, args.allow_oracle = scene_data / "dev/manifest.csv", False
    with pytest.raises(ValueError, match="allow-oracle"):
        feature_runner.run(args, backend_factory=Backend)


def test_bounded_feature_protocol_rejects_unfair_or_test_plans():
    from src.core.config import load_yaml
    from src.core.paths import CONFIG_ROOT
    from src.training.feature_study import verify_protocol

    protocol = load_yaml(CONFIG_ROOT / "experiments/role_features.yaml")
    verify_protocol(protocol)
    for key, value in (("methods", ["full_role"]), ("train_seeds", [211, 211]), ("steps", 2000)):
        bad = copy.deepcopy(protocol)
        bad[key] = value
        with pytest.raises(ValueError, match="bounded"):
            verify_protocol(bad)
    protocol["inference"]["splits"] = ["test"]
    with pytest.raises(ValueError, match="Dev"):
        verify_protocol(protocol)
