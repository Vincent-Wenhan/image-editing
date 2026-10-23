"""Numerical interventions are observations, never fake human or GPU results."""

import copy

import pytest

from src.core.config import load_yaml
from src.core.paths import CONFIG_ROOT
from src.inference.feature_diagnostic_study import verify_protocol


def test_protocol_forbids_training_and_expanded_or_test_budget():
    config = load_yaml(CONFIG_ROOT / "experiments/role_feature_diagnostics.yaml")
    verify_protocol(config)
    for key, value in [("training_updates", 1), ("max_parallel", 8), ("probe_steps", [0])]:
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            verify_protocol(changed)


def test_frozen_plan_rejects_protocol_asset_and_matrix_drift(tmp_path, monkeypatch):

    import yaml

    from src.core.io import sha256, write_json
    from src.inference import feature_diagnostic_study as study

    monkeypatch.setattr(study, "require_dev", lambda *args: {})
    p = load_yaml(CONFIG_ROOT / "experiments/role_feature_diagnostics.yaml")
    protocol, seal, previous = [tmp_path / n for n in ["source.yaml", "seal.json", "previous"]]
    protocol.write_text(yaml.safe_dump(p))
    seal.write_text("cohort fixture")
    previous.mkdir()
    (previous / "plan.json").write_text("input study fixture")
    checkpoints = []
    for seed in p["train_seeds"]:
        for mode in p["methods"]:
            lora, encoder, training = [
                tmp_path / f"{seed}_{mode}.{s}" for s in ["lora", "encoder", "json"]
            ]
            lora.write_bytes(b"LoRA fixture; not trained weights")
            encoder.write_bytes(b"encoder fixture")
            write_json(
                training,
                dict(
                    status="complete",
                    checkpoint_sha256=sha256(lora),
                    feature_checkpoint=dict(sha256=sha256(encoder)),
                    role_features=dict(mode=mode, cohort_sha256=sha256(seal)),
                    feature_statistics=dict(initial_seed=seed),
                ),
            )
            checkpoints.append(
                dict(
                    seed=seed,
                    method=mode,
                    lora=str(lora),
                    encoder=str(encoder),
                    training_report=str(training),
                    lora_sha256=sha256(lora),
                    encoder_sha256=sha256(encoder),
                    training_report_sha256=sha256(training),
                )
            )
    plan = dict(
        status="prepared",
        protocol=p,
        protocol_path=str(protocol),
        protocol_sha256=sha256(protocol),
        data=str(tmp_path),
        cohort=str(seal),
        cohort_sha256=sha256(seal),
        existing_study=str(previous),
        existing_plan_sha256=sha256(previous / "plan.json"),
        checkpoints=checkpoints,
    )
    write_json(tmp_path / "plan.json", plan)
    write_json(tmp_path / "protocol.json", p)
    assert len(study.load_plan(tmp_path)["checkpoints"]) == 8
    protocol.write_text("changed")
    with pytest.raises(ValueError, match="protocol"):
        study.load_plan(tmp_path)
    protocol.write_text(yaml.safe_dump(p))
    plan["checkpoints"] = checkpoints[:-1]
    write_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="matrix"):
        study.load_plan(tmp_path)
    plan["checkpoints"] = checkpoints
    write_json(tmp_path / "plan.json", plan)
    encoder.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        study.load_plan(tmp_path)


def test_interventions_keep_geometry_seed_and_original_inputs():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_interventions import VARIANTS, intervene

    torch.manual_seed(11)
    tokens = torch.rand(1, 16, 5)
    tokens[..., 4] = 1
    original, state = tokens.clone(), torch.random.get_rng_state()
    for name in VARIANTS:
        result = intervene(tokens, (4, 4), name, shift=1, seed=71)
        assert result.shape == tokens.shape and torch.isfinite(result).all()
        assert torch.equal(tokens, original) and torch.equal(state, torch.random.get_rng_state())
    identity = intervene(tokens, (4, 4), "identity")
    assert torch.equal(identity[..., 0], identity[..., 1])
    assert not identity[..., 2].any() and torch.equal(identity[..., 3], identity[..., 4])
    assert not intervene(tokens, (4, 4), "zero").any()
    shifted = intervene(tokens, (4, 4), "shift", shift=1).reshape(1, 4, 4, 5)
    assert not shifted[:, :, 0].any()
    assert torch.equal(shifted[:, :, 1:], tokens.reshape(1, 4, 4, 5)[:, :, :-1])
    shuffled = intervene(tokens, (4, 4), "shuffle", seed=71)
    assert torch.equal(shuffled, intervene(tokens, (4, 4), "shuffle", seed=71))
    assert not torch.equal(shuffled, intervene(tokens, (4, 4), "shuffle", seed=72))
    with pytest.raises(ValueError):
        intervene(tokens, (4, 4), "unknown")


def test_bf16_measurement_detects_lost_and_exact_updates():
    torch = pytest.importorskip("torch")
    from src.backends.feature_probe import fusion_measure

    ids = torch.tensor([[[0, 0, 0, 0], [10, 0, 0, 0]]])
    x = torch.ones(1, 2, 128, dtype=torch.bfloat16)
    for amount, lost in [(1e-6, 1.0), (0.03125, 0.0)]:
        r = torch.full((1, 1, 128), amount)
        fused = x.clone()
        fused[:, :1] += r.to(x.dtype)
        measured = fusion_measure(x, fused, r, ids, (1, 1))
        assert measured["lost_fraction"] == lost
        assert measured["quantization_relative_error"] == lost
        assert torch.equal(x[:, 1], fused[:, 1])
    measured = fusion_measure(x, x, torch.zeros(1, 1, 128), ids, (1, 1))
    assert measured["lost_fraction"] is None and measured["residual_rms"] == 0


def test_paired_fixed_state_replay_observes_true_fusion_without_mutation():
    torch = pytest.importorskip("torch")
    from src.backends.feature_fusion import feature_fusion
    from src.backends.feature_probe import StateCapture, replay_states
    from src.conditioning.feature_encoder import build_encoder
    from src.conditioning.features import EncoderConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = torch.nn.Linear(128, 16)
            self.output = torch.nn.Linear(16, 128)

        def forward(self, *, hidden_states, img_ids, encoder_hidden_states, timestep):
            return (
                self.output(self.x_embedder(hidden_states).tanh())
                + encoder_hidden_states.mean() * 0.01,
            )

    model, encoder = Model().eval(), build_encoder(EncoderConfig()).eval()
    with torch.no_grad():
        encoder.output.weight.normal_(std=0.1)
    ids = torch.cartesian_prod(
        torch.tensor([0, 10]), torch.arange(2), torch.arange(2), torch.tensor([0])
    )
    tokens, x = torch.rand(1, 4, 5), torch.randn(1, 8, 128)
    tokens[..., 4] = 1
    capture = StateCapture([0, 1])
    for time in (0.8, 0.2):
        for branch in (0, 1):
            kwargs = dict(
                hidden_states=x,
                img_ids=ids,
                encoder_hidden_states=torch.full((1, 2, 16), float(branch)),
                timestep=torch.tensor([time]),
            )
            capture(model, (), kwargs)
            with torch.no_grad(), feature_fusion(model, encoder, tokens, (2, 2)):
                capture.capture_output(model, (), model(**kwargs))
    capture.validate(2)
    before = x.clone()
    p = dict(shift_tokens=1, shuffle_seed=71, inference=dict(guidance=4.0))
    probes = replay_states(
        model, encoder, tokens, (2, 2), capture, ["original", "zero", "swap_old_new"], p
    )
    assert len(probes) == 2 and all(r["zero_disabled_exact"] for r in probes)
    assert probes[0]["positive"]["input_sha256"] == probes[0]["negative"]["input_sha256"]
    assert probes[0]["positive"]["text_sha256"] != probes[0]["negative"]["text_sha256"]
    assert probes[0]["cfg"]["swap_old_new"]["rms"] > 0
    assert torch.equal(x, before)
    assert not model._forward_pre_hooks and not model.x_embedder._forward_hooks
