"""One parameter-free repair preserves legacy, constant and zero-init contracts."""

import pytest

from src.conditioning.features import EncoderConfig
from src.core.paths import CONFIG_ROOT
from src.training.learnability_study import resolve_protocol


def test_protocol_override_changes_only_registered_normalization():
    path = CONFIG_ROOT / "experiments/role_learnability.yaml"
    original = resolve_protocol(path)
    normalized = resolve_protocol(path, "channel_rms")
    assert normalized["encoder"].pop("normalization") == "channel_rms"
    assert original == normalized
    assert "normalization" not in EncoderConfig().as_dict()
    with pytest.raises(ValueError, match="normalization"):
        resolve_protocol(path, "other")
    with pytest.raises(ValueError, match="configuration"):
        EncoderConfig(normalization=[])


def fields(torch):
    x = torch.zeros(1, 16, 5)
    x[..., 4] = 1
    x[:, :3, 0], x[:, 4:8, 1], x[:, 12:14, 2], x[:, :12, 3] = 1, 1, 0.25, 1
    return x


def test_channel_scaling_preserves_geometry_empty_channels_and_rng():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_encoder import category_inputs

    raw = fields(torch).requires_grad_(True)
    untouched, rng = raw.detach().clone(), torch.random.get_rng_state()
    actual = category_inputs(raw, "channel_rms")
    assert torch.allclose(actual.square().mean(1).sqrt(), torch.ones(1, 5))
    assert torch.equal(actual == 0, raw == 0) and torch.equal(raw, untouched)
    assert torch.equal(torch.random.get_rng_state(), rng)
    actual.sum().backward()
    assert raw.grad.isfinite().all()
    assert raw.grad[0, 0, 2] > raw.grad[0, 0, 4]
    empty = torch.zeros_like(raw, requires_grad=True)
    category_inputs(empty, "channel_rms").sum().backward()
    assert torch.equal(empty.grad, torch.ones_like(empty))
    assert category_inputs(raw, "none") is raw


def test_capacity_constant_zero_init_and_actual_role_effect():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_encoder import build_encoder

    plain = build_encoder(EncoderConfig())
    fixed = build_encoder(EncoderConfig(normalization="channel_rms"))
    fixed.load_state_dict(plain.state_dict())
    assert (
        sum(p.numel() for p in fixed.parameters())
        == sum(p.numel() for p in plain.parameters())
        == 13312
    )
    x = fields(torch)
    assert torch.equal(fixed(x, (4, 4)), plain(x, (4, 4)))
    with torch.no_grad():
        plain.output.weight.normal_(std=0.02)
        fixed.load_state_dict(plain.state_dict())
    ones = torch.ones_like(x)
    assert torch.equal(fixed(ones, (4, 4)), plain(ones, (4, 4)))
    assert not torch.equal(fixed(x, (4, 4)), plain(x, (4, 4)))
    zeros = torch.zeros_like(x)
    assert fixed(zeros, (4, 4)).count_nonzero() == 0
    fixed(x, (4, 4)).square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in fixed.parameters())


def test_training_batch_and_single_inference_have_same_field_encoding():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_encoder import build_encoder, category_inputs

    encoder = build_encoder(EncoderConfig(normalization="channel_rms"))
    with torch.no_grad():
        encoder.output.weight.normal_(std=0.02)
    values = torch.rand(6, 16, 5)
    values[..., 4] = 1
    values[0, :, 2] = 0
    values[..., 3] = 1 - values[..., 2]
    individual = torch.cat([category_inputs(x[None], "channel_rms") for x in values])
    assert torch.equal(category_inputs(values, "channel_rms"), individual)
    encoder.train()
    batched = encoder(values, (4, 4))
    encoder.eval()
    singles = torch.cat([encoder(x[None], (4, 4)) for x in values])
    assert torch.allclose(batched, singles, rtol=1e-5, atol=1e-7)


def test_legacy_and_normalized_checkpoint_metadata_reload(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from src.backends.feature_checkpoint import load_encoder, save_encoder
    from src.conditioning.feature_encoder import build_encoder

    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"paired-unit-test-adapter")
    for mode in ["none", "channel_rms"]:
        encoder = build_encoder(EncoderConfig(normalization=mode))
        with torch.no_grad():
            encoder.output.weight.normal_(std=0.02)
        path = tmp_path / f"{mode}.safetensors"
        saved = save_encoder(path, encoder, lora, "full_role")
        loaded, metadata = load_encoder(path, lora)
        assert metadata == saved["metadata"] and loaded.config.normalization == mode
        assert ("normalization" in metadata["encoder"]) == (mode != "none")
        assert torch.equal(encoder(fields(torch), (4, 4)), loaded(fields(torch), (4, 4)))


def test_repair_gates_require_edit_gain_and_preservation():
    from src.core.config import load_yaml
    from src.evaluation.learnability_comparison import repair_gates

    gate = load_yaml(CONFIG_ROOT / "experiments/role_learnability.yaml")["gate"]
    old = dict(
        nonzero=dict(t_target_mae=0.1, d_target_mae=0.2, p_source_drift=0.005),
        by_strength={"0.0": dict(target_mae=0.006)},
    )
    new = dict(
        nonzero=dict(t_target_mae=0.09, d_target_mae=0.18, p_source_drift=0.005),
        by_strength={"0.0": dict(target_mae=0.006)},
    )
    assert all(repair_gates(old, new, gate).values())
    new["nonzero"]["d_target_mae"] = 0.195
    new["by_strength"]["0.0"]["target_mae"] = 0.01
    result = repair_gates(old, new, gate)
    assert result["t"] and not result["d"] and not result["identity"]
    new["nonzero"]["t_target_mae"] = None
    assert not repair_gates(old, new, gate)["t"]
