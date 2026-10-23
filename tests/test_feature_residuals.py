"""Frozen repair correctness, budget parity and provenance; no claimed human/GPU results."""

import copy
from types import SimpleNamespace

import pytest


def trained_encoder():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_encoder import build_encoder
    from src.conditioning.features import EncoderConfig

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        encoder = build_encoder(EncoderConfig()).eval().requires_grad_(False)
        with torch.no_grad():
            encoder.output.weight.normal_(std=0.02)
            encoder.output.bias.fill_(0.01)
    return encoder


def test_energy_matching_is_bounded_and_zero_identity_exact():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_residuals import centered_residual

    r = torch.full((2, 16, 8), 0.01)
    identity = r.clone()
    identity[1, :4] += 0.0001
    candidate, stats = centered_residual(r, identity, 0.25)
    assert not candidate[0].any() and stats["gain"][0] == 0
    assert torch.allclose(candidate[1].square().mean(), r[1].square().mean())
    assert not candidate[1, 4:].any() and candidate.isfinite().all()
    clipped, stats = centered_residual(r * 100, identity * 100, 0.25)
    assert clipped.abs().max() <= 0.25 and stats["clipped_fraction"] > 0
    with pytest.raises(ValueError):
        centered_residual(r, identity * float("nan"), 0.25)


def test_sham_preserves_validity_strata_multiset_and_rng():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_residuals import residual_values

    tokens = torch.zeros(1, 16, 5)
    tokens[..., 4] = torch.tensor([0, 0.5, 1, 1] * 4)
    r = torch.arange(128).reshape(1, 16, 8).float() / 10000
    before = torch.random.get_rng_state()
    values, order, _ = residual_values(r, r.flip(1), tokens, 0.25, 71)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(tokens[..., 4], tokens[..., 4].gather(1, order))
    assert not torch.equal(order[0], torch.arange(16))
    for valid in tokens[..., 4].unique():
        mask = tokens[..., 4] == valid
        assert torch.equal(
            values["centered"][mask].flatten().sort().values,
            values["centered_shuffled"][mask].flatten().sort().values,
        )
    assert torch.allclose(
        values["centered"].square().sum(), values["centered_shuffled"].square().sum()
    )


def test_cache_has_same_parameters_two_encodes_and_strict_input_binding():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_interventions import intervene
    from src.conditioning.feature_residuals import cached_encoders

    encoder = trained_encoder()
    tokens = torch.rand(1, 16, 5)
    tokens[..., 4] = 1
    calls = []
    handle = encoder.register_forward_hook(lambda *args: calls.append(1))
    wrappers, _, stats = cached_encoders(encoder, tokens, (4, 4), 71)
    assert len(calls) == stats["encoder_evaluations"] == 2
    for wrapper in wrappers.values():
        assert sum(p.numel() for p in wrapper.parameters()) == stats["parameters"] == 13312
        assert {id(p) for p in wrapper.parameters()} == {id(p) for p in encoder.parameters()}
        assert wrapper(tokens, (4, 4)).shape == (1, 16, 128)
        with pytest.raises(ValueError, match="input/grid"):
            wrapper(tokens, (2, 8))
        with pytest.raises(ValueError, match="input/grid"):
            wrapper(tokens * 0.5, (4, 4))
    assert len(calls) == 2  # No hidden encoder work on any diffusion arm.
    identity = intervene(tokens, (4, 4), "identity")
    wrappers, _, _ = cached_encoders(encoder, identity, (4, 4), 71)
    assert torch.equal(wrappers["centered"](identity, (4, 4)), wrappers["zero"](identity, (4, 4)))
    handle.remove()


def test_cache_provenance_rejects_tamper_wrong_field_and_recipe(tmp_path):
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_residuals import cached_encoders
    from src.evaluation.feature_residual_provenance import save_conditions, verify_condition

    encoder = trained_encoder()
    tokens = torch.rand(1, 16, 5)
    tokens[..., 4] = 1
    field = SimpleNamespace(size=(64, 64), tokens=lambda: tokens[0].numpy())
    wrappers, arrays, stats = cached_encoders(encoder, tokens, (4, 4), 71)
    records = save_conditions(tmp_path / "cache", wrappers, arrays, stats)
    for name, record in records.items():
        verify_condition(record, encoder, field, name)
    wrong = copy.deepcopy(records["centered"])
    wrong["statistics"]["cap"] *= 2
    with pytest.raises(ValueError, match="budget"):
        verify_condition(wrong, encoder, field, "centered")
    with pytest.raises(ValueError, match="provenance"):
        verify_condition(records["centered"], encoder, field, "zero")
    wrong_field = SimpleNamespace(size=(64, 64), tokens=lambda: (tokens[0] * 0.5).numpy())
    with pytest.raises(ValueError, match="budget"):
        verify_condition(records["centered"], encoder, wrong_field, "centered")
    from pathlib import Path

    Path(records["centered"]["path"]).write_bytes(b"tampered residual")
    with pytest.raises(ValueError, match="provenance"):
        verify_condition(records["centered"], encoder, field, "centered")


def test_repair_preparation_requires_both_real_diagnostic_receipts(tmp_path, monkeypatch):
    from src.core.io import write_json
    from src.core.paths import CONFIG_ROOT
    from src.inference import feature_repair_study as module

    monkeypatch.setattr(module, "load_plan", lambda _: dict(protocol=dict(train_seeds=[211, 223])))
    write_json(tmp_path / "plan.json", {"fixture": True})
    receipt = tmp_path / "diagnostic"
    write_json(receipt / "report.json", {"fixture": True})
    monkeypatch.setattr(module, "phase_outputs", lambda *args: {"211": receipt})
    config = CONFIG_ROOT / "experiments/role_feature_repair.yaml"
    with pytest.raises(ValueError, match="both"):
        module.prepare(tmp_path, config)
    monkeypatch.setattr(module, "phase_outputs", lambda *args: {"211": receipt, "223": receipt})
    module.prepare(tmp_path, config)
    _, record = module.load_comparison(tmp_path)
    assert record["recipe"]["training_updates"] == 0
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, config)
    (receipt / "report.json").write_text("changed evidence")
    with pytest.raises(ValueError, match="provenance"):
        module.load_comparison(tmp_path)
