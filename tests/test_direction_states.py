"""Target-free common states, coordinate alignment, scoped hooks and frozen parameters."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def fixture_model():
    torch = pytest.importorskip("torch")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            return ((hidden_states * self.weight + encoder_hidden_states.mean()).bfloat16(),)

        def cache_context(self, _name):
            return nullcontext()

    model = Toy().eval().requires_grad_(False)
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    noise = torch.stack([torch.zeros(16), yy.flatten(), xx.flatten(), torch.zeros(16)], -1)
    source = noise.clone()
    source[:, 0] = 10
    ids = torch.cat([noise, source], 0)
    hidden = torch.randn((1, 32, 128), generator=torch.Generator().manual_seed(42)).bfloat16()
    states = {
        b: dict(
            hidden_states=hidden.clone(),
            img_ids=ids.clone(),
            timestep=torch.ones(1).bfloat16(),
            encoder_hidden_states=torch.ones((1, 2, 3)).bfloat16() * b,
            return_dict=False,
        )
        for b in [0, 1]
    }
    return torch, model, states


def test_common_states_are_source_only_and_shared_by_cfg_branches():
    from src.backends.direction_probe import build_state, geometry

    torch, _, states = fixture_model()
    positions = geometry(states[0]["img_ids"], (4, 4))
    anchor = torch.ones((1, 16, 128)).bfloat16()
    original = states[0]["hidden_states"].clone()
    for family in ["source_bridge", "noise_only"]:
        for t in [1.0, 0.9, 0.5, 0.1]:
            output, x, actual = build_state(states, positions, anchor, t, family)
            expected = (
                (1 - actual) * anchor.float() + actual * original[:, :16].float()
                if family == "source_bridge"
                else original[:, :16]
            )
            assert torch.equal(x, expected.bfloat16())
            assert torch.equal(output[0]["hidden_states"], output[1]["hidden_states"])
            assert torch.equal(output[0]["hidden_states"][:, 16:], original[:, 16:])
    assert torch.equal(states[0]["hidden_states"], original)
    states[0]["hidden_states"][0, 0, 0] = -0.0
    from src.core.metadata import tensor_sha

    _, x, _ = build_state(states, positions, anchor, 1.0, "source_bridge")
    assert tensor_sha(x) == tensor_sha(states[0]["hidden_states"][:, positions])


def test_spatial_ids_reorder_features_without_touching_references_or_weights():
    from src.backends.direction_probe import geometry, predict
    from src.conditioning.feature_encoder import build_encoder
    from src.conditioning.features import EncoderConfig

    torch, model, states = fixture_model()
    encoder = build_encoder(EncoderConfig()).eval().requires_grad_(False)
    with torch.no_grad():
        encoder.output.weight.fill_(0.1)
    tokens = torch.rand((1, 16, 5), generator=torch.Generator().manual_seed(12))
    tokens[..., 4] = 1
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        base, _, fusion = predict(
            model, encoder, tokens, (4, 4), states, geometry(states[0]["img_ids"], (4, 4))
        )
        permutation = torch.cat(
            [torch.randperm(16, generator=torch.Generator().manual_seed(4)), torch.arange(16, 32)]
        )
        moved = {
            b: dict(
                s,
                hidden_states=s["hidden_states"][:, permutation],
                img_ids=s["img_ids"][permutation],
            )
            for b, s in states.items()
        }
        other, _, moved_fusion = predict(
            model, encoder, tokens, (4, 4), moved, geometry(moved[0]["img_ids"], (4, 4))
        )
    assert all(torch.equal(base[k], other[k]) for k in base)
    assert all(torch.equal(fusion[k], moved_fusion[k]) for k in fusion)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert not model._forward_hooks and not model._forward_pre_hooks and not encoder._forward_hooks
    bad = states[0]["img_ids"].clone()
    bad[1] = bad[0]
    with pytest.raises(ValueError, match="unique"):
        geometry(bad, (4, 4))


def test_real_pipeline_capture_observes_native_cfg_and_restores_scheduler_on_failure():
    from PIL import Image

    from src.backends.direction_probe import capture_source, native_cfg
    from src.core.types import EditInput, GenerationOptions

    torch, model, states = fixture_model()

    class Scheduler:
        def step(self, value, *args, **kwargs):
            return (value,)

    scheduler = Scheduler()

    def generate(edit, images, opts, matched):
        assert images == [edit.source] and matched and opts.steps == 1
        p, n = [model(**states[b])[0][:, :16] for b in [0, 1]]
        scheduler.step(native_cfg(p, n, opts.guidance), 1, None)

    backend = SimpleNamespace(
        base=SimpleNamespace(
            pipe=SimpleNamespace(transformer=model, scheduler=scheduler), _generate=generate
        )
    )
    edit = EditInput(Image.new("RGB", (64, 64)), "fixed")
    _, _, checks = capture_source(backend, edit, GenerationOptions(64, 64, 1, 4.0, 42), (4, 4))
    assert checks["native_cfg_pipeline_exact"] and "step" not in scheduler.__dict__

    def fail(*args, **kwargs):
        raise RuntimeError("original prediction failure")

    backend.base._generate = fail
    with pytest.raises(RuntimeError, match="original prediction failure"):
        capture_source(backend, edit, GenerationOptions(64, 64, 1, 4.0, 42), (4, 4))
    assert (
        not model._forward_hooks
        and not model._forward_pre_hooks
        and "step" not in scheduler.__dict__
    )
