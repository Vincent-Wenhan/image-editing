"""Teacher targets stay in supervised evaluation; interventions preserve valid geometry."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def test_teacher_never_passes_target_as_reference_or_calls_generation(scene_data):
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_encoder import build_encoder
    from src.conditioning.features import EncoderConfig
    from src.core.io import read_rows
    from src.datasets.features import load_features
    from src.evaluation.feature_teacher import score_teacher

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.states = []

        def cache_context(self, name):
            assert name == "cond"
            return nullcontext()

        def forward(self, **kwargs):
            self.states.append(kwargs["hidden_states"].detach().clone())
            return (kwargs["hidden_states"],)

    class Pipe:
        def __init__(self):
            self.vae = SimpleNamespace(dtype=torch.float32)
            self.transformer = Model()
            self.targets = []

        def _encode_vae_image(self, value, generator):
            self.targets.append(value.detach().clone())
            return torch.zeros(1, 128, 4, 4)

        def _pack_latents(self, latent):
            return latent.flatten(2).transpose(1, 2)

        def __call__(self, *args, **kwargs):
            raise AssertionError("Supervised teacher must not generate images")

    row = read_rows(scene_data / "train/manifest.csv")[1]
    field = load_features(scene_data, row, "full_role")
    tokens = torch.from_numpy(field.tokens())[None]
    ids = torch.cartesian_prod(
        torch.tensor([0, 10]), torch.arange(4), torch.arange(4), torch.tensor([0])
    )
    state = dict(
        hidden_states=torch.full((1, 32, 128), 0.3),
        img_ids=ids,
        encoder_hidden_states=torch.ones(1, 2, 16),
        timestep=torch.ones(1),
    )
    untouched = state["hidden_states"].clone()
    pipe = Pipe()
    score = score_teacher(
        pipe,
        build_encoder(EncoderConfig()),
        tokens,
        (4, 4),
        state,
        scene_data,
        row,
        [0.1, 0.5, 0.9],
        42,
    )
    assert score["target_is_reference"] is False and len(pipe.targets) == 1
    assert len(score["records"]) == len(pipe.transformer.states) == 3
    assert score["records"][0]["velocity_mse"] > score["records"][-1]["velocity_mse"]
    assert len({r["noise_sha256"] for r in score["records"]}) == 1
    assert len({r["source_tokens_sha256"] for r in score["records"]}) == 1
    assert torch.equal(untouched, state["hidden_states"])
    assert all(torch.equal(s[:, 16:], untouched[:, 16:]) for s in pipe.transformer.states)
    assert not pipe.transformer._forward_pre_hooks
    from src.evaluation.learnability_provenance import verify_teacher

    item = dict(
        variant="original",
        teacher=score,
        source_tokens_sha256=score["records"][0]["source_tokens_sha256"],
        text_sha256=score["records"][0]["text_sha256"],
    )
    protocol = dict(teacher_times=[0.1, 0.5, 0.9])
    verify_teacher(item, row, protocol)
    score["target_is_reference"] = True
    with pytest.raises(ValueError, match="Supervised"):
        verify_teacher(item, row, protocol)
    item["teacher"] = None
    with pytest.raises(ValueError, match="missing"):
        verify_teacher(item, row, protocol)


def test_layer_pairs_use_image_stream_and_leave_forward_unchanged():
    torch = pytest.importorskip("torch")
    from src.backends.layer_response import LayerResponses

    class Double(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(16, 16)

        def forward(self, image, context):
            return context, self.layer(image)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = torch.nn.Linear(128, 16)
            self.transformer_blocks = torch.nn.ModuleList([Double(), Double()])
            self.single_transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(16, 16) for _ in range(3)]
            )

        def forward(self, image):
            image = self.x_embedder(image)
            text = image.new_ones(1, 2, 16)
            for block in self.transformer_blocks:
                text, image = block(image, text)
            image = torch.cat([text, image], 1)
            for block in self.single_transformer_blocks:
                image = block(image)
            return image[:, 2:]

    model, image = Model(), torch.randn(1, 8, 128)
    noise = torch.tensor([True] * 4 + [False] * 4)
    measured = LayerResponses(model, noise, 2)
    plain = model(image)
    with measured.observe("disabled"):
        baseline = model(image)
    assert torch.equal(plain, baseline)
    with measured.observe("original"):
        model(image + 0.01)
    with measured.observe("shuffle"):
        model(image + 0.02)
    assert len(measured.records["shuffle"]) == 6
    assert all(v["delta_to_original"] > 0 for v in measured.records["shuffle"].values())
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(RuntimeError):
        with measured.observe("bad"):
            raise RuntimeError("fixture")
    assert all(not module._forward_hooks for module in model.modules())


def test_research_shuffle_keeps_validity_rng_and_joint_field_values():
    torch = pytest.importorskip("torch")
    from src.conditioning.learnability import research_tokens

    tokens = torch.rand(1, 16, 5)
    tokens[..., 4] = torch.tensor([0, 0.5, 1, 1] * 4)
    tokens[..., :4] *= tokens[..., 4:5]
    rng = torch.random.get_rng_state()
    shuffled = research_tokens(tokens, (4, 4), "shuffle", 42)
    assert torch.equal(shuffled[..., 4], tokens[..., 4])
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert not torch.equal(shuffled, tokens)
    for value in tokens[..., 4].unique():
        mask = tokens[..., 4] == value
        assert torch.equal(
            shuffled[mask].flatten().sort().values, tokens[mask].flatten().sort().values
        )


def test_region_gradient_contributions_sum_to_global_without_reweighting():
    torch = pytest.importorskip("torch")
    from src.evaluation.feature_gradient import decompose_field_gradient

    fields = torch.arange(40).reshape(1, 8, 5).float().requires_grad_(True)
    errors = fields.square().mean(-1)
    masks = torch.eye(4).repeat_interleave(2, dim=1)
    result = decompose_field_gradient(errors, masks, fields)
    assert result["sum_gradient_relative_error"] < 1e-7
    assert sum(r["loss_contribution"] for r in result["regions"].values()) == pytest.approx(
        float(errors.mean())
    )
    assert result["regions"]["t"]["area_fraction"] == 0.25
    assert all(len(r["channel_rms"]) == 5 for r in result["regions"].values())
    with pytest.raises(ValueError, match="partition"):
        decompose_field_gradient(errors, masks * 0.5, fields)
