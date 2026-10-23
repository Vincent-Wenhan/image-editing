"""Non-RGB field contracts and actual spatial/category alignment."""

import copy

import numpy as np
import pytest

from src.conditioning.features import FEATURE_MODES, EncoderConfig, RoleFeatures
from src.core.io import read_rows
from src.datasets.features import load_features


def test_pixel_to_token_origin_and_area_pooling_on_rectangular_grid():
    values = np.zeros((5, 64, 96), np.float32)
    values[4] = 1
    values[0, 16:32, 64:80] = 1
    values[1, 0:8, 0:16] = 1
    field = RoleFeatures(values, metadata={"purpose": "oracle_upper_bound"})
    tokens = field.tokens().reshape(4, 6, 5)
    assert tokens[1, 4, 0] == 1 and tokens[..., 0].sum() == 1
    assert tokens[0, 0, 1] == 0.5 and tokens[..., 1].sum() == 0.5
    assert field.size == (96, 64)


def test_fields_do_not_open_target_rgb_and_constant_does_not_reveal_valid(scene_data, monkeypatch):
    from numpy.lib.npyio import NpzFile

    row = read_rows(scene_data / "dev/manifest.csv")[1]
    original, seen = NpzFile.__getitem__, []

    def guarded(self, key):
        assert key in {"source_occupancy", "target_occupancy"}
        seen.append(key)
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", guarded)
    (scene_data / row["target_path"]).unlink()
    fields = {m: load_features(scene_data, row, m) for m in FEATURE_MODES}
    assert seen == ["source_occupancy", "target_occupancy"] * 4
    assert np.all(fields["constant"].values == 1)
    assert len({f.digest for f in fields.values()}) == 4
    assert all(f.tokens().shape == (16, 5) for f in fields.values())
    full = fields["full_role"]
    assert not np.array_equal(full.values[0], full.values[1])
    assert all(f.metadata["image_order"] == "source" for f in fields.values())
    identity = load_features(scene_data, read_rows(scene_data / "dev/manifest.csv")[0], "full_role")
    assert identity.values[0].any() and np.array_equal(identity.values[0], identity.values[1])
    assert not identity.values[2].any()


def test_feature_contracts_and_hash_drift(scene_data):
    row = read_rows(scene_data / "dev/manifest.csv")[1]
    f = load_features(scene_data, row, "full_role")
    assert not f.values.flags.writeable
    assert f.size == (64, 64)
    values = f.values.copy()
    values[4, 0, 0], values[0, 0, 0] = 0, 1
    with pytest.raises(ValueError, match="finite aligned"):
        RoleFeatures(values, metadata=f.metadata)
    with pytest.raises(ValueError, match="explicit oracle"):
        RoleFeatures(np.zeros_like(values))
    with pytest.raises(TypeError, match="never target"):
        RoleFeatures(f.values, metadata=dict(f.metadata, target=np.zeros((2, 2))))
    (scene_data / row["evidence_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_features(scene_data, row, "binary")


def test_encoder_zero_compatibility_and_two_step_finite_learning():
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_encoder import build_encoder

    torch.manual_seed(7)
    encoder = build_encoder(EncoderConfig())
    tokens = torch.rand(2, 16, 5)
    tokens[..., 4] = 1
    assert torch.equal(encoder(tokens, (4, 4)), torch.zeros(2, 16, 128))
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    gradients = []
    for _ in range(2):
        optimizer.zero_grad()
        loss = (encoder(tokens, (4, 4)) - tokens[..., :1]).square().mean()
        loss.backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in encoder.parameters())
        gradients.append(float(encoder.categories.grad.abs().max()))
        optimizer.step()
    assert gradients[0] == 0 and gradients[1] > 0
    assert not encoder(torch.zeros_like(tokens), (4, 4)).any()
    assert not torch.equal(encoder(tokens, (4, 4)), encoder(tokens[..., [1, 0, 3, 2, 4]], (4, 4)))
    assert not torch.equal(encoder(tokens, (4, 4)).roll(1, 1), encoder(tokens.roll(1, 1), (4, 4)))
    assert sum(p.numel() for p in encoder.parameters()) == 13312
    assert encoder(tokens, (4, 4)).abs().max() <= 0.25


def test_fusion_reorders_using_ids_preserves_source_and_removes_hook():
    torch = pytest.importorskip("torch")
    from src.backends.feature_fusion import aligned_residual, feature_fusion
    from src.conditioning.feature_encoder import build_encoder

    ids = torch.tensor([[[0, 1, 1, 0], [10, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])
    residual = torch.arange(4).reshape(1, 4, 1).expand(-1, -1, 128).float()
    selected, values = aligned_residual(residual, ids, (2, 2))
    assert selected[0].tolist() == [0, 2, 3, 4]
    assert values[0][:, 0].tolist() == [3, 0, 2, 1]
    broken = ids.clone()
    broken[0, 0] = broken[0, 2]
    with pytest.raises(ValueError, match="unique"):
        aligned_residual(residual, broken, (2, 2))

    class Transformer(torch.nn.Module):
        def forward(self, *, hidden_states, img_ids):
            return hidden_states

    model, encoder = Transformer(), build_encoder(EncoderConfig())
    tokens, x = torch.ones(1, 4, 5), torch.randn(1, 5, 128)
    with feature_fusion(model, encoder, tokens, (2, 2)):
        assert torch.equal(model(hidden_states=x, img_ids=ids), x)
    with torch.no_grad():
        encoder.output.bias.fill_(0.3)
    with feature_fusion(model, encoder, tokens, (2, 2)) as stats:
        output = model(hidden_states=x, img_ids=ids)
        assert torch.equal(output[:, 1], x[:, 1])
        assert not torch.equal(output[:, 0], x[:, 0])
    assert stats["calls"] == 1 and not model._forward_pre_hooks
    with pytest.raises(RuntimeError), feature_fusion(model, encoder, tokens, (2, 2)):
        raise RuntimeError("test cleanup")
    assert not model._forward_pre_hooks
    with feature_fusion(model, None, None, None):
        assert torch.equal(model(hidden_states=x, img_ids=ids), x)


def test_paired_checkpoint_reloads_and_rejects_wrong_lora(tmp_path):
    torch = pytest.importorskip("torch")
    from src.backends.feature_checkpoint import load_encoder, save_encoder
    from src.conditioning.feature_encoder import build_encoder

    lora, path = tmp_path / "adapter.safetensors", tmp_path / "encoder.safetensors"
    lora.write_bytes(b"CPU checkpoint binding fixture")
    encoder = build_encoder(EncoderConfig())
    with torch.no_grad():
        encoder.output.weight.normal_(std=0.02)
    saved = save_encoder(path, encoder, lora, "full_role")
    loaded, meta = load_encoder(path, lora)
    tokens = torch.rand(1, 16, 5)
    assert torch.equal(encoder(tokens, (4, 4)), loaded(tokens, (4, 4)))
    assert saved["metadata"] == meta
    lora.write_bytes(b"wrong adapter")
    with pytest.raises(ValueError, match="LoRA SHA256"):
        load_encoder(path, lora)


def test_encoder_capacity_is_equal_for_all_information_ablations(scene_data):
    torch = pytest.importorskip("torch")
    from src.conditioning.feature_encoder import build_encoder

    row = read_rows(scene_data / "dev/manifest.csv")[1]
    state = build_encoder(EncoderConfig()).state_dict()
    for mode in FEATURE_MODES:
        encoder = build_encoder(EncoderConfig())
        encoder.load_state_dict(copy.deepcopy(state))
        features = load_features(scene_data, row, mode)
        out = encoder(torch.from_numpy(features.tokens())[None], (4, 4))
        assert out.shape == (1, 16, 128)
        assert sum(p.numel() for p in encoder.parameters()) == 13312
