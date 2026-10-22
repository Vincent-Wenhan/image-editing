"""Small real Diffusers transformer on CPU; this does not stand in for 4B GPU tests."""

import pytest


def test_real_cpu_flux2_transformer_alignment_gradients_and_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from einops import rearrange

    from src.backends.feature_checkpoint import load_encoder, save_encoder
    from src.backends.feature_fusion import feature_fusion
    from src.conditioning.feature_encoder import build_encoder
    from src.conditioning.features import EncoderConfig

    torch.manual_seed(211)
    model = Flux2Transformer2DModel(
        in_channels=128,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=(8, 8, 8, 8),
        guidance_embeds=False,
    ).requires_grad_(False)
    # Reviewed BFL c/pi/pj order agrees with Diffusers; both are row-major tokens.
    latent = torch.arange(32 * 8 * 8).reshape(1, 32, 8, 8).float()
    native = rearrange(latent, "b c (h pi) (w pj) -> b (h w) (c pi pj)", pi=2, pj=2)
    diffusers = Flux2KleinPipeline._pack_latents(Flux2KleinPipeline._patchify_latents(latent))
    assert torch.equal(native, diffusers)
    ids = torch.cartesian_prod(
        torch.tensor([0, 10]), torch.arange(4), torch.arange(4), torch.tensor([0])
    )
    kwargs = dict(
        hidden_states=torch.randn(1, 32, 128),
        img_ids=ids,
        encoder_hidden_states=torch.randn(1, 2, 32),
        timestep=torch.tensor([0.5]),
        txt_ids=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]]),
        return_dict=False,
    )
    encoder = build_encoder(EncoderConfig())
    features = torch.rand(1, 16, 5)
    features[..., 4] = 1
    with torch.no_grad():
        reference = model(**kwargs)[0]
        with feature_fusion(model, encoder, features, (4, 4)):
            assert torch.equal(reference, model(**kwargs)[0])
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    model.enable_gradient_checkpointing()
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        with feature_fusion(model, encoder, features, (4, 4)):
            model(**kwargs)[0].square().mean().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in encoder.parameters())
        optimizer.step()
    assert all(p.grad.abs().max() > 0 for p in encoder.parameters())
    model.eval()
    with torch.no_grad():
        with feature_fusion(model, encoder, features, (4, 4)):
            prediction = model(**kwargs)[0]
        with feature_fusion(model, encoder, features[..., [1, 0, 3, 2, 4]], (4, 4)):
            assert not torch.equal(prediction, model(**kwargs)[0])
        with feature_fusion(model, encoder, torch.zeros_like(features), (4, 4)):
            assert torch.equal(reference, model(**kwargs)[0])
        lora = tmp_path / "adapter.safetensors"
        lora.write_bytes(b"CPU paired metadata fixture; no 4B training claim")
        path = tmp_path / "encoder.safetensors"
        save_encoder(path, encoder, lora, "full_role")
        reloaded, _ = load_encoder(path, lora)
        with feature_fusion(model, reloaded, features, (4, 4)):
            assert torch.equal(prediction, model(**kwargs)[0])
