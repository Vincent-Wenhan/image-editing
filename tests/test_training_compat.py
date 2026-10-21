"""Check the pinned trainer adapter without loading a model or allocating CUDA."""

from types import SimpleNamespace

import pytest

from src.training.flux2_compat import SUPPORTED_TRAINER_REVISION, configure_flux2_trainer


@pytest.mark.parametrize("control", ["control_tensor", "control_tensor_list", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_reference_vae_placement_preserves_upstream_call(tmp_path, control, device):
    moves, calls = [], []

    class VAE:
        def __init__(self):
            self.device = device

        def to(self, *, device):
            self.device = device
            moves.append(device)

    class Model:
        def get_noise_prediction(self, latent, timestep, text, guidance, batch=None, **kwargs):
            calls.append((latent, timestep, text, guidance, batch, kwargs))
            return "upstream-result"

    configure_flux2_trainer(Model, tmp_path, revision=SUPPORTED_TRAINER_REVISION)
    wrapper = Model.get_noise_prediction
    configure_flux2_trainer(Model, tmp_path / "again", revision=SUPPORTED_TRAINER_REVISION)
    assert Model.get_noise_prediction is wrapper
    model = Model()
    model.vae = VAE()
    latent = SimpleNamespace(device="cuda:0")
    batch = SimpleNamespace(**{control: object()}) if control else None
    result = model.get_noise_prediction(latent, 1, "text", 4.0, batch, other="unchanged")
    assert result == "upstream-result"
    assert calls == [(latent, 1, "text", 4.0, batch, {"other": "unchanged"})]
    assert moves == (["cuda:0"] if control and device == "cpu" else [])
    assert Model.flux2_klein_te_path == str(tmp_path / "again")


def test_unreviewed_revision_is_rejected_without_modifying_class(tmp_path):
    class Model:
        def get_noise_prediction(self):
            pass

    original = Model.get_noise_prediction
    with pytest.raises(ValueError, match="reviewed"):
        configure_flux2_trainer(Model, tmp_path, revision="different")
    assert Model.get_noise_prediction is original
    assert not hasattr(Model, "flux2_klein_te_path")


def test_reference_encoder_follows_device_changes_without_changing_dtype(tmp_path):
    class VAE:
        device = "cpu"
        dtype = "bfloat16"

        def to(self, *, device):
            self.device = device

    class Model:
        vae = VAE()

        def get_noise_prediction(self, *args, **kwargs):
            return self.vae.device, self.vae.dtype

    configure_flux2_trainer(Model, tmp_path, revision=SUPPORTED_TRAINER_REVISION)
    model = Model()
    for device in ("cuda:0", "cpu", "cuda:1"):
        assert model.get_noise_prediction(
            SimpleNamespace(device=device),
            1,
            None,
            4,
            batch=SimpleNamespace(control_tensor_list=[[object()]]),
        ) == (device, "bfloat16")


def test_patch_preserves_cpu_loss_gradients_and_optimizer_updates(tmp_path):
    torch = pytest.importorskip("torch")
    import copy

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 3)
            self.vae = torch.nn.Linear(3, 3)
            self.vae.requires_grad_(False)
            self.vae.device = torch.device("cpu")

        def get_noise_prediction(self, latent, timestep, text, guidance, batch=None):
            return self.layer(latent) + self.vae(batch.control_tensor)

    torch.manual_seed(42)
    reference = Model()
    adapted = copy.deepcopy(reference)
    original = Model.get_noise_prediction
    optimizers = [
        torch.optim.AdamW(model.layer.parameters(), lr=1e-4) for model in (reference, adapted)
    ]
    configure_flux2_trainer(Model, tmp_path, revision=SUPPORTED_TRAINER_REVISION)
    configure_flux2_trainer(Model, tmp_path, revision=SUPPORTED_TRAINER_REVISION)
    for _ in range(2):
        latent, control, target = [torch.randn(2, 3) for _ in range(3)]
        batch = SimpleNamespace(control_tensor=control)
        before = original(reference, latent, 1, None, 4, batch)
        after = adapted.get_noise_prediction(latent, 1, None, 4, batch)
        losses = [(value - target).square().mean() for value in (before, after)]
        assert torch.equal(losses[0], losses[1])
        for optimizer, loss in zip(optimizers, losses, strict=True):
            optimizer.zero_grad()
            loss.backward()
        assert torch.equal(reference.layer.weight.grad, adapted.layer.weight.grad)
        for optimizer in optimizers:
            optimizer.step()
        for key, value in reference.state_dict().items():
            assert torch.equal(value, adapted.state_dict()[key])
