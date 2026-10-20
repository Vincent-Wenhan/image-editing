"""Check the pinned trainer adapter without loading a model or allocating CUDA."""

from types import SimpleNamespace

import pytest

from src.training.flux2_compat import configure_flux2_trainer


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

    configure_flux2_trainer(Model, tmp_path)
    model = Model()
    model.vae = VAE()
    latent = SimpleNamespace(device="cuda:0")
    batch = SimpleNamespace(**{control: object()}) if control else None
    result = model.get_noise_prediction(latent, 1, "text", 4.0, batch, other="unchanged")
    assert result == "upstream-result"
    assert calls == [(latent, 1, "text", 4.0, batch, {"other": "unchanged"})]
    assert moves == (["cuda:0"] if control and device == "cpu" else [])
    assert Model.flux2_klein_te_path == str(tmp_path)
