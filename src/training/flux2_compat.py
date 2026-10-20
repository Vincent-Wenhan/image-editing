"""Device-only adaptation for ai-toolkit a055947's raw-reference training path.

The trainer offloads the VAE after caching target latents. FLUX subsequently
encodes raw references and concatenates them with CUDA target tokens. Keep that
encoder on the latent device; delegate all computation and dtype choices to the
pinned upstream method. No external source files are changed.
"""

from functools import wraps
from pathlib import Path


def configure_flux2_trainer(model_type: type, text_encoder: Path) -> None:
    model_type.flux2_klein_te_path = str(text_encoder)
    original = model_type.get_noise_prediction

    @wraps(original)
    def predict(
        self,
        latent_model_input,
        timestep,
        text_embeddings,
        guidance_embedding_scale,
        batch=None,
        **kwargs,
    ):
        has_reference = batch is not None and (
            getattr(batch, "control_tensor_list", None) is not None
            or getattr(batch, "control_tensor", None) is not None
        )
        if has_reference and self.vae.device != latent_model_input.device:
            self.vae.to(device=latent_model_input.device)
        return original(
            self,
            latent_model_input,
            timestep,
            text_embeddings,
            guidance_embedding_scale,
            batch=batch,
            **kwargs,
        )

    model_type.get_noise_prediction = predict
