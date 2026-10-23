"""Supervised velocity probes only. Targets never become image references or role inputs."""

import numpy as np

from src.backends.feature_fusion import feature_fusion
from src.backends.flux2_encoding import reference_tensor
from src.core.metadata import tensor_sha
from src.datasets.scene_manifest import load_role_supervision


def score_teacher(pipe, encoder, tokens, grid, captured, root, row, times, seed):
    import torch

    supervision = load_role_supervision(root, row)
    target = reference_tensor(supervision.target, device=tokens.device, dtype=pipe.vae.dtype)
    ids = captured["img_ids"]
    noise_positions = (ids[0] if ids.ndim == 3 else ids)[:, 0] == 0
    context_sha = tensor_sha(captured["encoder_hidden_states"])
    source_sha = tensor_sha(captured["hidden_states"][:, ~noise_positions])
    with torch.inference_mode():
        latent = pipe._encode_vae_image(
            target, generator=torch.Generator(device=tokens.device).manual_seed(seed)
        )
        z0 = pipe._pack_latents(latent)
        generator = torch.Generator(device=tokens.device).manual_seed(seed)
        noise = pipe._pack_latents(
            torch.randn(latent.shape, generator=generator, device=tokens.device, dtype=latent.dtype)
        )
        if z0.shape != (1, grid[0] * grid[1], 128):
            raise ValueError("Teacher target geometry differs from role feature grid")
        labels = np.asarray(supervision.role_labels)
        h, w = labels.shape
        masks = torch.from_numpy(
            np.stack(
                [
                    (labels == k).reshape(h // 16, 16, w // 16, 16).mean((1, 3)).reshape(-1)
                    for k in (0, 1, 2)
                ]
            )
        ).to(device=tokens.device, dtype=torch.float32)
        desired = noise.float() - z0.float()
        rows = []
        for time in times:
            kwargs = dict(captured)
            hidden = captured["hidden_states"].clone()
            timestamp = torch.full_like(captured["timestep"], time)
            actual_time = float(timestamp.flatten()[0])
            hidden[:, noise_positions] = (
                (1 - actual_time) * z0.float() + actual_time * noise.float()
            ).to(hidden.dtype)
            kwargs.update(hidden_states=hidden, timestep=timestamp)
            with (
                feature_fusion(pipe.transformer, encoder, tokens, grid),
                pipe.transformer.cache_context("cond"),
            ):
                prediction = pipe.transformer(**kwargs)[0][:, noise_positions].float()
            if (
                tensor_sha(hidden[:, ~noise_positions]) != source_sha
                or tensor_sha(kwargs["encoder_hidden_states"]) != context_sha
            ):
                raise RuntimeError("Teacher probe mutated source/text conditions")
            squared = (prediction - desired).square().mean(-1)[0]
            if not squared.isfinite().all():
                raise FloatingPointError("Non-finite teacher velocity error")
            regions = {
                name: float((squared * mask).sum() / mask.sum()) if mask.sum() > 0 else None
                for name, mask in zip(["t", "d", "p"], masks, strict=True)
            }
            rows.append(
                dict(
                    time=time,
                    actual_time=float(kwargs["timestep"][0]),
                    velocity_mse=float(squared.mean()),
                    regions=regions,
                    target_latent_sha256=tensor_sha(z0),
                    noise_sha256=tensor_sha(noise),
                    source_tokens_sha256=source_sha,
                    text_sha256=context_sha,
                )
            )
    return dict(
        purpose="supervised_velocity_probe",
        target_sha256=row["target_sha256"],
        objective="unweighted diagnostic MSE, not configured upstream weighted loss",
        target_is_reference=False,
        records=rows,
    )
