"""T/D/P contributions to supervised velocity gradients, without changing the loss."""

import numpy as np

from src.backends.feature_fusion import feature_fusion
from src.core.metadata import tensor_sha
from src.datasets.scene_manifest import load_role_supervision
from src.evaluation.feature_teacher import TARGET_PREPROCESSING, target_tensor


def decompose_field_gradient(errors, masks, fields):
    """A common denominator preserves each region's share of the global mean."""
    import torch

    if masks.shape != (4, errors.shape[-1]) or not torch.allclose(
        masks.sum(0), torch.ones_like(errors[0])
    ):
        raise ValueError("T/D/P/ignore masks must partition the actual velocity grid")
    global_loss = errors.mean()
    total = torch.autograd.grad(global_loss, fields, retain_graph=True)[0]
    components, records = [], {}
    for name, mask in zip(["t", "d", "p", "ignore"], masks, strict=True):
        loss = (errors * mask[None]).mean()
        gradient = torch.autograd.grad(loss, fields, retain_graph=True)[0]
        if not gradient.isfinite().all():
            raise FloatingPointError("Non-finite role gradient decomposition")
        components.append(gradient)
        norm, reference = gradient.float().norm(), total.float().norm()
        records[name] = dict(
            area_fraction=float(mask.mean()),
            loss_contribution=float(loss.detach()),
            field_gradient_rms=float(gradient.float().square().mean().sqrt()),
            channel_rms=gradient.float().square().mean((0, 1)).sqrt().detach().cpu().tolist(),
            relative_l2=float(norm / reference) if reference else None,
            cosine_global=float((gradient.float() * total.float()).sum() / (norm * reference))
            if norm and reference
            else None,
        )
    difference = torch.stack(components).sum(0).float() - total.float()
    return dict(
        velocity_mse=float(global_loss.detach()),
        regions=records,
        global_channel_rms=total.float().square().mean((0, 1)).sqrt().detach().cpu().tolist(),
        sum_gradient_relative_error=float(difference.norm() / total.float().norm())
        if total.any()
        else 0.0,
        scope="unweighted supervised velocity MSE; BF16 autograd is a surrogate for rounding",
    )


def gradient_probe(pipe, encoder, tokens, grid, captured, root, row, times, seed):
    import torch

    supervision = load_role_supervision(root, row)
    inputs = {
        k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in captured.items()
    }
    ids = inputs["img_ids"]
    noise_positions = (ids[0] if ids.ndim == 3 else ids)[:, 0] == 0
    with torch.no_grad():
        target = target_tensor(supervision.target, device=tokens.device, dtype=pipe.vae.dtype)
        latent = pipe._encode_vae_image(
            target, generator=torch.Generator(device=tokens.device).manual_seed(seed)
        )
        z0 = pipe._pack_latents(latent).detach()
        noise = pipe._pack_latents(
            torch.randn(
                latent.shape,
                device=tokens.device,
                dtype=latent.dtype,
                generator=torch.Generator(device=tokens.device).manual_seed(seed),
            )
        )
    labels = np.asarray(supervision.role_labels)
    h, w = labels.shape
    areas = np.stack(
        [
            (labels == k).reshape(h // 16, 16, w // 16, 16).mean((1, 3)).reshape(-1)
            for k in [0, 1, 2, 255]
        ]
    )
    masks = torch.from_numpy(areas).to(device=tokens.device, dtype=torch.float32)
    model = pipe.transformer
    previous = model.gradient_checkpointing
    model.enable_gradient_checkpointing()
    result = []
    try:
        for time in times:
            fields = tokens.detach().clone().requires_grad_(True)
            timestamp = torch.full_like(inputs["timestep"], time)
            actual = float(timestamp.flatten()[0])
            hidden = inputs["hidden_states"].detach().clone()
            hidden[:, noise_positions] = ((1 - actual) * z0.float() + actual * noise.float()).to(
                hidden.dtype
            )
            kwargs = dict(inputs, hidden_states=hidden, timestep=timestamp)
            with (
                torch.enable_grad(),
                feature_fusion(model, encoder, fields, grid),
                model.cache_context("cond"),
            ):
                velocity = model(**kwargs)[0][:, noise_positions].float()
                errors = (velocity - (noise.float() - z0.float())).square().mean(-1)
                measured = decompose_field_gradient(errors, masks, fields)
            result.append(dict(time=time, actual_time=actual, **measured))
            del fields, velocity, errors
    finally:
        if not previous:
            model.disable_gradient_checkpointing()
    return dict(
        purpose="supervised_gradient_diagnostic",
        target_is_reference=False,
        target_sha256=row["target_sha256"],
        source_sha256=row["source_sha256"],
        target_latent_sha256=tensor_sha(z0),
        target_preprocessing=TARGET_PREPROCESSING,
        records=result,
    )
