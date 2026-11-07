"""Pipeline-aligned Qwen latent packing, timestep sampling and FM errors."""

from __future__ import annotations


def packed_inputs(batch, pipeline_class, mean, std, scheduler, device, dtype):
    import torch
    from diffusers.training_utils import compute_density_for_timestep_sampling

    target = (batch["target"].to(device=device, dtype=dtype) - mean) / std
    controls = (
        batch["controls"].to(device=device, dtype=dtype) - mean.unsqueeze(1)
    ) / std.unsqueeze(1)
    bsz, channels, temporal, height, width = target.shape
    if temporal != 1 or controls.shape[2:] != (channels, temporal, height, width):
        raise ValueError("Cached Qwen target/control latent geometry differs")
    prompt = batch["prompt"].to(device=device, dtype=dtype)
    prompt_mask = batch["prompt_mask"].to(device=device, dtype=torch.int32)
    noise = torch.randn_like(target)
    # The pinned trainer samples density on CPU and indexes a CPU scheduler table.
    density = compute_density_for_timestep_sampling(
        weighting_scheme="none", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29
    )
    indices = (density.cpu() * scheduler.config.num_train_timesteps).long()
    timesteps = scheduler.timesteps.cpu()[indices].to(device=device)
    sigma = scheduler.sigmas.cpu()[indices].to(device=device, dtype=dtype).view(bsz, 1, 1, 1, 1)
    noisy = (1.0 - sigma) * target + sigma * noise
    packed_noisy = pipeline_class._pack_latents(noisy.squeeze(2), bsz, channels, height, width)
    packed_controls, shapes = [], []
    for sample_controls in controls:
        packed_controls.append(
            torch.cat(
                [
                    pipeline_class._pack_latents(control, 1, channels, height, width)
                    for control in sample_controls
                ],
                dim=1,
            )
        )
        shapes.append([(1, height // 2, width // 2)] * (1 + len(sample_controls)))
    return {
        "target": target,
        "noise": noise,
        "noisy": torch.cat([packed_noisy, torch.cat(packed_controls, dim=0)], dim=1),
        "packed_noisy": packed_noisy,
        "timesteps": timesteps,
        "sigma": sigma,
        "prompt": prompt,
        "prompt_mask": prompt_mask,
        "img_shapes": shapes,
        "txt_seq_lens": prompt_mask.sum(dim=1).tolist(),
    }


def flow_error(transformer, pipeline_class, values, vae_scale_factor):
    from diffusers.training_utils import compute_loss_weighting_for_sd3

    model_pred = transformer(
        hidden_states=values["noisy"],
        timestep=values["timesteps"] / 1000,
        guidance=None,
        encoder_hidden_states_mask=values["prompt_mask"],
        encoder_hidden_states=values["prompt"],
        img_shapes=values["img_shapes"],
        txt_seq_lens=values["txt_seq_lens"],
        return_dict=False,
    )[0][:, : values["packed_noisy"].size(1)]
    model_pred = pipeline_class._unpack_latents(
        model_pred,
        values["target"].shape[-2] * vae_scale_factor,
        values["target"].shape[-1] * vae_scale_factor,
        vae_scale_factor,
    )
    if model_pred.shape != values["target"].shape:
        raise ValueError("Unpacked Qwen prediction differs from the FM target grid")
    # Keep the pinned trainer's [B,T,C,H,W] reduction layout and FP32 error.
    prediction = model_pred.permute(0, 2, 1, 3, 4).float()
    target = (values["noise"] - values["target"]).permute(0, 2, 1, 3, 4).float()
    weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=values["sigma"])
    return weighting.float() * (prediction - target).square()
