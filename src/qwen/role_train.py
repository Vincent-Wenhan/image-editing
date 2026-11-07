"""Project-side Qwen CJT trainer with optional area-balanced Role-aware FM."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from src.core.io import sha256, write_json
from src.qwen.role_data import loader, prepare_cache
from src.qwen.role_loss import weighted_region_loss

TARGET_MODULES = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "add_k_proj",
    "add_q_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "img_mod.1",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "txt_mod.1",
)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _vae_config(model_path: Path) -> dict:
    return json.loads((model_path / "vae/config.json").read_text(encoding="utf-8"))


def _sigmas(scheduler, timesteps, device, dtype):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[indices].flatten()
    while sigma.ndim < 5:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _normalise(latents, mean, std):
    return (latents - mean) / std


def _packed_inputs(batch, pipeline_class, mean, std, scheduler, device, dtype, vae_scale_factor):
    target = _normalise(batch["target"].to(device=device, dtype=dtype), mean, std)
    controls = _normalise(batch["controls"].to(device=device, dtype=dtype), mean, std)
    prompt = batch["prompt"].to(device=device, dtype=dtype)
    prompt_mask = batch["prompt_mask"].to(device=device, dtype=torch.int32)
    bsz, channels, temporal, height, width = target.shape
    if temporal != 1 or controls.shape[2:] != (channels, temporal, height, width):
        raise ValueError("Cached Qwen latent geometry differs between target and controls")
    noise = torch.randn_like(target)
    density = torch.rand(bsz, device=device)
    indices = (density * scheduler.config.num_train_timesteps).long()
    timesteps = scheduler.timesteps[indices].to(device=device)
    sigma = _sigmas(scheduler, timesteps, device, target.dtype)
    noisy = (1.0 - sigma) * target + sigma * noise
    noisy_2d = noisy.squeeze(2)
    packed_noisy = pipeline_class._pack_latents(noisy_2d, bsz, channels, height, width)
    packed_controls, shapes = [], []
    for index in range(bsz):
        sample_controls = []
        sample_shapes = [(1, height // 2, width // 2)]
        for control in controls[index]:
            control_2d = control.squeeze(1).unsqueeze(0)
            sample_controls.append(
                pipeline_class._pack_latents(control_2d, 1, channels, height, width)
            )
            sample_shapes.append((1, height // 2, width // 2))
        packed_controls.append(torch.cat(sample_controls, dim=1))
        shapes.append(sample_shapes)
    image_latents = torch.cat(packed_controls, dim=0)
    model_input = torch.cat([packed_noisy, image_latents], dim=1)
    return {
        "target": target,
        "noise": noise,
        "noisy": model_input,
        "packed_noisy": packed_noisy,
        "timesteps": timesteps,
        "sigma": sigma,
        "prompt": prompt,
        "prompt_mask": prompt_mask,
        "img_shapes": shapes,
        "txt_seq_lens": prompt_mask.sum(dim=1).tolist(),
        "height": height,
        "width": width,
        "vae_scale_factor": vae_scale_factor,
    }


def _flow_error(transformer, pipeline_class, values):
    model_pred = transformer(
        hidden_states=values["noisy"],
        timestep=values["timesteps"] / 1000,
        guidance=None,
        encoder_hidden_states_mask=values["prompt_mask"],
        encoder_hidden_states=values["prompt"],
        img_shapes=values["img_shapes"],
        txt_seq_lens=values["txt_seq_lens"],
        return_dict=False,
    )[0]
    model_pred = model_pred[:, : values["packed_noisy"].size(1)]
    model_pred = pipeline_class._unpack_latents(
        model_pred,
        values["height"] * values["vae_scale_factor"],
        values["width"] * values["vae_scale_factor"],
        values["vae_scale_factor"],
    )
    if model_pred.ndim == 4:
        model_pred = model_pred.unsqueeze(1)
    elif model_pred.ndim == 5 and model_pred.shape[2] == 1:
        model_pred = model_pred.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unexpected Qwen prediction shape: {tuple(model_pred.shape)}")
    target = (values["noise"] - values["target"]).permute(0, 2, 1, 3, 4)
    weighting = 1.0
    return (model_pred.float() - target.float()).square() * weighting


def _save_checkpoint(accelerator, transformer, pipeline_class, output: Path, step: int) -> str:
    if not accelerator.is_main_process:
        return ""
    path = output / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model = accelerator.unwrap_model(transformer)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
    converted = {}
    for key, value in state.items():
        converted[key.replace(".lora.down.", ".lora_A.").replace(".lora.up.", ".lora_B.")] = value
    pipeline_class.save_lora_weights(path, converted, safe_serialization=True)
    return str(path / "pytorch_lora_weights.safetensors")


def train(record: dict, args) -> dict:
    config = json.loads(json.dumps(args.config)) if isinstance(args.config, dict) else args.config
    if config.get("quantize") or config.get("adam8bit"):
        raise ValueError(
            "Project Role-aware trainer currently requires full BF16 weights and AdamW"
        )
    loss_mode = args.loss_mode or config.get("loss_mode", "ordinary")
    role_weight = float(
        args.role_loss_weight
        if args.role_loss_weight is not None
        else config.get("role_loss_weight", 0.0)
    )
    if loss_mode not in {"ordinary", "role_aware"} or not 0 <= role_weight <= 1:
        raise ValueError("Invalid Role-aware loss configuration")
    if loss_mode == "ordinary" and role_weight:
        raise ValueError("Ordinary FM must use role_loss_weight=0")
    configured_steps = int(config["max_train_steps"])
    steps = int(args.max_steps or configured_steps)
    if steps < 1 or steps > configured_steps:
        raise ValueError("--max-steps must be in [1, configured max_train_steps]")
    seed = int(config["seed"])
    _seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision=config["mixed_precision"],
        project_config=ProjectConfiguration(project_dir=record["train_config"]),
    )
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    root = Path(record["data_root"])
    model_path = Path(record["model_path"])
    samples = prepare_cache(
        root,
        Path(record["train_manifest"]),
        model_path,
        device=accelerator.device,
        dtype=dtype,
        seed=seed,
    )
    train_loader = loader(
        samples,
        seed=seed,
        workers=int(config["data_config"]["num_workers"]),
        batch_size=int(config["train_batch_size"]),
    )
    from src.qwen.inference import _pipeline_class

    pipeline_class = _pipeline_class()
    vae_cfg = _vae_config(model_path)
    latent_channels = int(vae_cfg["z_dim"])
    mean = torch.tensor(vae_cfg["latents_mean"], device=accelerator.device, dtype=dtype).view(
        1, latent_channels, 1, 1, 1
    )
    std = torch.tensor(vae_cfg["latents_std"], device=accelerator.device, dtype=dtype).view(
        1, latent_channels, 1, 1, 1
    )
    vae_scale_factor = 2 ** len(vae_cfg["temperal_downsample"])
    transformer = QwenImageTransformer2DModel.from_pretrained(
        str(model_path), subfolder="transformer"
    ).to(device=accelerator.device, dtype=dtype)
    transformer.add_adapter(
        LoraConfig(
            r=int(config["rank"]),
            lora_alpha=int(config["rank"]),
            init_lora_weights="gaussian",
            target_modules=list(TARGET_MODULES),
        )
    )
    transformer.requires_grad_(False)
    for name, parameter in transformer.named_parameters():
        parameter.requires_grad = "lora" in name
    transformer.train()
    transformer.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in transformer.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        weight_decay=float(config["adam_weight_decay"]),
        eps=float(config["adam_epsilon"]),
    )
    scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=int(config["lr_warmup_steps"]),
        num_training_steps=steps,
    )
    transformer, optimizer, train_loader, scheduler = accelerator.prepare(
        transformer, optimizer, train_loader, scheduler
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_path), subfolder="scheduler"
    )
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    iterator = iter(train_loader)
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        with accelerator.accumulate(transformer):
            values = _packed_inputs(
                batch,
                pipeline_class,
                mean,
                std,
                noise_scheduler,
                accelerator.device,
                dtype,
                vae_scale_factor,
            )
            error = _flow_error(transformer, pipeline_class, values)
            global_loss = error.mean()
            masks = batch["role_masks"].to(device=accelerator.device, dtype=torch.float32)
            role_loss, role_records = weighted_region_loss(
                error, masks, min_pixels=float(config.get("role_min_pixels", 4.0))
            )
            loss = (1.0 - role_weight) * global_loss + role_weight * role_loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    transformer.parameters(), float(config["max_grad_norm"])
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        records.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "global_loss": float(global_loss.detach()),
                "role_loss_weight": role_weight,
                **role_records,
            }
        )
        if step % int(config["checkpointing_steps"]) == 0 or step == steps:
            checkpoint = _save_checkpoint(accelerator, transformer, pipeline_class, output, step)
            if accelerator.is_main_process:
                write_json(
                    output.parent / "train_report.json",
                    {
                        "status": "running",
                        "loss_mode": loss_mode,
                        "role_loss_weight": role_weight,
                        "steps_completed": step,
                        "checkpoint": checkpoint,
                        "records": records,
                    },
                )
    accelerator.wait_for_everyone()
    report = {
        "status": "complete",
        "configured_steps": configured_steps,
        "loss_mode": loss_mode,
        "role_loss_weight": role_weight,
        "steps_completed": steps,
        "checkpoint": str(output / f"checkpoint-{steps}" / "pytorch_lora_weights.safetensors"),
        "checkpoint_sha256": sha256(
            output / f"checkpoint-{steps}" / "pytorch_lora_weights.safetensors"
        ),
        "records": records,
    }
    if accelerator.is_main_process:
        write_json(output.parent / "train_report.json", report)
    return report
