"""Matched Qwen CJT optimization with ordinary or region-balanced FM."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from src.core.io import sha256, write_json
from src.core.metadata import tensor_sha
from src.infra.gpu import require_exclusive_context
from src.qwen.role_cache import shared_cache
from src.qwen.role_checkpoint import restore_checkpoint, save_checkpoint, update_audit
from src.qwen.role_data import collate
from src.qwen.role_flow import flow_error, packed_inputs
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
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def batch_schedule(sample_count: int, seed: int, steps: int) -> list[int]:
    """A local RNG fixes batch order independently of model/cache/loss RNGs."""
    if sample_count < 1 or steps < 1:
        raise ValueError("Training needs nonempty samples and positive steps")
    rng = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < steps:
        schedule.extend(rng.permutation(sample_count).tolist())
    return schedule[:steps]


def _configuration(args) -> tuple[dict, str, float, int]:
    config = dict(args.config)
    mode = args.loss_mode or config.get("loss_mode", "ordinary")
    weight = args.role_loss_weight
    weight = float(config.get("role_loss_weight", 0.0) if weight is None else weight)
    steps = config["max_train_steps"] if args.max_steps is None else args.max_steps
    if mode not in {"ordinary", "role_aware"} or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid Qwen loss configuration")
    if mode == "ordinary" and weight:
        raise ValueError("Ordinary FM must use role_loss_weight=0")
    upper = max(240, config["max_train_steps"]) if args.resume else config["max_train_steps"]
    if int(steps) != steps or not 1 <= steps <= upper:
        raise ValueError("Steps must be in [1, configured max_train_steps]")
    if config.get("quantize") or config.get("adam8bit") or config["mixed_precision"] != "bf16":
        raise ValueError("Matched Qwen trainer requires BF16 weights and AdamW")
    if config["train_batch_size"] != 1 or config["gradient_accumulation_steps"] != 1:
        raise ValueError("Matched Qwen trainer currently supports batch1/accumulation1 only")
    return config, mode, weight, int(steps)


def train(record: dict, args) -> dict:
    import torch
    from accelerate import Accelerator

    config, mode, role_weight, steps = _configuration(args)
    seed, output = int(config["seed"]), Path(config["output_dir"])
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1)
    if accelerator.num_processes != 1:
        raise ValueError("Use one leased GPU per matched Qwen run")
    dtype, device = torch.bfloat16, accelerator.device
    probe = torch.empty(1, device=device)
    torch.cuda.synchronize()
    del probe
    require_exclusive_context()
    # Keep the first post-launch contention check before slow diffusers/Qwen imports.
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig

    from src.qwen.inference import _pipeline_class

    model_path = Path(record["model_path"])
    cache_root = Path(args.cache_root) if args.cache_root else output.parent / "encoding_cache"
    samples = shared_cache(record, cache_root, device=device, dtype=dtype)
    cache_sha = sha256(cache_root / "cache.json")
    require_exclusive_context()
    # Cache reuse must not change LoRA initialization or either training RNG stream.
    _seed(seed)
    pipeline_class = _pipeline_class()
    vae_cfg = json.loads((model_path / "vae/config.json").read_text())
    mean, std = [
        torch.tensor(vae_cfg[key], device=device, dtype=dtype).view(1, vae_cfg["z_dim"], 1, 1, 1)
        for key in ("latents_mean", "latents_std")
    ]
    vae_scale_factor = 2 ** len(vae_cfg["temperal_downsample"])
    transformer = QwenImageTransformer2DModel.from_pretrained(
        str(model_path), subfolder="transformer", torch_dtype=dtype, local_files_only=True
    )
    require_exclusive_context()
    transformer.to(device=device)
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
    start_step = 0
    if args.resume:
        start_step = restore_checkpoint(args.resume, transformer, optimizer, scheduler)
        if start_step >= steps:
            raise ValueError("Continuation must request more updates than the restored checkpoint")
    transformer, optimizer, scheduler = accelerator.prepare(transformer, optimizer, scheduler)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_path), subfolder="scheduler", local_files_only=True
    )
    output.mkdir(parents=True, exist_ok=True)
    snapshots, records = [], []
    if start_step:
        previous = json.loads((output.parent / "train_report.json").read_text())
        if (
            previous["cache_sha256"] != cache_sha
            or previous["loss_mode"] != mode
            or previous["role_loss_weight"] != role_weight
        ):
            raise ValueError("Qwen continuation cache/loss differs from the previous run")
        snapshots, records = previous["snapshots"], previous["records"]
        if previous["steps_completed"] != start_step:
            raise ValueError("Resume checkpoint is not the last recorded update")
    else:
        snapshots.append(
            save_checkpoint(
                accelerator, transformer, pipeline_class, output, 0, optimizer, scheduler
            )
        )
    initial_checkpoint = snapshots[0]["checkpoint"]
    report = {
        "status": "running",
        "configured_steps": int(config["max_train_steps"]),
        "loss_mode": mode,
        "role_loss_weight": role_weight,
        "steps_completed": start_step,
        "cache_sha256": cache_sha,
        "initial_checkpoint_sha256": snapshots[0]["sha256"],
        "train_ids": [sample.sample_id for sample in samples],
        "snapshots": snapshots,
        "records": records,
    }
    write_json(output.parent / "train_report.json", report)
    torch.cuda.reset_peak_memory_stats()
    schedule = batch_schedule(len(samples), seed, steps)
    for step in range(start_step + 1, steps + 1):
        require_exclusive_context()
        started = time.perf_counter()
        batch = collate([samples[schedule[step - 1]]])
        values = packed_inputs(batch, pipeline_class, mean, std, noise_scheduler, device, dtype)
        trace = {
            "sample_ids": batch["sample_id"],
            "timesteps": values["timesteps"].tolist(),
            "sigmas": values["sigma"].flatten().float().tolist(),
            **{f"{key}_sha256": tensor_sha(values[key]) for key in ("noisy", "target", "prompt")},
        }
        with accelerator.accumulate(transformer):
            error = flow_error(transformer, pipeline_class, values, vae_scale_factor)
            global_loss = error.mean()
            role_loss, details = weighted_region_loss(
                error,
                batch["role_masks"].to(device=device),
                min_pixels=float(config.get("role_min_pixels", 4.0)),
            )
            loss = (1.0 - role_weight) * global_loss + role_weight * role_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite FM loss at update {step}")
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(
                transformer.parameters(), float(config["max_grad_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at update {step}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        records.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "global_loss": float(global_loss.detach()),
                "grad_norm": float(grad_norm),
                "seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                **trace,
                **details,
            }
        )
        print(json.dumps(records[-1]), flush=True)
        del error, loss, role_loss, global_loss, values
        if step % int(config["checkpointing_steps"]) == 0 or step == steps:
            snapshot = save_checkpoint(
                accelerator, transformer, pipeline_class, output, step, optimizer, scheduler
            )
            snapshots.append(snapshot)
            report.update(
                steps_completed=step,
                checkpoint=snapshot["checkpoint"],
                checkpoint_sha256=snapshot["sha256"],
            )
            write_json(output.parent / "train_report.json", report)
    report.update(
        status="complete", **update_audit(Path(initial_checkpoint), Path(report["checkpoint"]))
    )
    if report["changed_parameters"] == 0:
        raise ValueError("Qwen adapter parameters did not update")
    write_json(output.parent / "train_report.json", report)
    accelerator.end_training()
    return report
