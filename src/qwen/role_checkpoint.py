"""Preserved LoRA snapshots, update audits and exact optimizer/RNG continuation."""

from __future__ import annotations

import random
from pathlib import Path

from src.core.io import sha256


def save_checkpoint(accelerator, model, pipeline_class, output, step, optimizer, scheduler):
    import torch
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(output) / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=False)
    state = get_peft_model_state_dict(accelerator.unwrap_model(model))
    converted = {
        key.replace(".lora.down.", ".lora_A.").replace(".lora.up.", ".lora_B."): value
        for key, value in convert_state_dict_to_diffusers(state).items()
    }
    pipeline_class.save_lora_weights(str(path), converted, safe_serialization=True)
    weights = path / "pytorch_lora_weights.safetensors"
    saved = load_file(str(weights))
    if set(saved) != {f"transformer.{key}" for key in converted}:
        raise ValueError("Saved Qwen LoRA keys differ from the trained adapter")
    for key, value in converted.items():
        if not torch.isfinite(value).all() or not torch.equal(
            saved[f"transformer.{key}"], value.cpu()
        ):
            raise ValueError("Saved Qwen LoRA tensors differ or contain non-finite parameters")
    torch.save(
        {
            "step": step,
            "lora": {key: value.detach().cpu() for key, value in state.items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
            "python_rng": random.getstate(),
        },
        path / "training_state.pt",
    )
    return {"step": step, "checkpoint": str(weights), "sha256": sha256(weights)}


def restore_checkpoint(path, model, optimizer, scheduler):
    import torch
    from peft.utils import set_peft_model_state_dict

    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    result = set_peft_model_state_dict(model, state["lora"])
    if result.unexpected_keys:
        raise ValueError("Unexpected adapter parameters during Qwen continuation")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return int(state["step"])


def update_audit(initial: Path, final: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    before, after = load_file(str(initial)), load_file(str(final))
    if set(before) != set(after):
        raise ValueError("Initialized and trained adapter parameters differ")
    squared, changed, count = 0.0, 0, 0
    for key in before:
        delta = after[key].float() - before[key].float()
        squared += float(delta.square().sum())
        changed += int(torch.count_nonzero(delta))
        count += delta.numel()
    return {"parameter_update_l2": squared**0.5, "changed_parameters": changed, "parameters": count}
