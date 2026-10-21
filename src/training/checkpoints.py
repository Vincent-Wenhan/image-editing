"""Validate saved adapters before declaring training complete."""

from pathlib import Path

from src.core.io import sha256


def validate_checkpoint(checkpoint: str | Path) -> dict:
    import torch
    from safetensors import safe_open

    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        if not keys or not any("lora" in key.lower() for key in keys):
            raise ValueError("Saved checkpoint contains no LoRA tensors")
        if any(not torch.isfinite(weights.get_tensor(key)).all() for key in keys):
            raise ValueError("Saved LoRA contains non-finite weights")
    return dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        lora_tensors=len(keys),
        checkpoint_finite=True,
    )
