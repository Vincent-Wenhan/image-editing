"""Strict paired LoRA/FP32 encoder checkpoints; missing features never fall back."""

import json
from pathlib import Path

from src.conditioning.feature_encoder import build_encoder
from src.conditioning.features import FEATURE_MODES, FEATURE_SCHEMA, EncoderConfig
from src.core.config import load_assets
from src.core.io import sha256


def save_encoder(path: Path, encoder, lora: Path, mode: str) -> dict:
    import torch
    from safetensors.torch import save_file

    if path.exists():
        raise FileExistsError(path)
    if mode not in FEATURE_MODES:
        raise ValueError("Unknown role feature checkpoint mode")
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in encoder.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("Non-finite encoder checkpoint")
    metadata = dict(
        schema=FEATURE_SCHEMA,
        mode=mode,
        encoder=encoder.config.as_dict(),
        base_revision=load_assets()["models"]["base"]["revision"],
        lora_sha256=sha256(lora),
    )
    save_file(tensors, str(path), metadata={"role_features": json.dumps(metadata, sort_keys=True)})
    return dict(
        path=str(path),
        sha256=sha256(path),
        metadata=metadata,
        parameters=sum(v.numel() for v in encoder.parameters()),
    )


def load_encoder(path: Path, lora: Path, *, device="cpu"):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["role_features"])
    if (
        metadata["schema"] != FEATURE_SCHEMA
        or metadata["mode"] not in FEATURE_MODES
        or metadata["lora_sha256"] != sha256(lora)
        or metadata["base_revision"] != load_assets()["models"]["base"]["revision"]
    ):
        raise ValueError("Role encoder schema/mode/base/LoRA SHA256 mismatch")
    encoder = build_encoder(EncoderConfig(**metadata["encoder"]))
    tensors = load_file(path, device="cpu")
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("Encoder must contain finite FP32 parameters")
    encoder.load_state_dict(tensors, strict=True)
    return encoder.to(device=device), metadata
