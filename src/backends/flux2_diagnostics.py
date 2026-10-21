"""Numerically check the two pinned VAE implementations and reference position IDs."""

import gc
from pathlib import Path

from src.backends.flux2_encoding import reference_tensor


def tensor_digest(tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def encoding_parity(pipe, images: list, vae_path: Path) -> dict:
    """Run only inside the leased worker with ai-toolkit as cwd, using its real code."""
    import torch
    from extensions_built_in.diffusion_models.flux2.src.autoencoder import (
        AutoEncoder,
        AutoEncoderParams,
    )
    from extensions_built_in.diffusion_models.flux2.src.sampling import encode_image_refs
    from safetensors.torch import load_file

    ae = AutoEncoder(AutoEncoderParams()).eval()
    ae.load_state_dict(load_file(str(vae_path)), strict=True)
    # The resident pipeline was loaded in BF16; compare the same weight precision
    # even during the FP32 arithmetic check, rather than restoring only one side.
    ae.to(device="cuda", dtype=torch.bfloat16)
    result = {}
    try:
        with torch.inference_mode():
            for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
                ae.to(dtype=dtype)
                pipe.vae.to(dtype=dtype)
                tensors = [reference_tensor(image, device="cuda", dtype=dtype) for image in images]
                pairs = []
                for tensor in tensors:
                    trained = ae.encode(tensor)
                    inferred = pipe._encode_vae_image(tensor, generator=None)
                    difference = (trained.float() - inferred.float()).abs()
                    relative_rmse = (
                        difference.square().mean().sqrt()
                        / trained.float().square().mean().sqrt().clamp_min(1e-8)
                    )
                    pairs.append(
                        dict(
                            max_abs=difference.max().item(),
                            relative_rmse=relative_rmse.item(),
                            shape=list(trained.shape),
                        )
                    )
                    tolerance = 0.001 if name == "float32" else 0.05
                    if not difference.isfinite().all() or relative_rmse > tolerance:
                        raise ValueError(f"{name} VAE encoding mismatch: {pairs[-1]}")
                train_tokens, train_ids = encode_image_refs(ae, tensors)
                infer_tokens, infer_ids = pipe.prepare_image_latents(
                    tensors, batch_size=1, generator=None, device="cuda", dtype=dtype
                )
                if not torch.equal(train_ids, infer_ids):
                    raise ValueError("Training/inference reference position IDs differ")
                if train_tokens.shape != infer_tokens.shape:
                    raise ValueError("Training/inference packed reference shapes differ")
                offsets, counts = torch.unique(infer_ids[0, :, 0], return_counts=True)
                if offsets.tolist() != [10, 20]:
                    raise ValueError("Expected source/mask reference coordinates 10,20")
                result[name] = dict(
                    references=pairs,
                    packed_shape=list(infer_tokens.shape),
                    time_offsets=offsets.tolist(),
                    token_counts=counts.tolist(),
                    packed_max_abs=(train_tokens.float() - infer_tokens.float()).abs().max().item(),
                )
    finally:
        pipe.vae.to(dtype=torch.bfloat16)
        del ae
        gc.collect()
        torch.cuda.empty_cache()
    return result
