"""Evaluation-only latent targets, encoded after all common-state model predictions."""

import numpy as np

from src.core.io import sha256, write_json
from src.datasets.scene_manifest import load_role_supervision
from src.evaluation.feature_teacher import TARGET_PREPROCESSING, target_tensor


def encode_rgb(pipe, image, device, seed):
    import torch

    with torch.inference_mode():
        value = target_tensor(image, device=device, dtype=pipe.vae.dtype)
        latent = pipe._encode_vae_image(
            value, generator=torch.Generator(device=device).manual_seed(seed)
        )
        packed = pipe._pack_latents(latent).detach()
        if (
            packed.shape != (1, (image.height // 16) * (image.width // 16), 128)
            or not packed.isfinite().all()
        ):
            raise ValueError("Direction latent geometry or finite-value check failed")
        return packed


def export_supervision(pipe, root, rows, output, seed):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = {}
    for row in rows:
        supervision = load_role_supervision(root, row)
        latent = encode_rgb(pipe, supervision.target, "cuda", seed)
        labels = np.asarray(supervision.role_labels)
        h, w = labels.shape
        masks = np.stack(
            [
                (labels == k).reshape(h // 16, 16, w // 16, 16).mean((1, 3)).reshape(-1)
                for k in (0, 1, 2)
            ]
        ).astype(np.float32)
        path = output / f"{row['id']}.npz"
        np.savez(path, target=latent.float().cpu().numpy()[0], masks=masks)
        result[row["id"]] = dict(
            path=str(path.resolve()),
            sha256=sha256(path),
            target_sha256=row["target_sha256"],
            source_sha256=row["source_sha256"],
            role_sha256=row["role_sha256"],
            evidence_sha256=row["evidence_sha256"],
            target_preprocessing=TARGET_PREPROCESSING,
        )
    write_json(output / "index.json", result)
    return result
