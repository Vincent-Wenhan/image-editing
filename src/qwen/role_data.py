"""Target-supervised training cache for the project-side Qwen trainer."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from src.core.io import read_rows, sha256
from src.core.paths import data_path
from src.datasets.scene_validation import checked_image
from src.qwen.conditions import condition_path
from src.qwen.inference import _pipeline_class
from src.qwen.role_loss import build_region_masks, latent_masks

if TYPE_CHECKING:
    import torch

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


@dataclass
class PreparedSample:
    sample_id: str
    target: torch.Tensor
    controls: torch.Tensor
    prompt: torch.Tensor
    prompt_mask: torch.Tensor
    role_masks: torch.Tensor


def _dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = (target_area * ratio) ** 0.5
    return round(width / 32) * 32, round(width / ratio / 32) * 32


def _load_evidence(root: Path, row: dict) -> dict[str, np.ndarray]:
    path = data_path(root, row["evidence_path"])
    if sha256(path) != row["evidence_sha256"]:
        raise ValueError(f"Role evidence SHA256 changed: {row['id']}")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in arrays.files}


def _image(root: Path, relative: str) -> Image.Image:
    with Image.open(condition_path(root, relative)) as image:
        return image.convert("RGB")


def _encode_vae(pipeline, vae, image: Image.Image, device, dtype) -> torch.Tensor:
    width, height = image.size
    vae_width, vae_height = _dimensions(VAE_IMAGE_SIZE, width / height)
    tensor = pipeline.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)
    tensor = tensor.to(device=device, dtype=dtype)
    latent = vae.encode(tensor).latent_dist.sample()
    latent = latent[0]
    if latent.shape[0] == vae.config.z_dim:
        latent = latent[:, :1]
    elif latent.shape[1] == vae.config.z_dim:
        latent = latent[:1].permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Unexpected Qwen VAE latent shape: {tuple(latent.shape)}")
    return latent.detach().cpu()


def prepare_cache(
    root: Path,
    manifest: Path,
    model_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> list[PreparedSample]:
    """Encode the exact CJT controls and target supervision once per run."""
    import torch

    split = manifest.stem
    if split not in {"train", "dev"}:
        raise ValueError(f"Qwen training cache requires train/dev manifest: {manifest}")
    rows = {row["id"]: row for row in read_rows(root / split / "manifest.csv")}
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    pipeline_class = _pipeline_class()
    pipeline = pipeline_class.from_pretrained(
        str(model_path), transformer=None, vae=None, torch_dtype=dtype, local_files_only=True
    )
    pipeline.to(device)
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        str(model_path), subfolder="vae", torch_dtype=dtype, local_files_only=True
    ).to(device)
    torch.manual_seed(seed)
    result = []
    with torch.no_grad():
        for sample in samples:
            row = rows[sample["id"]]
            source = _image(root, row["source_path"])
            control = _image(root, sample["mask_image_path"])
            images = []
            for image in (source, control):
                width, height = _dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
                images.append(pipeline.image_processor.resize(image, height, width))
            prompt, prompt_mask = pipeline.encode_prompt(
                image=images,
                prompt=[row["instruction"]],
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=1024,
            )
            target = _encode_vae(pipeline, vae, _image(root, row["target_path"]), device, dtype)
            controls = torch.stack(
                [
                    _encode_vae(pipeline, vae, source, device, dtype),
                    _encode_vae(pipeline, vae, control, device, dtype),
                ]
            )
            labels = np.asarray(checked_image(root, row, "role"))
            valid = np.asarray(checked_image(root, row, "valid")) > 0
            regions = build_region_masks(
                _load_evidence(root, row),
                labels,
                valid,
                shadow_threshold=float(
                    json.loads((root / "dataset.json").read_text())["config"]["roles"][
                        "shadow_high"
                    ]
                ),
            )
            masks = latent_masks(regions, target.shape[-2], target.shape[-1])
            result.append(
                PreparedSample(
                    sample_id=row["id"],
                    target=target,
                    controls=controls,
                    prompt=prompt[0].cpu(),
                    prompt_mask=prompt_mask[0].cpu(),
                    role_masks=torch.from_numpy(masks),
                )
            )
            print(f"Cached {row['id']}: target={tuple(target.shape)}", flush=True)
    vae.to("cpu")
    pipeline.to("cpu")
    del vae, pipeline, prompt, prompt_mask
    gc.collect()
    torch.cuda.empty_cache()
    return result


def collate(samples: list[PreparedSample]) -> dict:
    import torch

    max_length = max(sample.prompt.shape[0] for sample in samples)
    prompts, prompt_masks = [], []
    for sample in samples:
        pad = max_length - sample.prompt.shape[0]
        prompts.append(
            torch.cat([sample.prompt, sample.prompt.new_zeros(pad, sample.prompt.shape[1])])
        )
        prompt_masks.append(
            torch.cat([sample.prompt_mask, torch.zeros(pad, dtype=sample.prompt_mask.dtype)])
        )
    return {
        "sample_id": [sample.sample_id for sample in samples],
        "target": torch.stack([sample.target for sample in samples]),
        "controls": torch.stack([sample.controls for sample in samples]),
        "prompt": torch.stack(prompts),
        "prompt_mask": torch.stack(prompt_masks),
        "role_masks": torch.stack([sample.role_masks for sample in samples]),
    }
