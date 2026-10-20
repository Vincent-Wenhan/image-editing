"""Decode upstream image cells and align paired image geometry."""

import io

from PIL import Image

from src.conditioning.masks import editable_mask
from src.core.images import dimensions


def as_image(value: Image.Image | dict) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        source = io.BytesIO(value["bytes"]) if value.get("bytes") else value["path"]
        with Image.open(source) as image:
            return image.copy()
    raise TypeError(f"Unsupported image value: {type(value)}")


def align_images(
    source: Image.Image, target: Image.Image, mask: Image.Image, side: int | None = None
) -> tuple[Image.Image, Image.Image, Image.Image, str]:
    ratio = source.width / source.height
    if any(abs(image.width / image.height - ratio) > 0.001 for image in (target, mask)):
        raise ValueError("Different image aspect ratios; cannot safely align this pair")
    size = source.size
    if side:
        size = dimensions(size, side)
    source = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    target = target.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    mask, encoding = editable_mask(mask)
    return source, target, mask.resize(size, Image.Resampling.NEAREST), encoding
