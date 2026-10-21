"""Explicit FLUX reference ordering and binary VAE-input contract."""

import numpy as np
from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.static import MASK_KINDS, binary_image, mask_rgb, validate_reference_size
from src.core.types import EditInput, GenerationOptions


def reference_images(
    edit_input: EditInput, controls: ControlBundle, options: GenerationOptions
) -> Image.Image | list[Image.Image]:
    controls.require_inference_safe()
    controls.validate_geometry(edit_input.source.size)
    if edit_input.strength is not None:
        raise NotImplementedError("Strength conditioning is not implemented")
    if controls.kind == ControlKind.NONE:
        if edit_input.mask is not None:
            raise ValueError("Compile the input mask into an explicit ControlBundle")
        return edit_input.source
    if controls.kind not in MASK_KINDS:
        raise NotImplementedError(f"FLUX.2 does not implement {controls.kind}")
    validate_reference_size(edit_input.source.size)
    if edit_input.source.size != (options.width, options.height):
        raise ValueError("Mask/source/output dimensions must match without implicit resizing")
    mask = binary_image(controls.maps["mask"])
    if edit_input.mask is not None and not np.array_equal(binary_image(edit_input.mask), mask):
        raise ValueError("Input mask differs from compiled control; supply one unambiguous mask")
    return [edit_input.source, mask_rgb(mask)]
