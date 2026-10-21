"""Match the pinned raw-reference casting order without changing No-mask sampling."""

from contextlib import contextmanager

import numpy as np
from PIL import Image


def reference_tensor(image: Image.Image, *, device, dtype):
    import torch

    values = np.array(image, dtype=np.float32) / 255
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)
    # The trainer casts [0,1] pixels before normalizing, which matters in BF16.
    return tensor.to(device=device, dtype=dtype) * 2 - 1


@contextmanager
def joint_preprocessing(pipe, *, enabled: bool):
    if not enabled:
        yield
        return
    processor = pipe.image_processor
    original = processor.preprocess

    def preprocess(image, *, height, width, resize_mode):
        if image.mode != "RGB" or image.size != (width, height) or resize_mode != "crop":
            raise ValueError("Joint reference preprocessing cannot change RGB geometry")
        return reference_tensor(image, device=pipe._execution_device, dtype=pipe.vae.dtype)

    processor.preprocess = preprocess
    try:
        yield
    finally:
        processor.preprocess = original
