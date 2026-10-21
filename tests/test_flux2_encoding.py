"""Reference pixel casting order and scoped restoration, independently of CUDA."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.backends.flux2_encoding import joint_preprocessing, reference_tensor


def test_raw_reference_pixels_match_training_cast_order():
    torch = pytest.importorskip("torch")
    values = np.array([[[127, 128, 129], [0, 255, 128]]], dtype=np.uint8)
    image = Image.fromarray(values)
    raw = torch.from_numpy(values.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    encoded = reference_tensor(image, device="cpu", dtype=torch.bfloat16)
    assert torch.equal(encoded, raw.to(torch.bfloat16) * 2 - 1)
    assert not torch.equal(encoded, (raw * 2 - 1).to(torch.bfloat16))


def test_joint_preprocessor_restores_no_mask_behavior_after_failure():
    torch = pytest.importorskip("torch")

    def original(**kwargs):
        return "legacy"

    pipe = SimpleNamespace(
        image_processor=SimpleNamespace(preprocess=original),
        vae=SimpleNamespace(dtype=torch.bfloat16),
        _execution_device="cpu",
    )
    with pytest.raises(RuntimeError, match="test"):
        with joint_preprocessing(pipe, enabled=True):
            assert pipe.image_processor.preprocess is not original
            image = Image.new("RGB", (64, 64))
            tensor = pipe.image_processor.preprocess(image, width=64, height=64, resize_mode="crop")
            assert tensor.shape == (1, 3, 64, 64) and tensor.min() == -1
            raise RuntimeError("test")
    assert pipe.image_processor.preprocess is original
    with joint_preprocessing(pipe, enabled=False):
        assert pipe.image_processor.preprocess is original
