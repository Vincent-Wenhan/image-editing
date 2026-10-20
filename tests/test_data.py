"""Regression tests for data corruption and train/dev leakage."""

from pathlib import Path

import pytest
from PIL import Image

from image_editing.data import align_images, editable_mask, export_rows
from image_editing.validation import validate


def sample(session, color):
    return dict(
        img_id=session,
        turn_index=1,
        instruction="Make the square red",
        source_img=Image.new("RGB", (32, 48), color),
        target_img=Image.new("RGB", (32, 48), "red"),
        mask_img=Image.new("L", (32, 48), 255),
    )


def prepare(root):
    export_rows([sample("1", "blue")], root, "train", 1)
    export_rows([sample("2", "green")], root, "dev", 1)


def test_missing_mask_fails(tmp_path):
    prepare(tmp_path)
    next((tmp_path / "dev/mask").glob("*.png")).unlink()
    with pytest.raises(FileNotFoundError):
        validate(tmp_path)


def test_session_leakage_fails(tmp_path):
    export_rows([sample("1", "blue")], tmp_path, "train", 1)
    export_rows([sample("1", "green")], tmp_path, "dev", 1)
    with pytest.raises(ValueError, match="leakage"):
        validate(tmp_path)


def test_content_leakage_fails(tmp_path):
    export_rows([sample("1", "blue")], tmp_path, "train", 1)
    export_rows([sample("2", "blue")], tmp_path, "dev", 1)
    with pytest.raises(ValueError, match="leakage"):
        validate(tmp_path)


def test_geometry_and_caption_integrity(tmp_path):
    prepare(tmp_path)
    assert validate(tmp_path)["train"] == 1
    with Image.open(next((tmp_path / "train/reference").glob("*.png"))) as image:
        assert image.size == (32, 48)
    caption = next((tmp_path / "train/target").glob("*.txt"))
    caption.write_text("different caption")
    with pytest.raises(ValueError, match="Caption mismatch"):
        validate(tmp_path)


def test_failed_export_cleans_partial_and_preserves_existing(tmp_path):
    with pytest.raises(ValueError, match="Only found"):
        export_rows([sample("1", "blue")], tmp_path, "train", 2)
    assert not list(Path(tmp_path).iterdir())
    export_rows([sample("1", "blue")], tmp_path, "train", 1)
    with pytest.raises(FileExistsError):
        export_rows([sample("2", "green")], tmp_path, "train", 1)


def test_magicbrush_transparency_and_different_native_resolutions():
    source = Image.new("RGB", (500, 500), "blue")
    target = Image.new("RGB", (1024, 1024), "red")
    mask = Image.new("RGBA", (1024, 1024), (100, 200, 50, 255))
    mask.paste((0, 0, 0, 0), (256, 256, 768, 768))
    src, tgt, editable, encoding = align_images(source, target, mask, side=512)
    assert src.size == tgt.size == editable.size == (512, 512)
    assert encoding == "inverse_alpha"
    assert editable.getpixel((256, 256)) == 255
    assert editable.getpixel((0, 0)) == 0
    assert {index for index, count in enumerate(editable.histogram()) if count} == {0, 255}


def test_misaligned_aspect_ratios_are_rejected():
    with pytest.raises(ValueError, match="aspect ratios"):
        align_images(
            Image.new("RGB", (500, 400)),
            Image.new("RGB", (1024, 1024)),
            Image.new("L", (1024, 1024)),
        )


def test_colored_image_cannot_be_interpreted_as_a_gray_mask():
    image = Image.new("RGB", (2, 1), "red")
    image.putpixel((1, 0), (0, 255, 255))
    with pytest.raises(ValueError, match="ambiguous"):
        editable_mask(image)
