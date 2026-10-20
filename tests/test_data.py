"""Regression tests for data corruption and train/dev leakage."""

from pathlib import Path

import pytest
from PIL import Image

from src.conditioning.masks import editable_mask
from src.core.io import read_rows, write_rows
from src.datasets.export import export_rows
from src.datasets.images import align_images
from src.datasets.validation import validate


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


def test_image_checksum_corruption_fails(paired_data):
    row = read_rows(paired_data / "dev/manifest.csv")[0]
    Image.new("RGB", (32, 48), "yellow").save(paired_data / row["source_path"])
    with pytest.raises(ValueError, match="Checksum"):
        validate(paired_data)


@pytest.mark.parametrize("change", ["duplicate", "split", "escape"])
def test_manifest_identity_split_and_path_guards(paired_data, change):
    manifest = paired_data / "dev/manifest.csv"
    rows = read_rows(manifest)
    if change == "duplicate":
        rows.append(rows[0].copy())
    elif change == "split":
        rows[0]["split"] = "train"
    else:
        rows[0]["target_path"] = "../outside.png"
    write_rows(manifest, rows)
    with pytest.raises(ValueError):
        validate(paired_data)


def test_threshold_semantics_for_gray_and_alpha_masks():
    gray = Image.new("L", (2, 1))
    gray.putdata([127, 128])
    mask, encoding = editable_mask(gray)
    assert [mask.getpixel((x, 0)) for x in range(2)] == [0, 255] and encoding == "white_is_edit"
    rgba = Image.new("RGBA", (2, 1), "blue")
    rgba.putalpha(gray)
    mask, encoding = editable_mask(rgba)
    assert [mask.getpixel((x, 0)) for x in range(2)] == [255, 0] and encoding == "inverse_alpha"
