"""Mask geometry, deterministic controls and target-free inference loading."""

import json

import numpy as np
import pytest
from PIL import Image

from src.conditioning.controls import ControlKind
from src.conditioning.static import MaskSpec, mask_rgb, transform_mask
from src.core.io import read_rows
from src.datasets.control_export import donors, export_controls
from src.datasets.controls import ControlDataset


def test_dilation_preserves_binary_geometry_without_target_pixels():
    mask = np.zeros((9, 13), dtype=np.float32)
    mask[4, 6] = 1
    result = np.asarray(transform_mask(mask, MaskSpec(ControlKind.DILATED_MASK, 2)))
    expected = np.zeros((9, 13), dtype=np.uint8)
    expected[2:7, 4:9] = 255
    assert np.array_equal(result, expected)
    rgb = np.asarray(mask_rgb(Image.fromarray(expected)))
    assert rgb.shape == (9, 13, 3) and (rgb == expected[:, :, None]).all()
    with pytest.raises(ValueError, match="binary"):
        transform_mask(np.full((9, 13), 0.25), MaskSpec(ControlKind.STATIC_MASK))


@pytest.mark.parametrize("radius", [1, 2, 16])
def test_integral_dilation_matches_independent_pillow_reference(radius):
    from PIL import ImageFilter

    random = np.random.default_rng(42)
    values = (random.random((64, 80)) > 0.998).astype(np.uint8) * 255
    values[0, 0] = values[-1, -1] = 255
    image = Image.fromarray(values)
    expected = image.filter(ImageFilter.MaxFilter(2 * radius + 1))
    actual = transform_mask(image, MaskSpec(ControlKind.DILATED_MASK, radius))
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "kind", [ControlKind.STATIC_MASK, ControlKind.DILATED_MASK, ControlKind.SHUFFLED_MASK]
)
def test_export_and_read_validate_exact_conditions(spatial_data, tmp_path, kind):
    spec = MaskSpec(kind, 2 if kind == ControlKind.DILATED_MASK else 0, 42)
    root = tmp_path / kind
    export_controls(spatial_data, root, spec)
    controls = ControlDataset(root, spatial_data, kind)
    row = read_rows(spatial_data / "dev/manifest.csv")[0]
    bundle = controls.bundle(row, (128, 128))
    assert bundle.kind == kind and bundle.maps["mask"].size == (128, 128)
    assert set(np.asarray(bundle.maps["mask"]).ravel()) <= {0, 255}
    assert bundle.metadata["image_order"] == "source,mask"
    with pytest.raises(FileExistsError):
        export_controls(spatial_data, root, spec)
    # Actual inference condition loading remains usable when target supervision is absent.
    for split in ("train", "dev"):
        for path in (spatial_data / split / "target").glob("*.png"):
            path.unlink()
    again = ControlDataset(root, spatial_data, kind)
    assert np.array_equal(again.image(row), controls.image(row))


def test_shuffle_is_split_safe_stable_and_independent_of_manifest_order(spatial_data):
    spec = MaskSpec(ControlKind.SHUFFLED_MASK, shuffle_seed=17)
    for split in ("train", "dev"):
        rows = read_rows(spatial_data / split / "manifest.csv")
        mapping = donors(rows, spec)
        assert mapping == donors(list(reversed(rows)), spec)
        assert all(key != row["id"] and row["split"] == split for key, row in mapping.items())
        assert {row["id"] for row in mapping.values()} == {row["id"] for row in rows}
        with pytest.raises(ValueError, match="at least two"):
            donors(rows[:1], spec)


@pytest.mark.parametrize("change", ["file", "original", "order", "method", "donor", "content"])
def test_modified_conditions_fail_closed(spatial_data, tmp_path, change):
    root = tmp_path / "controls"
    export_controls(spatial_data, root, MaskSpec(ControlKind.STATIC_MASK))
    index_path = root / "controls.json"
    index = json.loads(index_path.read_text())
    row = read_rows(spatial_data / "dev/manifest.csv")[0]
    entry = index["splits"]["dev"]["entries"][row["id"]]
    if change == "file":
        (root / entry["path"]).unlink()
    elif change == "original":
        Image.new("L", (64, 64)).save(spatial_data / row["mask_path"])
    elif change == "order":
        index["order"].reverse()
    elif change == "method":
        index["spec"]["kind"] = "shuffled_mask"
    elif change == "donor":
        entry["donor_id"] = "train_other"
    else:
        from src.core.io import sha256

        Image.new("RGB", (64, 64)).save(root / entry["path"])
        entry["sha256"] = sha256(root / entry["path"])
    index_path.write_text(json.dumps(index))
    with pytest.raises((ValueError, FileNotFoundError)):
        ControlDataset(root, spatial_data, "static_mask")


def test_snapshot_preserves_manifests_and_excludes_old_caches(spatial_data, tmp_path):
    from src.datasets.snapshot import snapshot_pairs

    (spatial_data / "train/target/old_cache.bin").write_bytes(b"not a dataset asset")
    destination = tmp_path / "snapshot"
    result = snapshot_pairs(spatial_data, destination)
    assert result["validation"]["train"] == 3
    assert not (destination / "train/target/old_cache.bin").exists()
    for split in ("train", "dev"):
        assert (destination / split / "manifest.csv").read_bytes() == (
            spatial_data / split / "manifest.csv"
        ).read_bytes()
    with pytest.raises(FileExistsError):
        snapshot_pairs(spatial_data, destination)
