"""Synthetic evidence fixtures verify causal labels; these are not Blender renders."""

import copy
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from src.conditioning.role_truth import derive_roles
from src.core.config import load_yaml
from src.core.io import read_rows, sha256, write_json, write_rows
from src.core.metadata import config_digest
from src.core.paths import CONFIG_ROOT
from src.datasets.scene import generate
from src.datasets.scene_config import scene_group, validate_scene_config
from src.datasets.scene_pairs import compile_pairs
from src.datasets.scene_validation import validate_scene_dataset


def evidence(color, strength=1):
    background = np.full((64, 64, 3), 0.4, dtype=np.float32)
    old = np.zeros((64, 64), dtype=np.float32)
    old[10:26, 10:26] = 1
    new = np.zeros_like(old)
    new[10:26, 34:50] = 1
    source_off, target_off = background.copy(), background.copy()
    source_off[old > 0] = color
    target_off[new > 0] = color
    source, target = source_off.copy(), target_off.copy()
    source[40:48, 10:20] -= 0.1
    target[40:48, 38:48] -= 0.1
    if strength == 0:
        target, target_off, new = source.copy(), source_off.copy(), old.copy()
    return dict(
        source=source,
        target=target,
        source_no_shadow=source_off,
        target_no_shadow=target_off,
        source_occupancy=old,
        target_occupancy=new,
        background=background,
    )


@pytest.fixture
def scene_data(tmp_path):
    config = load_yaml(CONFIG_ROOT / "datasets/blender_translation.yaml")
    config["renderer"].update(resolution=64, samples=1)
    config["scenes"] = [config["scenes"][0], config["scenes"][2]]
    config["strengths"] = [0.0, 1.0]
    root = tmp_path / "scenes"
    root.mkdir()
    (root / "scene_config.yaml").write_text(yaml.safe_dump(config))
    groups = []
    for spec in config["scenes"]:
        folder = root / "raw" / spec["id"]
        folder.mkdir(parents=True)
        entries, files = [], {}
        for index, strength in enumerate(config["strengths"]):
            path = folder / f"s{index:03d}.npz"
            np.savez_compressed(path, **evidence(spec["color"], strength))
            entries.append(
                dict(
                    strength=strength,
                    displacement=strength * config["max_translation_widths"] * spec["object_width"],
                    evidence_path=path.relative_to(root / "raw").as_posix(),
                    sha256=sha256(path),
                )
            )
        # Hash-only EXR fixture: unit tests do not claim real Blender execution.
        raw = folder / "background.exr"
        raw.write_bytes(b"unit-test fixture, not an EXR")
        files[raw.relative_to(root / "raw").as_posix()] = sha256(raw)
        groups.append(
            dict(scene=spec, group=scene_group(config, spec), entries=entries, raw_files=files)
        )
    write_json(
        root / "raw/index.json",
        dict(
            schema_version=1,
            renderer="4.5.3",
            device="CPU",
            config_sha256=config_digest(config),
            groups=groups,
        ),
    )
    compile_pairs(root, config)
    return root


def rehash_manifest(root, rows, split="dev"):
    write_rows(root / split / "manifest.csv", rows)
    dataset = json.loads((root / "dataset.json").read_text())
    dataset["manifests"][split] = sha256(root / split / "manifest.csv")
    write_json(root / "dataset.json", dataset)


def test_roles_use_shadow_intervention_and_keep_uncertain_changes_invalid():
    policy = load_yaml(CONFIG_ROOT / "datasets/blender_translation.yaml")["roles"]
    values = evidence([0.7, 0.1, 0.1])
    labels, valid = derive_roles(values, 1, policy)
    assert labels[18, 18] == labels[18, 42] == 0
    assert labels[43, 15] == labels[43, 43] == 1
    assert labels[55, 55] == 2
    # An albedo change in both shadow-on/off is not a shadow consequence label.
    values["target"][55, 55] += 0.1
    values["target_no_shadow"][55, 55] += 0.1
    labels, valid = derive_roles(values, 1, policy)
    assert labels[55, 55] == 255 and valid[55, 55] == 0
    values["target"][52, 52] -= 0.01
    labels, valid = derive_roles(values, 1, policy)
    assert labels[52, 52] == 255
    identity, _ = derive_roles(evidence([0.7, 0.1, 0.1], 0), 0, policy)
    assert (identity == 2).all()
    with pytest.raises(ValueError, match="identity"):
        derive_roles(values, 0, policy)


def test_scene_pairing_strength_roles_and_background_validate(scene_data):
    result = validate_scene_dataset(scene_data)
    assert result["train"] == result["dev"] == 2
    assert result["scene_groups"] == 2 and result["scene_group_overlap"] == 0
    moved = [row for row in result["samples"] if row["strength"]]
    assert all(row["exposed_pixels"] > 0 and row["background_max_error"] == 0 for row in moved)
    assert all(row["role_pixels"][role] > 0 for row in moved for role in ("T", "D", "P"))
    with pytest.raises(FileExistsError):
        compile_pairs(scene_data, json.loads((scene_data / "dataset.json").read_text())["config"])


@pytest.mark.parametrize(
    "change", ["label", "valid", "strength", "source", "mask", "raw", "config"]
)
def test_scene_validation_rejects_semantic_and_byte_changes(scene_data, change):
    rows = read_rows(scene_data / "dev/manifest.csv")
    row = rows[1]
    if change in {"label", "valid", "source", "mask"}:
        key = "role" if change == "label" else change
        mode, color = ("RGB", "black") if key == "source" else ("L", 2)
        path = scene_data / row[f"{key}_path"]
        Image.new(mode, (64, 64), color).save(path)
        row[f"{key}_sha256"] = sha256(path)
        rehash_manifest(scene_data, rows)
    elif change == "strength":
        row["displacement"] = "1000"
        rehash_manifest(scene_data, rows)
    elif change == "raw":
        next((scene_data / "raw").glob("*/*.exr")).write_bytes(b"changed")
    else:
        path = scene_data / "scene_config.yaml"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        validate_scene_dataset(scene_data)


def test_background_completion_is_checked_separately_from_role_recomputation(scene_data):
    rows = read_rows(scene_data / "dev/manifest.csv")
    row = rows[1]
    path = scene_data / row["evidence_path"]
    with np.load(path) as file:
        arrays = dict(file)
    arrays["background"][:] = 0
    np.savez_compressed(path, **arrays)
    row["evidence_sha256"] = sha256(path)
    raw_path = scene_data / "raw/index.json"
    raw = json.loads(raw_path.read_text())
    raw["groups"][1]["entries"][1]["sha256"] = sha256(path)
    write_json(raw_path, raw)
    dataset_path = scene_data / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["raw_index_sha256"] = sha256(raw_path)
    write_json(dataset_path, dataset)
    rehash_manifest(scene_data, rows)
    with pytest.raises(ValueError, match="background completion"):
        validate_scene_dataset(scene_data)


@pytest.mark.parametrize("change", ["split", "duplicate_scene", "strength", "geometry", "device"])
def test_invalid_scene_configuration_fails_before_rendering(change):
    config = copy.deepcopy(load_yaml(CONFIG_ROOT / "datasets/blender_translation.yaml"))
    if change == "split":
        for scene in config["scenes"]:
            scene["split"] = "train"
    elif change == "duplicate_scene":
        scene = config["scenes"][0].copy()
        scene.update(id="duplicate_dev", split="dev")
        config["scenes"][2] = scene
    elif change == "strength":
        config["strengths"] = [0, float("nan")]
    elif change == "geometry":
        config["renderer"]["resolution"] = 65
    else:
        config["renderer"]["device"] = "CUDA"
    with pytest.raises(ValueError):
        validate_scene_config(config)


def test_renderer_failure_preserves_report_and_original_exception(tmp_path, monkeypatch):
    import subprocess

    error = subprocess.CalledProcessError(17, ["renderer"])

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("src.datasets.scene.subprocess.run", fail)
    output = tmp_path / "failed"
    with pytest.raises(subprocess.CalledProcessError) as raised:
        generate(CONFIG_ROOT / "datasets/blender_translation.yaml", output)
    assert raised.value is error
    report = json.loads((output / "generation.json").read_text())
    assert report["status"] == "failed" and "CalledProcessError" in report["traceback"]
    with pytest.raises(FileExistsError):
        generate(CONFIG_ROOT / "datasets/blender_translation.yaml", output)


def test_oracle_loading_never_reads_target_or_counterfactual_rgb(scene_data, monkeypatch):
    from src.datasets.manifest import load_edit_input
    from src.datasets.scene_manifest import load_oracle_condition, load_role_supervision

    row = read_rows(scene_data / "dev/manifest.csv")[1]
    supervision = load_role_supervision(scene_data, row)
    assert supervision.target.mode == "RGB" and supervision.role_labels.mode == "L"
    for path in (scene_data / "dev/target").glob("*.png"):
        path.unlink()
    for path in (scene_data / "raw").glob("*/*.npz"):
        path.unlink()
    original = Image.open

    def safe_open(path, *args, **kwargs):
        if "/target/" in str(path):
            raise AssertionError("Target read during oracle conditioning")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", safe_open)
    edit = load_edit_input(scene_data, row)
    control = load_oracle_condition(scene_data, row)
    assert not hasattr(edit, "role_labels") and not hasattr(edit, "target")
    assert control.kind == "oracle_role" and control.metadata["sample_id"] == row["id"]
    with pytest.raises(ValueError, match="ordinary inference"):
        control.require_inference_safe()
