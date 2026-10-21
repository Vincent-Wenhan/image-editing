"""Compile Blender evidence into ordinary pairs plus separate role supervision."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.conditioning.role_truth import derive_roles
from src.core.io import sha256, write_json, write_rows
from src.core.metadata import config_digest
from src.core.paths import data_path
from src.datasets.scene_config import validate_scene_config


def rgb_image(linear: np.ndarray) -> Image.Image:
    """Explicit Standard/sRGB display transform; role computation stays scene-linear."""
    values = np.clip(linear.astype(np.float64), 0, 1)
    srgb = np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)
    return Image.fromarray(np.rint(srgb * 255).astype(np.uint8))


def instruction(strength: float, displacement: float) -> str:
    if strength == 0:
        return "Keep the scene unchanged."
    return (
        f"Move the object to the right by {displacement:.3f} scene units "
        f"(strength {strength:.3f}); update its shadow and reveal the background."
    )


def compile_pairs(root: Path, config: dict) -> dict:
    validate_scene_config(config)
    raw_path = root / "raw/index.json"
    raw = json.loads(raw_path.read_text())
    if raw["config_sha256"] != config_digest(config):
        raise ValueError("Render evidence uses a different scene configuration")
    rows = {
        split: []
        for split in ("train", "dev", "test")
        if any(s["split"] == split for s in config["scenes"])
    }
    for group in raw["groups"]:
        scene = group["scene"]
        split = scene["split"]
        for index, entry in enumerate(group["entries"]):
            identifier = f"{scene['id']}_s{index:03d}"
            path = data_path(root / "raw", entry["evidence_path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError("Render evidence SHA256 changed")
            with np.load(path, allow_pickle=False) as file:
                evidence = dict(file)
            labels, valid = derive_roles(evidence, entry["strength"], config["roles"])
            mask = Image.fromarray((evidence["source_occupancy"] > 0.5).astype(np.uint8) * 255)
            source, target = rgb_image(evidence["source"]), rgb_image(evidence["target"])
            record = dict(
                id=identifier,
                split=split,
                source_session=group["group"],
                turn_index=1,
                scene_id=scene["id"],
                group_id=group["group"],
                strength=entry["strength"],
                displacement=entry["displacement"],
                object_width=scene["object_width"],
                instruction=instruction(entry["strength"], entry["displacement"]),
                width=source.width,
                height=source.height,
                mask_encoding="source_object_occupancy",
                evidence_path=f"raw/{entry['evidence_path']}",
                evidence_sha256=entry["sha256"],
            )
            for key, folder, image in (
                ("source", "reference", source),
                ("target", "target", target),
                ("mask", "mask", mask),
                ("role", "roles", Image.fromarray(labels)),
                ("valid", "valid", Image.fromarray(valid)),
            ):
                relative = f"{split}/{folder}/{identifier}.png"
                destination = root / relative
                if destination.exists():
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination)
                record[f"{key}_path"], record[f"{key}_sha256"] = relative, sha256(destination)
            (root / split / "target" / f"{identifier}.txt").write_text(record["instruction"] + "\n")
            rows[split].append(record)
    for split, records in rows.items():
        write_rows(root / split / "manifest.csv", records)
    dataset = dict(
        schema_version=1,
        kind="blender_translation_v1",
        status="complete",
        config=config,
        config_sha256=config_digest(config),
        raw_index_sha256=sha256(raw_path),
        config_file_sha256=sha256(root / "scene_config.yaml"),
        manifests={split: sha256(root / split / "manifest.csv") for split in rows},
        scope="direct diffuse light, hard point-light shadows, procedural scene-group splits",
    )
    write_json(root / "dataset.json", dataset)
    return dataset
