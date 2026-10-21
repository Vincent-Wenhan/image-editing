"""Copy only validated paired assets into a new experiment, excluding trainer caches."""

import json
import shutil
from pathlib import Path

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.validation import dataset_splits, validate


def snapshot_pairs(source: Path, destination: Path) -> dict:
    validation = validate(source)
    synthetic = (source / "scene_config.yaml").is_file()
    if synthetic:
        from src.datasets.scene_validation import validate_scene_dataset

        validate_scene_dataset(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    hashes = {}
    for split in dataset_splits(source):
        manifest = source / split / "manifest.csv"
        for row in read_rows(manifest):
            for key in (
                ("source", "target", "mask", "role", "valid")
                if synthetic
                else ("source", "target", "mask")
            ):
                relative = row[f"{key}_path"]
                path = data_path(destination, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_path(source, relative), path)
            if synthetic:
                relative = row["evidence_path"]
                path = data_path(destination, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_path(source, relative), path)
            caption = f"{split}/target/{row['id']}.txt"
            shutil.copy2(data_path(source, caption), data_path(destination, caption))
        shutil.copy2(manifest, destination / split / "manifest.csv")
        hashes[split] = sha256(manifest)
    if (source / "dataset.json").exists():
        shutil.copy2(source / "dataset.json", destination / "dataset.json")
    if synthetic:
        (destination / "raw").mkdir(exist_ok=True)
        shutil.copy2(source / "raw/index.json", destination / "raw/index.json")
        for group in json.loads((source / "raw/index.json").read_text())["groups"]:
            for relative in group["raw_files"]:
                path = data_path(destination / "raw", relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_path(source / "raw", relative), path)
        for relative in ("scene_config.yaml", "generation.json"):
            if (source / relative).exists():
                shutil.copy2(source / relative, destination / relative)
        validate_scene_dataset(destination)
    result = dict(source=str(source.resolve()), manifests=hashes, validation=validate(destination))
    if validation != result["validation"]:
        raise ValueError("Dataset snapshot differs from original")
    write_json(destination / "snapshot.json", result)
    return result
