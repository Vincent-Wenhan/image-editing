"""Copy only validated paired assets into a new experiment, excluding trainer caches."""

import shutil
from pathlib import Path

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.validation import validate


def snapshot_pairs(source: Path, destination: Path) -> dict:
    validation = validate(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    hashes = {}
    for split in ("train", "dev"):
        manifest = source / split / "manifest.csv"
        for row in read_rows(manifest):
            for key in ("source", "target", "mask"):
                relative = row[f"{key}_path"]
                path = data_path(destination, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_path(source, relative), path)
            caption = f"{split}/target/{row['id']}.txt"
            shutil.copy2(data_path(source, caption), data_path(destination, caption))
        shutil.copy2(manifest, destination / split / "manifest.csv")
        hashes[split] = sha256(manifest)
    if (source / "dataset.json").exists():
        shutil.copy2(source / "dataset.json", destination / "dataset.json")
    result = dict(source=str(source.resolve()), manifests=hashes, validation=validate(destination))
    if validation != result["validation"]:
        raise ValueError("Dataset snapshot differs from original")
    write_json(destination / "snapshot.json", result)
    return result
