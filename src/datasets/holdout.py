"""Bind new research cohorts to actual files and keep Test out of model selection."""

import json
from pathlib import Path

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.scene_validation import validate_scene_dataset
from src.datasets.validation import dataset_splits


def identities(root: Path) -> tuple[set[str], set[str]]:
    groups, sources = set(), set()
    for split in dataset_splits(root):
        for row in read_rows(root / split / "manifest.csv"):
            if sha256(data_path(root, row["source_path"])) != row["source_sha256"]:
                raise ValueError("Historical/new source SHA256 changed")
            groups.add(row["group_id"])
            sources.add(row["source_sha256"])
    return groups, sources


def seal_cohort(root: Path, previous: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    validate_scene_dataset(root)
    if set(dataset_splits(root)) != {"train", "dev", "test"} or not previous:
        raise ValueError("Declare three new splits and historical cohorts")
    groups, sources = identities(root)
    history = []
    for old in previous:
        old_groups, old_sources = identities(old)
        if groups & old_groups or sources & old_sources:
            raise ValueError("New cohorts overlap historical scene groups or source SHA256")
        history.append(
            dict(
                root=str(old.resolve()),
                dataset_sha256=sha256(old / "dataset.json"),
                manifests={s: sha256(old / s / "manifest.csv") for s in dataset_splits(old)},
            )
        )
    result = dict(
        schema_version=1,
        purpose="dev_only_failure_analysis",
        test_status="sealed",
        root=str(root.resolve()),
        dataset_sha256=sha256(root / "dataset.json"),
        manifests={s: sha256(root / s / "manifest.csv") for s in dataset_splits(root)},
        historical=history,
        groups=sorted(groups),
        dev_ids=[r["id"] for r in read_rows(root / "dev/manifest.csv")],
    )
    write_json(output, result)
    return result


def require_dev(root: Path, seal: Path, manifest: Path | None = None) -> dict:
    record = json.loads(seal.read_text())
    if (
        record["schema_version"] != 1
        or record["purpose"] != "dev_only_failure_analysis"
        or record["test_status"] != "sealed"
        or sha256(root / "dataset.json") != record["dataset_sha256"]
        or any(sha256(root / s / "manifest.csv") != h for s, h in record["manifests"].items())
        or (manifest is not None and manifest.resolve() != (root / "dev/manifest.csv").resolve())
    ):
        raise ValueError("Cohort changed or attempted evaluation outside sealed Dev protocol")
    return record
