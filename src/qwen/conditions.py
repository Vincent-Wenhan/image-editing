"""Versioned, target-free control snapshots for the Qwen manifest bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.core.io import sha256
from src.core.paths import data_path
from src.datasets.validation import dataset_splits


@dataclass(frozen=True)
class ConditionEntry:
    path: str
    sha256: str


@dataclass(frozen=True)
class ConditionManifest:
    kind: str
    root: Path | None
    index_sha256: str | None
    entries: dict[str, ConditionEntry]

    def record(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "root": str(self.root) if self.root is not None else None,
            "index_sha256": self.index_sha256,
        }


def condition_path(data_root: str | Path, path: str) -> Path:
    """Resolve a JSON condition path, allowing only declared absolute snapshots."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return data_path(data_root, path)


def canonical_conditions(rows: list[dict[str, str]]) -> ConditionManifest:
    return ConditionManifest(
        kind="canonical_mask",
        root=None,
        index_sha256=None,
        entries={
            row["id"]: ConditionEntry(row["mask_path"], row["mask_sha256"])
            for row in rows
        },
    )


def load_conditions(
    data_root: str | Path,
    split: str,
    rows: list[dict[str, str]],
    control_root: str | Path | None = None,
) -> ConditionManifest:
    """Load and verify a control snapshot without opening target images."""
    data_root = Path(data_root).resolve()
    if control_root is None:
        return canonical_conditions(rows)
    root = Path(control_root).resolve()
    index_path = root / "controls.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing Qwen control index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    schema = index.get("schema_version")
    if schema not in {1, 2} or not isinstance(index.get("splits"), dict):
        raise ValueError("Unsupported Qwen control snapshot")
    if set(index["splits"]) != set(dataset_splits(data_root)):
        raise ValueError("Qwen control splits differ from the dataset splits")
    section = index["splits"].get(split)
    if not isinstance(section, dict):
        raise ValueError(f"Qwen control snapshot has no {split} split")
    if section.get("manifest_sha256") != sha256(data_root / split / "manifest.csv"):
        raise ValueError("Qwen control manifest SHA256 changed")
    entries = section.get("entries")
    if not isinstance(entries, dict) or set(entries) != {row["id"] for row in rows}:
        raise ValueError("Qwen control sample pairing changed")
    if schema == 1:
        spec = index.get("spec")
        if not isinstance(spec, dict) or index.get("order") != ["source", "mask"]:
            raise ValueError("Invalid mask control snapshot")
        kind = str(spec.get("kind"))
    else:
        if index.get("purpose") != "oracle_upper_bound":
            raise ValueError("Invalid oracle control snapshot")
        kind = str(index.get("kind"))
    if not kind or kind == "None":
        raise ValueError("Qwen control kind is missing")
    result = {}
    for row in rows:
        entry = entries[row["id"]]
        relative = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"Invalid Qwen control entry: {row['id']}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Qwen control path escapes snapshot: {row['id']}")
        if path == data_path(data_root, row["target_path"]):
            raise ValueError(f"Qwen control points to target: {row['id']}")
        if sha256(path) != digest:
            raise ValueError(f"Qwen control checksum mismatch: {row['id']}")
        if entry.get("source_sha256") != row["source_sha256"]:
            raise ValueError(f"Qwen control source pairing changed: {row['id']}")
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"L", "RGB"} or image.size != (
                int(row["width"]),
                int(row["height"]),
            ):
                raise ValueError(f"Qwen control shape/mode mismatch: {row['id']}")
        json_path = (
            path.relative_to(data_root).as_posix()
            if path.is_relative_to(data_root)
            else str(path)
        )
        result[row["id"]] = ConditionEntry(json_path, digest)
    return ConditionManifest(kind, root, sha256(index_path), result)


def condition_from_record(record: dict) -> ConditionManifest | None:
    """Recover a declared condition root; old manifests default to canonical masks."""
    declaration = record.get("condition")
    if declaration is None:
        return None
    if not isinstance(declaration, dict) or declaration.get("kind") == "canonical_mask":
        if declaration != canonical_conditions([]).record():
            raise ValueError("Invalid Qwen condition declaration")
        return None
    root = declaration.get("root")
    index_sha = declaration.get("index_sha256")
    if not isinstance(root, str) or not isinstance(index_sha, str):
        raise ValueError("Invalid Qwen condition declaration")
    return ConditionManifest(str(declaration["kind"]), Path(root).resolve(), index_sha, {})
