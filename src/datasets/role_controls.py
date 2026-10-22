"""Hashed hard-role references; both methods are ground-truth upper bounds.

Export validates counterfactual evidence. Consumption reads source/labels/valid
only; target and counterfactual RGB never enter the inference condition path.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.oracle import role_rgb
from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.scene_manifest import load_oracle_condition
from src.datasets.scene_validation import validate_scene_dataset
from src.datasets.validation import dataset_splits

ORACLE_KINDS = (ControlKind.ORACLE_ROLE, ControlKind.DYNAMIC_BINARY)


def condition_image(bundle: ControlBundle, kind: str) -> Image.Image:
    if kind == ControlKind.ORACLE_ROLE:
        return role_rgb(bundle.maps["roles"])
    if kind != ControlKind.DYNAMIC_BINARY:
        raise ValueError("Unsupported oracle encoding")
    labels = np.asarray(bundle.maps["roles"])
    return Image.fromarray(np.isin(labels, [0, 1]).astype(np.uint8) * 255).convert("RGB")


def export_role_controls(data_root: Path, output: Path, kind: str) -> Path:
    if kind not in ORACLE_KINDS:
        raise ValueError("Use a declared oracle upper-bound method")
    validate_scene_dataset(data_root)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    folder = "role_rgb" if kind == ControlKind.ORACLE_ROLE else "mask_rgb"
    index = dict(
        schema_version=2,
        kind=kind,
        purpose="oracle_upper_bound",
        order=["source", folder],
        splits={},
    )
    for split in dataset_splits(data_root):
        manifest = data_root / split / "manifest.csv"
        entries = {}
        for row in read_rows(manifest):
            oracle = load_oracle_condition(data_root, row)
            relative = f"{split}/{folder}/{row['id']}.png"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            condition_image(oracle, kind).save(path)
            entries[row["id"]] = dict(path=relative, sha256=sha256(path), **dict(oracle.metadata))
        index["splits"][split] = dict(manifest_sha256=sha256(manifest), entries=entries)
    write_json(output / "controls.json", index)
    return output / "controls.json"


class RoleControlDataset:
    """Exact-size prepared references for training and explicitly opted-in inference."""

    def __init__(self, root: Path, data_root: Path, kind: str) -> None:
        self.root, self.data_root = root.resolve(), data_root.resolve()
        self.kind = ControlKind(kind)
        self.folder = "role_rgb" if self.kind == ControlKind.ORACLE_ROLE else "mask_rgb"
        self.index_path = self.root / "controls.json"
        self.digest = sha256(self.index_path)
        self.index = json.loads(self.index_path.read_text())
        if (
            self.kind not in ORACLE_KINDS
            or self.index["schema_version"] != 2
            or self.index["kind"] != kind
            or self.index["purpose"] != "oracle_upper_bound"
            or self.index["order"] != ["source", self.folder]
            or set(self.index["splits"]) != set(dataset_splits(self.data_root))
        ):
            raise ValueError("Invalid oracle control protocol")
        self.rows = {}
        for split in dataset_splits(self.data_root):
            manifest = self.data_root / split / "manifest.csv"
            section = self.index["splits"][split]
            rows = read_rows(manifest)
            self.rows[split] = {row["id"]: row for row in rows}
            if (
                sha256(manifest) != section["manifest_sha256"]
                or len(self.rows[split]) != len(rows)
                or set(section["entries"]) != set(self.rows[split])
            ):
                raise ValueError("Oracle manifest SHA256 or pairing changed")
            for row in rows:
                self.image(row)

    def entry(self, row: dict) -> dict:
        if row != self.rows[row["split"]].get(row["id"]):
            raise ValueError("Oracle row differs from canonical manifest")
        entry = self.index["splits"][row["split"]]["entries"][row["id"]]
        oracle = load_oracle_condition(self.data_root, row)
        if (
            any(entry.get(k) != v for k, v in oracle.metadata.items())
            or entry["path"] != f"{row['split']}/{self.folder}/{row['id']}.png"
        ):
            raise ValueError("Oracle control provenance changed")
        return entry

    def image(self, row: dict) -> Image.Image:
        entry = self.entry(row)
        path = data_path(self.root, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Oracle reference SHA256 changed")
        expected = condition_image(load_oracle_condition(self.data_root, row), self.kind)
        with Image.open(path) as image:
            if (
                image.mode != "RGB"
                or image.size != expected.size
                or not np.array_equal(image, expected)
            ):
                raise ValueError("Oracle reference differs from declared label encoding")
            return image.copy()

    def bundle(self, row: dict, size: tuple[int, int]) -> ControlBundle:
        if size != (int(row["width"]), int(row["height"])):
            raise ValueError("Oracle references cannot be implicitly resized")
        self.image(row)
        oracle = load_oracle_condition(self.data_root, row)
        metadata = dict(oracle.metadata, index_sha256=self.digest)
        if self.kind == ControlKind.ORACLE_ROLE:
            return ControlBundle(self.kind, oracle.maps, metadata)
        return ControlBundle(
            self.kind,
            {"mask": self.image(row).convert("L")},
            dict(metadata, image_order="source,mask", role_encoding="oracle_union_td_binary_v1"),
        )

    def metadata(self) -> dict:
        return dict(
            root=str(self.root),
            data_root=str(self.data_root),
            index_sha256=self.digest,
            kind=self.kind.value,
            order=["source", self.folder],
            purpose="oracle_upper_bound",
        )


def load_controls(root: Path, data_root: Path, kind: str):
    from src.datasets.controls import ControlDataset

    if kind in ORACLE_KINDS:
        return RoleControlDataset(root, data_root, kind)
    return ControlDataset(root, data_root, kind)
