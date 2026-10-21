"""Read checked mask conditions using source/mask only, never target supervision."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.static import MaskSpec, mask_rgb, transform_mask
from src.core.io import read_rows, sha256
from src.core.paths import data_path
from src.datasets.control_export import donors


class ControlDataset:
    """Small immutable-on-disk condition snapshot shared by training and inference."""

    def __init__(self, root: Path, data_root: Path, kind: str) -> None:
        self.root, self.data_root = root.resolve(), data_root.resolve()
        self.index_path = self.root / "controls.json"
        self.digest = sha256(self.index_path)
        self.index = json.loads(self.index_path.read_text())
        if self.index["schema_version"] != 1 or self.index["order"] != ["source", "mask"]:
            raise ValueError("Unsupported control schema or image order")
        self.spec = MaskSpec(**self.index["spec"])
        if self.spec.kind != ControlKind(kind):
            raise ValueError("Control method differs from requested configuration")
        self.rows = {}
        for split in ("train", "dev"):
            manifest = self.data_root / split / "manifest.csv"
            section = self.index["splits"][split]
            if sha256(manifest) != section["manifest_sha256"]:
                raise ValueError("Control manifest SHA256 changed")
            rows = read_rows(manifest)
            mapping = donors(rows, self.spec)
            self.rows[split] = {row["id"]: row for row in rows}
            if len(self.rows[split]) != len(rows) or set(section["entries"]) != set(mapping):
                raise ValueError("Control sample pairing differs from manifest")
            for row in rows:
                entry, donor = section["entries"][row["id"]], mapping[row["id"]]
                if (
                    row["split"] != split
                    or entry["donor_id"] != donor["id"]
                    or entry["donor_mask_sha256"] != donor["mask_sha256"]
                    or entry["source_sha256"] != row["source_sha256"]
                    or (entry["width"], entry["height"]) != (int(row["width"]), int(row["height"]))
                    or entry["path"] != f"{split}/mask_rgb/{row['id']}.png"
                ):
                    raise ValueError("Invalid control provenance or pairing")
                for key, sample in (("source", row), ("mask", donor)):
                    if (
                        sha256(data_path(self.data_root, sample[f"{key}_path"]))
                        != sample[f"{key}_sha256"]
                    ):
                        raise ValueError("Control input checksum mismatch")
                image = self.image(row)
                with Image.open(data_path(self.data_root, donor["mask_path"])) as mask:
                    expected = mask_rgb(transform_mask(mask, self.spec))
                if not np.array_equal(image, expected):
                    raise ValueError("Control image differs from declared mask transform")

    def entry(self, row: dict) -> dict:
        canonical = self.rows[row["split"]][row["id"]]
        if row != canonical:
            raise ValueError("Inference row differs from control manifest")
        return self.index["splits"][row["split"]]["entries"][row["id"]]

    def image(self, row: dict) -> Image.Image:
        entry = self.entry(row)
        path = data_path(self.root, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Control image SHA256 changed")
        with Image.open(path) as image:
            image.load()
            values = np.asarray(image)
            if (
                image.mode != "RGB"
                or image.size != (entry["width"], entry["height"])
                or not np.isin(values, [0, 255]).all()
                or not (values == values[:, :, :1]).all()
            ):
                raise ValueError("Mask reference must be binary RGB with identical channels")
            return image.copy()

    def bundle(self, row: dict, size: tuple[int, int]) -> ControlBundle:
        image = self.image(row).convert("L").resize(size, Image.Resampling.NEAREST)
        entry = self.entry(row)
        return ControlBundle(
            self.spec.kind,
            {"mask": image},
            dict(
                index_sha256=self.digest,
                mask_sha256=entry["sha256"],
                donor_id=entry["donor_id"],
                donor_mask_sha256=entry["donor_mask_sha256"],
                dilation_radius=self.spec.dilation_radius,
                shuffle_seed=self.spec.shuffle_seed,
                image_order="source,mask",
            ),
        )

    def metadata(self) -> dict:
        return dict(
            root=str(self.root),
            data_root=str(self.data_root),
            index_sha256=self.digest,
            kind=self.spec.kind.value,
            order=["source", "mask"],
        )
