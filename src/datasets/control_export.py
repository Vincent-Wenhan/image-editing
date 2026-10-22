"""Materialize versioned joint conditions without changing original manifests."""

import argparse
import hashlib
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from src.conditioning.controls import ControlKind
from src.conditioning.static import MASK_KINDS, MaskSpec, mask_rgb, transform_mask
from src.core.config import ProjectSettings, load_yaml
from src.core.io import read_rows, sha256, write_json
from src.core.paths import CONFIG_ROOT, data_path
from src.datasets.validation import dataset_splits, validate


def donors(rows: list[dict], spec: MaskSpec) -> dict[str, dict]:
    """Stable full-split derangement within geometry groups, independent of limit."""
    if spec.kind != ControlKind.SHUFFLED_MASK:
        return {row["id"]: row for row in rows}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["width"], row["height"])].append(row)
    result = {}
    for group in groups.values():
        if len(group) < 2:
            raise ValueError("Shuffled Mask requires at least two same-split/same-size samples")
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(f"{spec.shuffle_seed}:{row['id']}".encode()).digest(),
        )
        result.update((row["id"], ordered[(i + 1) % len(ordered)]) for i, row in enumerate(ordered))
    return result


def export_controls(data_root: Path, output: Path, spec: MaskSpec) -> Path:
    validate(data_root)
    if output.exists():
        raise FileExistsError(f"Use a new control directory: {output}")
    output.mkdir(parents=True)
    index = dict(schema_version=1, spec=asdict(spec), order=["source", "mask"], splits={})
    try:
        for split in dataset_splits(data_root):
            manifest = data_root / split / "manifest.csv"
            rows = read_rows(manifest)
            mapping = donors(rows, spec)
            entries = {}
            for row in rows:
                donor = mapping[row["id"]]
                relative = f"{split}/mask_rgb/{row['id']}.png"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(data_path(data_root, donor["mask_path"])) as mask:
                    control = mask_rgb(transform_mask(mask, spec))
                control.save(path)
                entries[row["id"]] = dict(
                    path=relative,
                    sha256=sha256(path),
                    source_sha256=row["source_sha256"],
                    donor_id=donor["id"],
                    donor_mask_sha256=donor["mask_sha256"],
                    width=control.width,
                    height=control.height,
                )
            index["splits"][split] = dict(manifest_sha256=sha256(manifest), entries=entries)
        write_json(output / "controls.json", index)
    except BaseException:
        shutil.rmtree(output)
        raise
    return output / "controls.json"


def run(args: argparse.Namespace) -> None:
    from src.datasets.role_controls import ORACLE_KINDS, export_role_controls

    if args.kind in ORACLE_KINDS:
        print(export_role_controls(Path(args.data_root), Path(args.output), args.kind))
        return
    spec = MaskSpec(
        ControlKind(args.kind),
        args.dilation_radius if args.kind == ControlKind.DILATED_MASK else 0,
        args.shuffle_seed,
    )
    print(export_controls(Path(args.data_root), Path(args.output), spec))


def configure(parser: argparse.ArgumentParser) -> None:
    from src.datasets.role_controls import ORACLE_KINDS

    defaults = load_yaml(CONFIG_ROOT / "experiments/stage1.yaml")["controls"]
    parser.add_argument("--data-root", default=ProjectSettings.load().dataset["root"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=[*MASK_KINDS, *ORACLE_KINDS], required=True)
    parser.add_argument("--dilation-radius", type=int, default=defaults["dilation_radius"])
    parser.add_argument("--shuffle-seed", type=int, default=defaults["shuffle_seed"])
    parser.set_defaults(func=run)
