"""Export a deterministic first-turn subset without changing source geometry."""

import io
import json
import re
import shutil
from pathlib import Path

from PIL import Image

from image_editing.common import now, sha256, write_json, write_rows


def as_image(value):
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        source = io.BytesIO(value["bytes"]) if value.get("bytes") else value["path"]
        with Image.open(source) as image:
            return image.copy()
    raise TypeError(f"Unsupported image value: {type(value)}")


def export_rows(rows, root, split, count):
    if split not in {"train", "dev"} or count < 1:
        raise ValueError("Use train/dev and a positive count")
    root = Path(root)
    destination = root / split
    temporary = root / f".{split}.incomplete"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Existing export: {destination} or {temporary}")
    for folder in ("reference", "target", "mask"):
        (temporary / folder).mkdir(parents=True, exist_ok=True)
    manifest, sessions = [], set()
    try:
        for row in rows:
            session, turn = str(row["img_id"]), int(row["turn_index"])
            if turn != 1 or session in sessions:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
                raise ValueError(f"Invalid session ID: {session}")
            instruction = row["instruction"].strip()
            if not instruction:
                raise ValueError(f"Empty instruction: {session}")
            source, target, mask = [
                as_image(row[k]) for k in ("source_img", "target_img", "mask_img")
            ]
            if not source.size == target.size == mask.size:
                raise ValueError(f"Unaligned images: {session}")
            # The official MagicBrush card defines WHITE as the edit region.
            # Keep native geometry and record the source mode to audit mask semantics.
            stem = f"{split}_{session}_t{turn}"
            record = dict(
                id=stem,
                split=split,
                source_session=session,
                turn_index=turn,
                instruction=instruction,
                width=source.width,
                height=source.height,
                mask_original_mode=mask.mode,
            )
            for folder, key, image in [
                ("reference", "source", source.convert("RGB")),
                ("target", "target", target.convert("RGB")),
                ("mask", "mask", mask.convert("L")),
            ]:
                path = temporary / folder / f"{stem}.png"
                image.save(path)
                record[f"{key}_path"] = f"{split}/{folder}/{stem}.png"
                record[f"{key}_sha256"] = sha256(path)
            (temporary / "target" / f"{stem}.txt").write_text(instruction + "\n")
            manifest.append(record)
            sessions.add(session)
            if len(manifest) == count:
                break
        if len(manifest) != count:
            raise ValueError(f"Only found {len(manifest)}/{count} first-turn samples in {split}")
        write_rows(temporary / "manifest.csv", manifest)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"Exported {count} {split} samples -> {destination}", flush=True)
    return manifest


def run(args):
    from datasets import Image as DatasetImage
    from datasets import load_dataset

    lock = json.loads(Path(args.lock).read_text())["dataset"]
    root = Path(args.root)
    metadata = dict(
        lock,
        created=now(),
        selection="first turn; fixed upstream order",
        requested={"train": args.train, "dev": args.dev},
    )
    for split, count in (("train", args.train), ("dev", args.dev)):
        if count <= 0:
            raise ValueError("Both train and dev counts must be positive")
        rows = load_dataset(
            lock["repo_id"],
            revision=lock["revision"],
            split=split,
            data_files={split: f"data/{split}-*.parquet"},
            streaming=True,
        )
        for column in ("source_img", "target_img", "mask_img"):
            rows = rows.cast_column(column, DatasetImage(decode=False))
        export_rows(rows, root, split, count)
    write_json(root / "dataset.json", metadata)
    from image_editing.validation import validate

    write_json(root / "validation.json", validate(root))


def configure(parser):
    parser.add_argument("--root", default="data/magicbrush")
    parser.add_argument("--lock", default="configs/assets.lock.json")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--dev", type=int, default=30)
    parser.set_defaults(func=run)
