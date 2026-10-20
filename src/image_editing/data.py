"""Export a deterministic first-turn subset without changing source geometry."""

import io
import json
import re
import shutil
from pathlib import Path

from PIL import Image, ImageChops

from image_editing.common import now, sha256, write_json, write_rows


def as_image(value):
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        source = io.BytesIO(value["bytes"]) if value.get("bytes") else value["path"]
        with Image.open(source) as image:
            return image.copy()
    raise TypeError(f"Unsupported image value: {type(value)}")


def editable_mask(mask):
    if "A" in mask.getbands():
        return mask.getchannel("A").point(lambda value: 255 if value < 128 else 0), "inverse_alpha"
    red, green, blue = mask.convert("RGB").split()
    if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
        raise ValueError("Colored mask without alpha has ambiguous edit semantics")
    return mask.convert("L").point(lambda value: 255 if value >= 128 else 0), "white_is_edit"


def align_images(source, target, mask, side=None):
    ratio = source.width / source.height
    if any(abs(image.width / image.height - ratio) > 0.001 for image in (target, mask)):
        raise ValueError("Different image aspect ratios; cannot safely align this pair")
    size = source.size
    if side:
        scale = side / max(size)
        size = tuple(max(16, round(value * scale / 16) * 16) for value in size)
    source = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    target = target.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    mask, encoding = editable_mask(mask)
    return source, target, mask.resize(size, Image.Resampling.NEAREST), encoding


def export_rows(rows, root, split, count, side=None):
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
            original_sizes = {
                f"{key}_original_size": f"{image.width}x{image.height}"
                for key, image in (("source", source), ("target", target), ("mask", mask))
            }
            original_mode = mask.mode
            source, target, mask, encoding = align_images(source, target, mask, side)
            stem = f"{split}_{session}_t{turn}"
            record = dict(
                id=stem,
                split=split,
                source_session=session,
                turn_index=turn,
                instruction=instruction,
                width=source.width,
                height=source.height,
                mask_original_mode=original_mode,
                mask_encoding=encoding,
                **original_sizes,
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


def streaming_rows(lock, split):
    from datasets import Image as DatasetImage
    from datasets import load_dataset

    rows = load_dataset(
        lock["repo_id"],
        revision=lock["revision"],
        split=split,
        data_files={split: f"data/{split}-*.parquet"},
        streaming=True,
    )
    for column in ("source_img", "target_img", "mask_img"):
        rows = rows.cast_column(column, DatasetImage(decode=False))
    yield from rows


def parquet_rows(info, split, cache):
    """Download only the successive shards needed to satisfy the subset count."""
    import pyarrow.parquet as parquet

    from image_editing.transport import download_file

    files = sorted(
        (item for item in info["siblings"] if item["rfilename"].startswith(f"data/{split}-")),
        key=lambda item: item["rfilename"],
    )
    for item in files:
        path = download_file(info["id"], info["sha"], item, cache, repo_type="dataset")
        with parquet.ParquetFile(path) as shard:
            for batch in shard.iter_batches(batch_size=16):
                yield from batch.to_pylist()


def prepare(args):
    from concurrent.futures import ThreadPoolExecutor

    from image_editing.transport import repo_info
    from image_editing.validation import scan, validate

    lock = json.loads(Path(args.lock).read_text())["dataset"]
    root = Path(args.root)
    state = dict(
        lock,
        selection="first turn; fixed upstream order",
        requested={"train": args.train, "dev": args.dev},
        side=args.side,
        mask_policy="inverse_alpha_or_white_binary",
    )
    state_path = root / "dataset.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if any(previous[k] != state[k] for k in ("repo_id", "revision", "selection", "requested")):
            raise ValueError("Existing export provenance differs; use a new root")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Existing data has no provenance; use a new root")
    if args.side < 16 or min(args.train, args.dev) < 1:
        raise ValueError("Positive sample counts and side >=16 required")
    if state_path.exists() and any((root / x / "manifest.csv").exists() for x in ("train", "dev")):
        if any(previous.get(k) != state[k] for k in ("side", "mask_policy")):
            raise ValueError("Existing export uses different preprocessing")
    state.update(created=now(), status="running", transport=args.transport)
    write_json(state_path, state)
    info = (
        repo_info(lock["repo_id"], lock["revision"], repo_type="dataset")
        if args.transport == "shards"
        else None
    )

    def export(split, count):
        if (root / split / "manifest.csv").exists():
            rows, *_ = scan(root, split)
            if len(rows) != count:
                raise ValueError(f"Unexpected count for {split}")
            return
        cache = root.parent / "raw" / root.name
        rows = parquet_rows(info, split, cache) if info else streaming_rows(lock, split)
        export_rows(rows, root, split, count, args.side)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(export, split, count)
            for split, count in (("train", args.train), ("dev", args.dev))
        ]
        for future in futures:
            future.result()
    write_json(root / "validation.json", validate(root))
    write_json(state_path, state | {"status": "complete"})


def configure(parser):
    parser.add_argument("--root", default="data/magicbrush")
    parser.add_argument("--lock", default="configs/assets.lock.json")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--dev", type=int, default=30)
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--transport", choices=["streaming", "shards"], default="streaming")
    parser.set_defaults(func=prepare)
