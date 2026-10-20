"""Atomic deterministic first-turn export with the existing manifest schema."""

import re
import shutil
from pathlib import Path

from src.core.io import sha256, write_rows
from src.datasets.images import align_images, as_image


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
