"""Immutable small Train views; canonical data and sealed splits remain unchanged."""

import json
import shutil
from pathlib import Path

from src.core.io import read_rows, sha256, write_json
from src.datasets.holdout import require_dev
from src.datasets.validation import verify_images


def selected_rows(root: Path, groups: int) -> list[dict]:
    rows = read_rows(root / "train/manifest.csv")
    chosen = sorted({r["group_id"] for r in rows})[:groups]
    result = [r for r in rows if r["group_id"] in chosen]
    if (
        len(chosen) != groups
        or len(result) != groups * 3
        or any(
            sorted(float(r["strength"]) for r in result if r["group_id"] == g) != [0, 0.5, 1]
            or len({r["source_sha256"] for r in result if r["group_id"] == g}) != 1
            for g in chosen
        )
    ):
        raise ValueError("Need complete Train scene groups with shared source and three strengths")
    return result


def create_view(root: Path, cohort: Path, output: Path, prompt: str, groups: int) -> dict:
    require_dev(root, cohort)
    if output.exists():
        raise FileExistsError(output)
    if not prompt.strip():
        raise ValueError("An explicit common prompt is required")
    rows = selected_rows(root, groups)
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Research training views cannot contain Dev or Test")
        verify_images(root, row)
    for kind in ["target", "reference"]:
        (output / kind).mkdir(parents=True)
    for row in rows:
        for kind, source in [("target", "target"), ("reference", "source")]:
            shutil.copy2(root / row[f"{source}_path"], output / kind / f"{row['id']}.png")
        (output / "target" / f"{row['id']}.txt").write_text(prompt, encoding="utf-8")
    record = dict(
        schema_version=1,
        purpose="controlled_train_overfit",
        data_root=str(root.resolve()),
        cohort=str(cohort.resolve()),
        cohort_sha256=sha256(cohort),
        dataset_sha256=sha256(root / "dataset.json"),
        train_manifest_sha256=sha256(root / "train/manifest.csv"),
        groups=groups,
        ids=[r["id"] for r in rows],
        prompt=prompt,
        captions={r["id"]: sha256(output / "target" / f"{r['id']}.txt") for r in rows},
    )
    write_json(output / "view.json", record)
    verify_view(output / "view.json")
    return record


def verify_view(path: Path, *, root: Path | None = None) -> dict:
    record = json.loads(path.read_text())
    actual_root = Path(record["data_root"])
    if root is not None and actual_root.resolve() != root.resolve():
        raise ValueError("Research view canonical data root changed")
    if (
        record["schema_version"] != 1
        or record["purpose"] != "controlled_train_overfit"
        or not record["prompt"].strip()
        or sha256(record["cohort"]) != record["cohort_sha256"]
        or sha256(actual_root / "dataset.json") != record["dataset_sha256"]
        or sha256(actual_root / "train/manifest.csv") != record["train_manifest_sha256"]
    ):
        raise ValueError("Research view data/cohort provenance changed")
    require_dev(actual_root, Path(record["cohort"]))
    rows = selected_rows(actual_root, record["groups"])
    ids = [r["id"] for r in rows]
    if record["ids"] != ids or set(record["captions"]) != set(ids):
        raise ValueError("Research view selection differs from the preregistered Train groups")
    for kind in ["target", "reference"]:
        if {p.stem for p in (path.parent / kind).glob("*.png")} != set(ids):
            raise ValueError("Research view contains missing or extra image examples")
    for row in rows:
        verify_images(actual_root, row)
        for kind, source in [("target", "target"), ("reference", "source")]:
            if sha256(path.parent / kind / f"{row['id']}.png") != row[f"{source}_sha256"]:
                raise ValueError("Research view source/target pairing or actual SHA256 changed")
        caption = path.parent / "target" / f"{row['id']}.txt"
        if (
            caption.read_text() != record["prompt"]
            or sha256(caption) != record["captions"][row["id"]]
        ):
            raise ValueError("Research view common caption changed")
    return record
