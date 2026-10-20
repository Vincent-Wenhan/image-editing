"""Small shared IO and provenance helpers."""

import csv
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def write_rows(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def command(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def versions():
    packages = ("torch", "diffusers", "transformers", "peft", "datasets", "huggingface-hub")
    return {name: importlib.metadata.version(name) for name in packages}


def sample_seed(seed, sample_id):
    digest = hashlib.sha256(sample_id.encode()).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)
