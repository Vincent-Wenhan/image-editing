"""Fail closed on missing, corrupt, mismatched or leaking paired-edit samples."""

from pathlib import Path

from PIL import Image

from image_editing.common import read_rows, sha256, write_json


def scan(root, split):
    rows = read_rows(root / split / "manifest.csv")
    identifiers, sessions, sources = set(), set(), set()
    coverage = []
    for row in rows:
        if row["id"] in identifiers or row["split"] != split or not row["source_session"]:
            raise ValueError(f"Invalid/duplicate sample: {row['id']}")
        identifiers.add(row["id"])
        sessions.add(row["source_session"])
        dimensions = (int(row["width"]), int(row["height"]))
        for key in ("source", "target", "mask"):
            path = (root / row[f"{key}_path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Path escapes data root: {path}")
            with Image.open(path) as image:
                image.load()
                expected_mode = "L" if key == "mask" else "RGB"
                if image.size != dimensions or image.mode != expected_mode:
                    raise ValueError(f"Image shape/mode mismatch: {path}")
                if key == "mask":
                    histogram = image.histogram()
                    coverage.append(sum(histogram[128:]) / (image.width * image.height))
            if sha256(path) != row[f"{key}_sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
        caption = root / split / "target" / f"{row['id']}.txt"
        if not row["instruction"].strip() or caption.read_text().strip() != row["instruction"]:
            raise ValueError(f"Caption mismatch: {caption}")
        sources.add(row["source_sha256"])
    return rows, sessions, sources, coverage


def validate(root):
    root = Path(root)
    train, tr_sessions, tr_sources, tr_coverage = scan(root, "train")
    dev, dv_sessions, dv_sources, dv_coverage = scan(root, "dev")
    if tr_sessions & dv_sessions or tr_sources & dv_sources:
        raise ValueError("Train/dev leakage: overlapping sessions or identical source images")
    return {
        "train": len(train),
        "dev": len(dev),
        "session_overlap": 0,
        "source_overlap": 0,
        "mask_edit_fraction": {
            "train_mean": sum(tr_coverage) / len(tr_coverage),
            "dev_mean": sum(dv_coverage) / len(dv_coverage),
        },
    }


def run(args):
    result = validate(Path(args.root))
    write_json(Path(args.root) / "validation.json", result)
    print(result)


def configure(parser):
    parser.add_argument("--root", default="data/magicbrush")
    parser.set_defaults(func=run)
