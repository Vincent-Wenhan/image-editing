"""Local contact sheets and an HTML gallery; never upload dataset images."""

import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from src.conditioning.controls import ControlKind
from src.conditioning.geometry import geometry_rgb
from src.conditioning.oracle import role_rgb
from src.core.config import ProjectSettings
from src.core.io import read_rows
from src.core.paths import data_path
from src.datasets.role_controls import load_controls
from src.datasets.scene_manifest import load_role_supervision
from src.datasets.scene_validation import validate_scene_dataset
from src.datasets.validation import verify_images
from src.evaluation.provenance import verify_run


def control_image(controls, row, size: tuple[int, int]) -> tuple[str, Image.Image]:
    bundle = controls.bundle(row, size)
    if bundle.kind in {ControlKind.ORACLE_ROLE, ControlKind.ORACLE_GEOMETRY}:
        if bundle.kind == ControlKind.ORACLE_ROLE:
            return "role_input", role_rgb(bundle.maps["roles"])
        return "geometry_input", geometry_rgb(bundle.maps)
    return "control_input", bundle.maps["mask"].convert("RGB")


def verify_prediction_images(predictions: Path, record: dict) -> None:
    for item in record["results"]:
        path = predictions / f"{item['id']}.png"
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (item["width"], item["height"]):
                raise ValueError(f"Prediction mode/geometry differs from its run record: {path}")


def run(args):
    root = Path(args.data_root)
    if args.limit < 1:
        raise ValueError("Gallery limit must be positive")
    show_roles = getattr(args, "show_roles", False)
    if show_roles:
        validate_scene_dataset(root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new gallery directory: {output}")
    rows = read_rows(args.manifest)
    for row in rows:
        verify_images(root, row)
    predictions = Path(args.predictions) if args.predictions else None
    controls, results = None, {}
    if predictions:
        record = json.loads((predictions / "run.json").read_text())
        canonical = verify_run(root, args.manifest, record)
        verify_prediction_images(predictions, record)
        results = {row["id"]: row for row in record["results"]}
        rows = [canonical[row["id"]] for row in record["results"]]
        config = record["config"]
        if config.get("control", "none") != "none":
            controls = load_controls(Path(config["control_root"]), root, config["control"])
            if controls.metadata() != record.get("conditioning"):
                raise ValueError("Gallery control index SHA256 differs from inference")
    else:
        rows = rows[: args.limit]
    rows = rows[: args.limit]
    if not rows:
        raise ValueError("No gallery samples")
    output.mkdir(parents=True)
    cards = []
    for page, start in enumerate(range(0, len(rows), 6)):
        subset = rows[start : start + 6]
        columns = 3 + int(predictions is not None) + int(controls is not None) + 2 * show_roles
        sheet = Image.new("RGB", (224 * columns, len(subset) * 260), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(subset):
            pairs = [
                (key, data_path(root, row[f"{key}_path"])) for key in ("source", "target", "mask")
            ]
            if show_roles:
                supervision = load_role_supervision(root, row)
                directory = output / "role_truth"
                directory.mkdir(exist_ok=True)
                path = directory / f"{row['id']}.png"
                role_rgb(supervision.role_labels).save(path)
                pairs.extend(
                    [
                        ("T=red D=green P=blue ignore=black", path),
                        ("valid_region", data_path(root, row["valid_path"])),
                    ]
                )
            if controls:
                item = results[row["id"]]
                label, control = control_image(controls, row, (item["width"], item["height"]))
                path = output / "control_inputs" / f"{row['id']}.png"
                path.parent.mkdir(exist_ok=True)
                control.save(path)
                pairs[2] = ("annotation_mask", pairs[2][1])
                pairs.append((label, path))
            if predictions:
                pairs.append(("prediction", predictions / f"{row['id']}.png"))
            images = []
            for column, (label, path) in enumerate(pairs):
                with Image.open(path) as image:
                    thumb = ImageOps.contain(image.convert("RGB"), (220, 220))
                sheet.paste(thumb, (column * 224, index * 260 + 24))
                draw.text((column * 224 + 4, index * 260 + 4), label, fill="black")
                relative = html.escape(
                    os.path.relpath(path.resolve(), output.resolve()), quote=True
                )
                images.append(
                    f'<figure><img src="{relative}"><figcaption>{label}</figcaption></figure>'
                )
            draw.text((4, index * 260 + 244), row["id"], fill="black")
            cards.append(
                f"<article><h2>{html.escape(row['id'])}</h2>"
                f"<p>{html.escape(row['instruction'])}</p><div>{''.join(images)}</div></article>"
            )
        sheet.save(output / f"contact_{page:02d}.jpg")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Editing gallery</title>'
        "<style>body{font-family:system-ui;margin:2rem;background:#eee}article{background:white;"
        "padding:1rem;margin-bottom:1rem}div{display:flex;flex-wrap:wrap}figure{margin:4px}"
        "img{width:240px;max-width:100%}h2{font-size:1rem}</style>" + "\n".join(cards)
    )
    print(f"Gallery: {output / 'index.html'} ({len(rows)} samples)")


def configure(parser):
    settings = ProjectSettings.load()
    parser.add_argument("--data-root", default=settings.dataset["root"])
    parser.add_argument("--manifest", default=settings.manifest)
    parser.add_argument("--predictions")
    parser.add_argument("--output", default="outputs/data_gallery")
    parser.add_argument("--limit", type=int, default=settings.dataset["dev"])
    parser.add_argument(
        "--show-roles", action="store_true", help="Validate and display synthetic T/D/P"
    )
    parser.set_defaults(func=run)
