"""Local contact sheets and an HTML gallery; never upload dataset images."""

import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from src.core.config import ProjectSettings
from src.core.io import read_rows
from src.datasets.controls import ControlDataset


def run(args):
    root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)[: args.limit]
    predictions = Path(args.predictions) if args.predictions else None
    controls, results = None, {}
    if predictions:
        record = json.loads((predictions / "run.json").read_text())
        results = {row["id"]: row for row in record["results"]}
        rows = [row for row in rows if row["id"] in results]
        config = record.get("config", {})
        if config.get("control", "none") != "none":
            controls = ControlDataset(Path(config["control_root"]), root, config["control"])
            if controls.digest != record.get("conditioning", {}).get("index_sha256"):
                raise ValueError("Gallery control index SHA256 differs from inference")
    if not rows:
        raise ValueError("No gallery samples")
    cards = []
    for page, start in enumerate(range(0, len(rows), 6)):
        subset = rows[start : start + 6]
        columns = 3 + int(predictions is not None) + int(controls is not None)
        sheet = Image.new("RGB", (224 * columns, len(subset) * 260), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(subset):
            pairs = [(key, root / row[f"{key}_path"]) for key in ("source", "target", "mask")]
            if controls:
                item = results[row["id"]]
                bundle = controls.bundle(row, (item["width"], item["height"]))
                path = output / "control_inputs" / f"{row['id']}.png"
                path.parent.mkdir(exist_ok=True)
                bundle.maps["mask"].convert("RGB").save(path)
                pairs[-1] = ("annotation_mask", pairs[-1][1])
                pairs.append(("control_input", path))
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
    parser.set_defaults(func=run)
