"""Non-blinded multi-condition contacts with calibrated position overlays and raw links."""

import argparse
import html
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.core.io import read_rows, sha256, write_json
from src.datasets.validation import verify_images


def export(summary: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(summary.read_text())
    if result["status"] != "complete":
        raise ValueError("Gallery requires a completed hash-verified analysis")
    root = Path(result["data"])
    if sha256(root / "dev/manifest.csv") != result["manifest_sha256"]:
        raise ValueError("Gallery Dev manifest changed")
    rows = {r["id"]: r for r in read_rows(root / "dev/manifest.csv")}
    output.mkdir(parents=True)
    font = ImageFont.load_default(size=13)
    cards, contacts = [], []
    for seed, methods in result["seeds"].items():
        identifiers = sorted({r["id"] for m in methods.values() for r in m["records"]})
        for identifier in identifiers:
            row = rows[identifier]
            verify_images(root, row)
            items = [(label, root / row[f"{label}_path"], None) for label in ["source", "target"]]
            for name, method in methods.items():
                record = next(r for r in method["records"] if r["id"] == identifier)
                path = Path(record["prediction_path"])
                if sha256(path) != record["prediction_sha256"]:
                    raise ValueError("Visualized prediction SHA256 changed")
                items.append((name, path, record))
            sheet = Image.new("RGB", (1152, 55 + math.ceil(len(items) / 4) * 310), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (6, 4),
                f"{identifier} / training seed {seed}; non-blinded proxies, no human scores",
                fill="black",
                font=font,
            )
            draw.text(
                (6, 23),
                "Red=old, green=target, blue=template candidate; raw images linked below",
                fill="black",
                font=font,
            )
            links = []
            for i, (name, path, record) in enumerate(items):
                with Image.open(path) as image:
                    image = image.convert("RGB")
                overlay = ImageDraw.Draw(image)
                calibration = result["calibration"][identifier]
                for key, color in [("old_centroid", "red"), ("new_centroid", "lime")]:
                    x, y = calibration[key]
                    overlay.line((x - 7, y, x + 7, y), fill=color, width=2)
                    overlay.line((x, y - 7, x, y + 7), fill=color, width=2)
                if record:
                    for peak in record["localization"]["candidates"]:
                        x, y = peak["x"], peak["y"]
                        overlay.ellipse((x - 7, y - 7, x + 7, y + 7), outline="cyan", width=2)
                x, y = (i % 4) * 288, 55 + (i // 4) * 310
                sheet.paste(ImageOps.contain(image, (256, 256)), (x, y))
                draw.text((x + 3, y + 257), name, fill="black", font=font)
                if record:
                    metrics = [
                        record[k] for k in ["t_target_mae", "d_target_mae", "p_source_drift"]
                    ]
                    text = "T/D/P: " + "/".join("NA" if v is None else f"{v:.4f}" for v in metrics)
                    draw.text((x + 3, y + 273), text, fill="black", font=font)
                    draw.text(
                        (x + 3, y + 289),
                        "locator: " + record["localization"]["status"],
                        fill="black",
                        font=font,
                    )
                link = html.escape(os.path.relpath(path.resolve(), output.resolve()), quote=True)
                links.append(f'<a href="{link}">{html.escape(name)} raw</a>')
            path = output / f"{seed}_{identifier}.png"
            sheet.save(path)
            contacts.append(dict(seed=seed, id=identifier, path=path.name, sha256=sha256(path)))
            cards.append(
                f"<article><h2>{html.escape(identifier)} / {seed}</h2><p>{' | '.join(links)}</p>"
                f'<img src="{path.name}" style="max-width:100%"></article>'
            )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Non-blinded feature diagnostics</title>'
        "<p>Research contact sheets. Template candidates and pixel proxies are not human scores. "
        "Do not distribute this gallery to blinded reviewers.</p>" + "\n".join(cards)
    )
    write_json(
        output / "index.json",
        dict(
            status="complete",
            summary_sha256=sha256(summary),
            contacts=contacts,
            scope="non-blinded agent diagnostics",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.summary, args.output)


if __name__ == "__main__":
    main()
