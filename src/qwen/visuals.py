"""CPU-only paired Qwen Dev galleries and bounded failure cases."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path


def _specs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in result:
            raise ValueError("Methods must use unique LABEL=metrics.json values")
        result[label] = Path(path).resolve()
    if not result:
        raise ValueError("At least one scored method is required")
    return result


def _image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != size:
            raise ValueError(f"Gallery image geometry/mode differs: {path}")
        return image.copy()


def _method_records(path: Path) -> tuple[dict[str, dict], Path]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    predictions = Path(metrics["predictions"]).resolve()
    inference = json.loads((predictions / "inference.json").read_text(encoding="utf-8"))
    if inference.get("status") != "complete":
        raise ValueError(f"Gallery requires completed inference: {predictions}")
    checks = {item["sample_id"]: item for item in inference["results"]}
    records = {item["id"]: item for item in metrics["records"]}
    if set(records) != set(checks):
        raise ValueError(f"Metrics/inference cohorts differ: {path}")
    for sample_id, item in checks.items():
        prediction = predictions / f"{sample_id}.png"
        if sha256(prediction) != item["prediction_sha256"]:
            raise ValueError(f"Gallery prediction checksum changed: {sample_id}")
    return records, predictions


def _panel(
    row: dict,
    labels: list[str],
    images: dict[str, Image.Image],
    width: int = 256,
    label_height: int = 24,
) -> Image.Image:
    height = width + label_height
    canvas = Image.new("RGB", (width * len(labels), height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, label in enumerate(labels):
        image = images[label].resize((width, width), Image.Resampling.LANCZOS)
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.rectangle((x, 0, x + width - 1, label_height - 1), fill="white")
        draw.text((x + 4, 5), label, fill="black", font=font)
    return canvas


def _contact(
    rows: list[dict],
    labels: list[str],
    images: dict[str, dict[str, Image.Image]],
    output: Path,
) -> None:
    if not rows:
        raise ValueError("Cannot create an empty Qwen gallery")
    panels = [
        _panel(row, labels, {label: images[row["id"]][label] for label in labels}) for row in rows
    ]
    canvas = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
    y = 0
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, panel in zip(rows, panels, strict=True):
        canvas.paste(panel, (0, y))
        draw.text((4, y + panel.height - 14), row["id"], fill="yellow", font=font)
        y += panel.height
    canvas.save(output)


def run(args) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new Qwen gallery directory: {output}")
    if args.failures < 1:
        raise ValueError("--failures must be positive")
    methods = _specs(args.method)
    run_root = Path(args.run).resolve()
    record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    root = Path(record["data_root"])
    rows = {row["id"]: row for row in read_rows(root / "dev/manifest.csv")}
    method_rows, prediction_dirs = {}, {}
    for label, metrics in methods.items():
        method_rows[label], prediction_dirs[label] = _method_records(metrics)
    ids = set(rows)
    for label, scored in method_rows.items():
        if set(scored) != ids:
            raise ValueError(f"Gallery cohort differs for {label}")
    selected = sorted(rows.values(), key=lambda row: (row["group_id"], float(row["strength"])))
    labels = ["source", "target", *methods]
    images = {}
    for row in selected:
        size = int(row["width"]), int(row["height"])
        images[row["id"]] = {
            "source": _image(data_path(root, row["source_path"]), size),
            "target": _image(data_path(root, row["target_path"]), size),
        }
        for label, directory in prediction_dirs.items():
            images[row["id"]][label] = _image(directory / f"{row['id']}.png", size)
    output.mkdir(parents=True)
    _contact(selected, labels, images, output / "paired_dev.png")
    failures = sorted(
        selected,
        key=lambda row: max(method_rows[label][row["id"]]["target_mae"] for label in methods),
        reverse=True,
    )[: args.failures]
    failure_records = []
    for row in failures:
        scores = {label: method_rows[label][row["id"]]["target_mae"] for label in methods}
        failure_records.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "strength": float(row["strength"]),
                "target_mae": scores,
                "selection": "highest target MAE among compared methods; diagnostic only",
            }
        )
    _contact(failures, labels, images, output / "failure_cases.png")
    write_json(
        output / "failures.json",
        {
            "schema_version": "qwen_gallery_v1",
            "run": str(run_root),
            "methods": {label: str(path) for label, path in methods.items()},
            "records": failure_records,
            "scope": "paired visual diagnostics; no human rating or semantic success claim",
        },
    )
    print({"output": str(output), "samples": len(selected), "failures": len(failures)})


def configure(parser) -> None:
    parser.add_argument("--run", required=True, help="Any matched Qwen run containing run.json")
    parser.add_argument("--method", action="append", required=True, help="LABEL=metrics.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures", type=int, default=4)
    parser.set_defaults(func=run)
