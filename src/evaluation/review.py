"""Hash-checked blinded human assessment, separate from automatic pixel diagnostics."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.io import sha256, write_json, write_rows
from src.core.paths import data_path
from src.datasets.validation import verify_images
from src.evaluation.provenance import verify_comparison, verify_run
from src.evaluation.review_page import FAILURES, FIELDS, page


def export_review(
    root: Path, manifest: Path, runs: dict[str, Path], output: Path, seed: int
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if len(runs) < 2 or any(not name.strip() for name in runs):
        raise ValueError("Human comparison requires at least two named runs")
    records, tasks, reference = {}, [], None
    for method, directory in sorted(runs.items()):
        record = json.loads((directory / "run.json").read_text())
        verification_root = Path(record.get("conditioning", {}).get("data_root", root))
        rows = verify_run(verification_root, manifest, record)
        for row in rows.values():
            verify_images(root, row)
        if reference is not None:
            verify_comparison(reference, record)
        reference = record
        records[method] = dict(
            path=str(directory.resolve()),
            sha256=sha256(directory / "run.json"),
            verification_root=str(verification_root.resolve()),
        )
        for result in record["results"]:
            identifier = result["id"]
            with Image.open(directory / f"{identifier}.png") as image:
                image.load()
                if image.size != (result["width"], result["height"]) or image.mode != "RGB":
                    raise ValueError("Prediction shape/mode differs from its run record")
            task_id = hashlib.sha256(f"{seed}:{method}:{identifier}".encode()).hexdigest()[:20]
            paths = dict(
                source=data_path(root, rows[identifier]["source_path"]),
                target=data_path(root, rows[identifier]["target_path"]),
                prediction=directory / f"{identifier}.png",
            )
            tasks.append(
                dict(
                    task_id=task_id,
                    method=method,
                    sample_id=identifier,
                    instruction=rows[identifier]["instruction"],
                    files={
                        key: dict(path=str(path.resolve()), sha256=sha256(path))
                        for key, path in paths.items()
                    },
                )
            )
    tasks.sort(key=lambda task: task["task_id"])
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate review tasks")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "private").mkdir()
    mapping = dict(
        schema_version=1,
        seed=seed,
        data_root=str(root.resolve()),
        manifest=str(manifest.resolve()),
        manifest_sha256=sha256(manifest),
        runs=records,
        tasks=tasks,
    )
    for task in tasks:
        for key, info in task["files"].items():
            shutil.copyfile(info["path"], output / "images" / f"{task['task_id']}_{key}.png")
    write_json(output / "private/mapping.json", mapping)
    index = dict(
        schema_version=1,
        status="pending",
        tasks=len(tasks),
        fixed_samples=len(reference["results"]),
        mapping_sha256=sha256(output / "private/mapping.json"),
    )
    write_json(output / "review.json", index)
    write_rows(
        output / "ratings.csv",
        [dict.fromkeys(FIELDS, "") | {"task_id": task["task_id"]} for task in tasks],
    )
    (output / "index.html").write_text(page(tasks), encoding="utf-8")
    return index


def verify_review(bundle: Path) -> dict:
    index = json.loads((bundle / "review.json").read_text())
    if (
        index["schema_version"] != 1
        or sha256(bundle / "private/mapping.json") != index["mapping_sha256"]
    ):
        raise ValueError("Review mapping SHA256 changed")
    mapping = json.loads((bundle / "private/mapping.json").read_text())
    root, manifest = Path(mapping["data_root"]), Path(mapping["manifest"])
    if sha256(manifest) != mapping["manifest_sha256"]:
        raise ValueError("Review manifest SHA256 changed")
    for info in mapping["runs"].values():
        run = Path(info["path"]) / "run.json"
        if sha256(run) != info["sha256"]:
            raise ValueError("Reviewed inference record SHA256 changed")
        verify_run(Path(info.get("verification_root", root)), manifest, json.loads(run.read_text()))
    for task in mapping["tasks"]:
        for key, info in task["files"].items():
            copy = bundle / "images" / f"{task['task_id']}_{key}.png"
            if sha256(copy) != info["sha256"] or sha256(info["path"]) != info["sha256"]:
                raise ValueError("Reviewed image SHA256 changed")
    return mapping


def score_review(bundle: Path, ratings: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    mapping = verify_review(bundle)
    with ratings.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    expected = {task["task_id"]: task for task in mapping["tasks"]}
    if len(rows) != len(expected) or {row["task_id"] for row in rows} != set(expected):
        raise ValueError("Ratings must contain every review task exactly once")
    methods = {name: [] for name in mapping["runs"]}
    failures, pending = [], 0
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("Unexpected rating columns")
        task = expected[row["task_id"]]
        values = [row[key].strip() for key in ("edit_success", "preservation")]
        if values == ["", ""]:
            if row["failure"].strip() or row["comment"].strip():
                raise ValueError("Unscored tasks cannot contain a failure judgement")
            pending += 1
            continue
        if any(value not in {"0", "1", "2", "NA"} for value in values):
            raise ValueError("Scores must both be 0/1/2/NA, or both empty")
        if not row["reviewer"].strip():
            raise ValueError("Completed ratings require a reviewer")
        failed = values != ["2", "2"]
        if row["failure"] and row["failure"] not in FAILURES:
            raise ValueError("Unknown failure category")
        if failed and (not row["failure"] or not row["comment"].strip()):
            raise ValueError("Low or unjudgeable scores require a failure category and explanation")
        if "NA" in values and row["failure"] != "unjudgeable":
            raise ValueError("NA requires unjudgeable category")
        methods[task["method"]].append(values)
        if failed or row["failure"]:
            failures.append(dict(method=task["method"], sample_id=task["sample_id"], **row))
    summary = dict(
        status="pending" if pending else "complete",
        pending=pending,
        task_count=len(rows),
        ratings_sha256=sha256(ratings),
        review_sha256=sha256(bundle / "review.json"),
        methods={},
        failure_cases=failures,
    )
    for method, values in methods.items():
        entry = dict(rated=len(values))
        for i, key in enumerate(("edit_success", "preservation")):
            scores = [int(row[i]) for row in values if row[i] != "NA"]
            entry[key] = dict(
                judgeable=len(scores),
                mean=float(np.mean(scores)) if scores else None,
                full_score_fraction=sum(x == 2 for x in scores) / len(scores) if scores else None,
            )
        summary["methods"][method] = entry
    raw = output.with_suffix(".ratings.csv")
    if raw.exists():
        raise FileExistsError(raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ratings, raw)
    summary["raw_ratings"] = str(raw.resolve())
    write_json(output, summary)
    return summary


def run(args) -> None:
    runs = {}
    for argument in args.run:
        label, separator, path = argument.partition("=")
        if not separator or not path or label in runs:
            raise ValueError("Use unique --run METHOD=PATH arguments")
        runs[label] = Path(path)
    print(
        export_review(Path(args.data_root), Path(args.manifest), runs, Path(args.output), args.seed)
    )


def configure_export(parser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", action="append", required=True, help="METHOD=PATH, at least twice")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=run)


def score(args) -> None:
    print(score_review(Path(args.bundle), Path(args.ratings), Path(args.output)))


def configure_score(parser) -> None:
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=score)


def configure(parser) -> None:
    from src.evaluation.fairness import configure as configure_audit
    from src.evaluation.review_import import configure as configure_imports

    actions = parser.add_subparsers(dest="review_action", required=True)
    configure_export(actions.add_parser("export"))
    configure_score(actions.add_parser("score"))
    configure_imports(actions)
    configure_audit(actions)
