"""Immutable single-reviewer imports, independent public panels and merged analysis."""

import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np

from src.core.io import read_rows, sha256, write_json, write_rows
from src.core.metadata import now
from src.evaluation.review_page import FIELDS, page
from src.evaluation.review_stats import summarize


def rating_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in ("task_id", "edit_success", "preservation", "failure", "reviewer"):
            if key in row:
                row[key] = row[key].strip()
    return rows


def task_groups(mapping: dict, annotations: Path | None) -> dict[str, str]:
    samples = {task["sample_id"] for task in mapping["tasks"]}
    if annotations is None:
        return dict.fromkeys(samples, "unclassified")
    rows = read_rows(annotations)
    if (
        len(rows) != len(samples)
        or {row.get("sample_id") for row in rows} != samples
        or any(
            set(row) != {"sample_id", "edit_task"} or not row["edit_task"].strip() for row in rows
        )
    ):
        raise ValueError("Task annotations must cover each sample exactly once")
    return {row["sample_id"]: row["edit_task"] for row in rows}


def import_review(
    bundle: Path, ratings: Path, output: Path, reviewer: str, *, independent: bool
) -> dict:
    from src.evaluation.review import score_review

    if output.exists():
        raise FileExistsError(output)
    if not independent or not re.fullmatch(r"[A-Za-z0-9_-]+", reviewer):
        raise ValueError("Declare independent review and use a stable non-identifying reviewer ID")
    rows = rating_rows(ratings)
    if not rows or any(row.get("reviewer", "").strip() != reviewer for row in rows):
        raise ValueError(
            "One file must contain exactly the declared reviewer, including pending rows"
        )
    # score_review validates all task IDs/values and keeps the exact input bytes.
    summary = score_review(bundle, ratings, output / "summary.json")
    record = dict(
        schema_version=1,
        imported=now(),
        reviewer=reviewer,
        independent_declared=True,
        ratings_sha256=summary["ratings_sha256"],
        review_sha256=summary["review_sha256"],
        summary_sha256=sha256(output / "summary.json"),
        status=summary["status"],
        source_path=str(ratings.resolve()),
    )
    write_json(output / "import.json", record)
    return record


def merge_reviews(
    bundle: Path,
    imports: list[Path],
    output: Path,
    *,
    annotations: Path | None = None,
    seed: int = 42,
    resamples: int = 2000,
) -> dict:
    from src.evaluation.review import score_review, verify_review

    if output.exists():
        raise FileExistsError(output)
    if not 100 <= resamples <= 100000:
        raise ValueError("Bootstrap resamples must be between 100 and 100000")
    mapping = verify_review(bundle)
    tasks = {task["task_id"]: task for task in mapping["tasks"]}
    groups = task_groups(mapping, annotations)
    manifest = {row["id"]: row for row in read_rows(mapping["manifest"])}
    reviewers, judgements, sources = {}, [], []
    # Validate every selected version before creating the immutable analysis directory.
    for directory in imports:
        record = json.loads((directory / "import.json").read_text())
        reviewer = record["reviewer"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", reviewer):
            raise ValueError("Unsafe imported reviewer ID")
        if reviewer in reviewers:
            raise ValueError("Select only one import version per independent reviewer")
        raw = directory / "summary.ratings.csv"
        if (
            record["schema_version"] != 1
            or record["independent_declared"] is not True
            or sha256(raw) != record["ratings_sha256"]
            or sha256(directory / "summary.json") != record["summary_sha256"]
            or record["review_sha256"] != sha256(bundle / "review.json")
        ):
            raise ValueError("Imported review provenance SHA256 changed")
        rows = rating_rows(raw)
        if any(row["reviewer"] != reviewer for row in rows):
            raise ValueError("Imported reviewer changed")
        summary = json.loads((directory / "summary.json").read_text())
        reviewers[reviewer] = summary["status"]
        sources.append(
            dict(path=str(directory.resolve()), sha256=sha256(directory / "import.json"))
        )
        for row in rows:
            if not row["edit_success"]:
                continue
            task = tasks[row["task_id"]]
            sample = task["sample_id"]
            judgements.append(
                dict(
                    row,
                    method=task["method"],
                    sample_id=sample,
                    scene_group=manifest[sample]["source_session"],
                    edit_task=groups[sample],
                )
            )
    ready = len(reviewers) >= 3 and all(state == "complete" for state in reviewers.values())
    # Revalidate semantics as well as the immutable hashes; these copies preserve raw bytes.
    for directory, reviewer in zip(imports, reviewers, strict=True):
        score_review(
            bundle,
            directory / "summary.ratings.csv",
            output / "imports" / reviewer / "summary.json",
        )
    methods = sorted(mapping["runs"])
    parameters = dict(ready=ready, seed=seed, resamples=resamples)
    result = dict(
        schema_version=1,
        status="complete" if ready else "pending",
        missing_complete_reviewers=max(0, 3 - sum(s == "complete" for s in reviewers.values())),
        reviewers=reviewers,
        imports=sources,
        review_sha256=sha256(bundle / "review.json"),
        annotations_sha256=sha256(annotations) if annotations else None,
        statistics=dict(
            seed=seed,
            resamples=resamples,
            confidence=0.95,
            unit="equal scene; crossed scene/reviewer bootstrap; paired scores",
        ),
        **summarize(judgements, methods, **parameters),
        by_edit_task={},
        by_failure={},
        failure_cases=[row for row in judgements if row["failure"]],
    )
    for key, field in (("by_edit_task", "edit_task"), ("by_failure", "failure")):
        categories = (
            set(groups.values()) if field == "edit_task" else {r[field] for r in judgements} - {""}
        )
        for value in sorted(categories):
            selected = [r for r in judgements if r[field] == value]
            # Failure strata are outcome-selected descriptions, not efficacy inference.
            scoped = parameters | dict(ready=False) if field == "failure" else parameters
            result[key][value] = summarize(selected, methods, **scoped)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    if annotations:
        shutil.copyfile(annotations, output / "task_annotations.csv")
    return result


def export_panels(bundle: Path, reviewers: list[str], output: Path, seed: int) -> dict:
    from src.evaluation.review import verify_review

    if output.exists():
        raise FileExistsError(output)
    if (
        len(set(reviewers)) < 3
        or len(set(reviewers)) != len(reviewers)
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+", reviewer) for reviewer in reviewers)
    ):
        raise ValueError("Provide at least three distinct safe reviewer IDs")
    mapping = verify_review(bundle)
    output.mkdir(parents=True)
    shutil.copytree(bundle / "images", output / "images")
    rng = np.random.default_rng(seed)
    for reviewer in reviewers:
        tasks = [mapping["tasks"][int(i)] for i in rng.permutation(len(mapping["tasks"]))]
        folder = output / reviewer
        folder.mkdir()
        (folder / "index.html").write_text(
            page(tasks, image_prefix="../images", reviewer=reviewer), encoding="utf-8"
        )
        write_rows(
            folder / "ratings.csv",
            [
                dict.fromkeys(FIELDS, "") | dict(task_id=t["task_id"], reviewer=reviewer)
                for t in tasks
            ],
        )
    result = dict(
        status="pending",
        reviewers=reviewers,
        seed=seed,
        tasks_per_reviewer=len(mapping["tasks"]),
        review_sha256=sha256(bundle / "review.json"),
    )
    write_json(output / "panels.json", result)
    return result


def configure(actions) -> None:
    parser = actions.add_parser("import")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--independent", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(
        func=lambda a: print(
            import_review(a.bundle, a.ratings, a.output, a.reviewer, independent=a.independent)
        )
    )
    parser = actions.add_parser("merge")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--import", dest="imports", action="append", type=Path, default=[])
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.set_defaults(
        func=lambda a: print(
            merge_reviews(
                a.bundle,
                a.imports,
                a.output,
                annotations=a.annotations,
                seed=a.seed,
                resamples=a.resamples,
            )
        )
    )
    parser = actions.add_parser("panels")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reviewer", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=lambda a: print(export_panels(a.bundle, a.reviewer, a.output, a.seed)))
