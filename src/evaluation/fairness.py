"""Compare actual training recipes/checkpoints and grouped pixel diagnostics."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.core.config import load_assets, load_yaml
from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.datasets.role_controls import load_controls
from src.datasets.validation import dataset_splits, validate, verify_images
from src.evaluation.metrics import load_image, metrics
from src.evaluation.provenance import verify_comparison, verify_run
from src.evaluation.review_import import task_groups


def matched_training(reports: dict[str, Path]) -> dict:
    """Same data and updates; two references also share sequence length at fixed size."""
    reference, manifests, methods = None, None, {}
    for method, path in sorted(reports.items()):
        report = json.loads(path.read_text())
        if (
            report["status"] != "complete"
            or report["trainer_revision"] != load_assets()["repositories"]["ai-toolkit"]["revision"]
        ):
            raise ValueError("Expected completed training at the pinned upstream revision")
        config_path = Path(report["config"])
        if (
            sha256(config_path) != report["config_sha256"]
            or sha256(report["checkpoint"]) != report["checkpoint_sha256"]
        ):
            raise ValueError("Training configuration/checkpoint SHA256 changed")
        config = load_yaml(config_path)
        metadata = config["meta"]["spatial_control"]
        data = load_controls(Path(metadata["root"]), Path(metadata["data_root"]), metadata["kind"])
        if data.metadata() != metadata or metadata != report["conditioning"]:
            raise ValueError("Training controls differ from report/configuration")
        counts = validate(data.data_root)
        actual = {
            split: sha256(data.data_root / split / "manifest.csv")
            for split in dataset_splits(data.data_root)
        }
        if manifests is not None and actual != manifests:
            raise ValueError("Matched training requires identical split manifests")
        manifests = actual
        process = copy.deepcopy(config["config"]["process"][0])
        if (
            report["steps"] != process["train"]["steps"]
            or report["batch_size"] != process["train"]["batch_size"]
        ):
            raise ValueError("Actual training budget differs from recipe")
        if report.get("joint_control_batches", 0) < report["steps"]:
            raise ValueError("Missing actual joint-reference batch verification")
        dataset = process["datasets"][0]
        if len(process["datasets"]) != 1 or dataset["control_path"] != [
            str(data.data_root / "train/reference"),
            str(data.root / "train" / data.folder),
        ]:
            raise ValueError("Training reference order/paths differ")
        for key in ("folder_path", "control_path"):
            dataset.pop(key)
        process.pop("training_folder")
        if reference is not None and process != reference:
            raise ValueError("Training recipe/update budget differs between methods")
        reference = process
        methods[method] = dict(
            report=str(path.resolve()),
            report_sha256=sha256(path),
            config_sha256=report["config_sha256"],
            checkpoint=report["checkpoint"],
            checkpoint_sha256=report["checkpoint_sha256"],
            conditioning=metadata,
            counts=counts,
        )
    if len(methods) < 2:
        raise ValueError("Compare at least two completed training runs")
    return dict(
        status="verified",
        manifests=manifests,
        methods=methods,
        effective_recipe=reference,
        scope="equal samples/updates/capacity/reference geometry; time reported separately",
    )


def grouped_diagnostics(
    root: Path, manifest: Path, runs: dict[str, Path], annotations: Path
) -> dict:
    records, reference, data_rows = {}, None, None
    for method, directory in sorted(runs.items()):
        record = json.loads((directory / "run.json").read_text())
        own_root = Path(record.get("conditioning", {}).get("data_root", root))
        data_rows = verify_run(own_root, manifest, record)
        for row in data_rows.values():
            verify_images(root, row)
        if reference:
            verify_comparison(reference, record)
        reference = record
        records[method] = record
    mapping = dict(tasks=[dict(sample_id=item["id"]) for item in reference["results"]])
    groups = task_groups(mapping, annotations)
    result = dict(
        scope="pixel diagnostics only; edit success and preservation ratings pending",
        annotations_sha256=sha256(annotations),
        methods={},
    )
    for method, directory in runs.items():
        rows = read_rows(directory / "metrics.csv")
        if {row["id"] for row in rows} != set(groups) or len(rows) != len(groups):
            raise ValueError("Metric cohort differs from fixed inference samples")
        threshold = json.loads((directory / "metrics.json").read_text())["threshold"]
        for row, item in zip(rows, records[method]["results"], strict=True):
            if row["id"] != item["id"]:
                raise ValueError("Metric row order differs from fixed inference")
            sample, size = data_rows[row["id"]], (item["width"], item["height"])
            actual = metrics(
                load_image(data_path(root, sample["source_path"]), size),
                load_image(data_path(root, sample["target_path"]), size),
                load_image(directory / f"{row['id']}.png", size),
                load_image(data_path(root, sample["mask_path"]), size, "L"),
                threshold,
            )
            if set(row) != {"id", *actual} or any(
                (
                    row[key] != ""
                    if value is None
                    else not np.isclose(float(row[key]), value, atol=1e-12, rtol=0)
                )
                for key, value in actual.items()
            ):
                raise ValueError("Saved metrics differ from actual images")
        grouped = {}
        for group in sorted(set(groups.values())):
            selected = [row for row in rows if groups[row["id"]] == group]
            grouped[group] = dict(count=len(selected))
            for key in rows[0]:
                if key != "id":
                    values = [float(row[key]) for row in selected if row[key] != ""]
                    grouped[group][key] = float(np.mean(values)) if values else None
        result["methods"][method] = dict(
            run_sha256=sha256(directory / "run.json"),
            metrics_sha256=sha256(directory / "metrics.csv"),
            by_edit_task=grouped,
        )
    return result


def pairs(arguments: list[str]) -> dict[str, Path]:
    result = {}
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if not separator or not key or not value or key in result:
            raise ValueError("Use unique METHOD=PATH arguments")
        result[key] = Path(value)
    return result


def run(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    runs, training = pairs(args.run), pairs(args.training)
    if set(runs) != set(training):
        raise ValueError("Provide a training report for each inference method")
    result = dict(
        training=matched_training(training),
        diagnostics=grouped_diagnostics(args.data_root, args.manifest, runs, args.annotations),
    )
    write_json(args.output, result)
    print(result)


def configure(actions) -> None:
    parser = actions.add_parser("audit")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--training", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    configure(parser.add_subparsers(required=True))
    args = parser.parse_args()
    args.func(args)
