"""Verify equal data/update budgets before reusing a completed No-mask baseline."""

import json
from pathlib import Path

from src.core.config import load_assets, load_yaml
from src.core.io import sha256
from src.datasets.controls import ControlDataset
from src.datasets.validation import validate


def verify_training_budget(reference_report: Path, candidate_config: Path) -> dict:
    report = json.loads(reference_report.read_text())
    revision = load_assets()["repositories"]["ai-toolkit"]["revision"]
    if report["status"] != "complete" or report["trainer_revision"] != revision:
        raise ValueError("Budget reference must be a completed pinned training run")
    config_path = Path(report["config"])
    if sha256(config_path) != report["config_sha256"]:
        raise ValueError("Reference training configuration SHA256 changed")
    if sha256(report["checkpoint"]) != report["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint SHA256 changed")
    reference, candidate = load_yaml(config_path), load_yaml(candidate_config)
    if reference.get("meta", {}).get("spatial_control") or report.get("conditioning"):
        raise ValueError("Expected the preserved No-mask baseline")
    metadata = candidate["meta"]["spatial_control"]
    controls = ControlDataset(Path(metadata["root"]), Path(metadata["data_root"]), metadata["kind"])
    if metadata != controls.metadata():
        raise ValueError("Candidate control provenance differs from its recipe")
    left, right = [value["config"]["process"][0] for value in (reference, candidate)]
    for key in ("train", "network", "model", "sample"):
        if left[key] != right[key]:
            raise ValueError(f"Training budget or recipe differs: {key}")
    for key in ("steps", "batch_size"):
        if left["train"][key] != report[key]:
            raise ValueError("Reference report differs from its effective training configuration")
    if len(left["datasets"]) != 1 or len(right["datasets"]) != 1:
        raise ValueError("Budget comparison requires one paired dataset per method")
    excluded = {"folder_path", "control_path", "full_size_control_images"}
    datasets = [value["datasets"][0] for value in (left, right)]
    dataset_settings = [{k: v for k, v in ds.items() if k not in excluded} for ds in datasets]
    if dataset_settings[0] != dataset_settings[1]:
        raise ValueError("Training dataset settings differ")
    roots = [Path(ds["folder_path"]).resolve().parent.parent for ds in datasets]
    if datasets[0]["control_path"] != str(roots[0] / "train/reference"):
        raise ValueError("Reference recipe is not source-only No-mask training")
    counts = [validate(root) for root in roots]
    manifests = {}
    for split in ("train", "dev"):
        hashes = [sha256(root / split / "manifest.csv") for root in roots]
        if len(set(hashes)) != 1:
            raise ValueError(f"Training comparison uses different {split} manifests")
        manifests[split] = hashes[0]
    return dict(
        reference_report=str(reference_report.resolve()),
        reference_report_sha256=sha256(reference_report),
        reference_checkpoint_sha256=report["checkpoint_sha256"],
        reference_config_sha256=report["config_sha256"],
        candidate_config_sha256=sha256(candidate_config),
        manifests=manifests,
        train_count=counts[0]["train"],
        dev_count=counts[0]["dev"],
        steps=report["steps"],
        batch_size=report["batch_size"],
        gradient_accumulation=left["train"]["gradient_accumulation"],
        seed=left["train"]["seed"],
        scope="equal samples/updates/recipe; extra mask tokens increase compute",
    )
