"""Audit equal actual samples/noise, optimizer capacity, updates and feature geometry."""

import argparse
import copy
import json
from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.conditioning.features import FEATURE_MODES
from src.core.config import load_assets, load_yaml
from src.core.io import sha256, write_json
from src.datasets.scene_validation import validate_scene_dataset
from src.training.feature_config import verify_features


def audit_seed(jobs: list[dict]) -> dict:
    if {j["method"] for j in jobs} != set(FEATURE_MODES) or len(jobs) != len(FEATURE_MODES):
        raise ValueError("Audit all four feature methods at one seed")
    reference, records = None, {}
    for job in jobs:
        directory = Path(job["directory"])
        report_path = directory / "train_report.json"
        report = json.loads(report_path.read_text())
        path = directory / "train.yaml"
        config = load_yaml(path)
        meta = verify_features(config)
        stats, checkpoint = report["feature_statistics"], report["feature_checkpoint"]
        process = copy.deepcopy(config["config"]["process"][0])
        if (
            report["status"] != "complete"
            or meta != report["role_features"]
            or sha256(path) != report["config_sha256"]
            or sha256(path) != job["config_sha256"]
            or report["trainer_revision"] != load_assets()["repositories"]["ai-toolkit"]["revision"]
            or sha256(report["checkpoint"]) != report["checkpoint_sha256"]
            or sha256(checkpoint["path"]) != checkpoint["sha256"]
            or meta["mode"] != job["method"]
            or process["train"]["seed"] != job["seed"]
            or stats["initial_seed"] != job["seed"]
            or report["steps"] != process["train"]["steps"]
            or report["batch_size"] != process["train"]["batch_size"]
            or stats["fusion_calls"] != report["steps"]
            or stats["batches"] != report["steps"]
            or len(stats["input_trace"]) != report["steps"]
            or stats["source_tokens"] != [stats["feature_tokens"]] * report["batch_size"]
            or any(len(t["ids"]) != report["batch_size"] for t in stats["input_trace"])
        ):
            raise ValueError("Actual feature training provenance/budget differs")
        encoder, actual = load_encoder(Path(checkpoint["path"]), Path(report["checkpoint"]))
        if (
            actual != checkpoint["metadata"]
            or sum(p.numel() for p in encoder.parameters()) != stats["parameters"]
        ):
            raise ValueError("Feature checkpoint capacity/metadata differs")
        validate_scene_dataset(Path(meta["data_root"]))
        dataset = process["datasets"][0]
        for key in ("folder_path", "control_path"):
            dataset.pop(key)
        process.pop("training_folder")
        fingerprint = dict(
            process=process,
            manifests=meta["manifests"],
            data=meta["dataset_sha256"],
            encoder=meta["encoder"],
            feature_lr=meta["lr"],
            parameters=stats["parameters"],
            total_parameters=stats["total_trainable_parameters"],
            feature_tokens=stats["feature_tokens"],
            initial_sha256=stats["initial_sha256"],
            input_trace=stats["input_trace"],
        )
        if reference is not None and fingerprint != reference:
            raise ValueError(
                "Feature methods differ in actual inputs/capacity/initialization/budget"
            )
        reference = fingerprint
        records[job["method"]] = dict(
            report_sha256=sha256(report_path),
            lora_sha256=report["checkpoint_sha256"],
            encoder_sha256=checkpoint["sha256"],
            seconds=report["seconds"],
            peak_allocated_gib=report["peak_allocated_gib"],
            peak_reserved_gib=report["peak_reserved_gib"],
        )
    return dict(status="verified", effective_recipe=reference, methods=records)


def audit(study: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((study / "plan.json").read_text())
    seeds = {
        str(seed): audit_seed(
            [j for j in plan["jobs"] if j["phase"] == "train" and j["seed"] == seed]
        )
        for seed in plan["protocol"]["train_seeds"]
    }
    write_json(
        output, dict(status="verified", seeds=seeds, plan_sha256=sha256(study / "plan.json"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.study, args.output)


if __name__ == "__main__":
    main()
