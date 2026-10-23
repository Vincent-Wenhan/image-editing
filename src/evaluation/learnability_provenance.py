"""Bind controlled generation to Train selection, actual paired weights and noise policy."""

import json
import math
from pathlib import Path

from src.conditioning.learnability import research_tokens
from src.core.io import sha256
from src.core.metadata import tensor_sha
from src.core.random import sample_seed
from src.datasets.features import load_features
from src.inference.learnability import case_policy, checkpoint_pairs, evaluation_rows
from src.training.learnability_study import completed


def verify_teacher(item, row, protocol):
    teacher = item["teacher"]
    if item["variant"] != "original":
        if teacher is not None:
            raise ValueError("Intervention metadata incorrectly declares an original teacher")
        return
    if (
        teacher is None
        or teacher["purpose"] != "supervised_velocity_probe"
        or teacher["target_is_reference"]
        or teacher["target_sha256"] != row["target_sha256"]
        or [r["time"] for r in teacher["records"]] != protocol["teacher_times"]
        or any(
            r["source_tokens_sha256"] != item["source_tokens_sha256"]
            or r["text_sha256"] != item["text_sha256"]
            or not math.isfinite(r["velocity_mse"])
            or any(v is not None and not math.isfinite(v) for v in r["regions"].values())
            for r in teacher["records"]
        )
    ):
        raise ValueError("Supervised teacher diagnostic identity changed or is missing")


def verify_inference(study: Path, plan: dict, job: dict):
    import torch

    directory = Path(job["directory"])
    report_path = directory / "inference/report.json"
    report = json.loads(report_path.read_text())
    execution = json.loads((directory / "inference_execution/execution.json").read_text())
    trained = completed(job)
    pairs = checkpoint_pairs(trained, job["method"])
    p = plan["protocol"]
    if (
        report["status"] != "complete"
        or report["job"] != job
        or report["plan_sha256"] != sha256(study / "plan.json")
        or report["training_report_sha256"] != sha256(directory / "train_report.json")
        or report["pairs"] != pairs
        or report["prompt"] != p["prompt"]
        or report["seed_policy"] != "group_id"
        or report["test_status"] != "sealed"
        or execution["status"] != "complete"
        or execution["returncode"] != 0
        or execution["source"] != report["source"]
    ):
        raise ValueError("Incomplete or mismatched controlled inference provenance")
    rows = evaluation_rows(plan, job)
    lookup = {r["id"]: r for r in rows}
    expected = set()
    for pair in pairs:
        chosen = (
            rows
            if pair["update"] in {0, p["steps"]}
            else [r for r in rows if r["split"] == "train"]
        )
        for row in chosen:
            names, _ = case_policy(job["method"], pair["update"], row, rows, p)
            expected |= {(pair["update"], row["id"], name) for name in names}
    declared = [(r["update"], r["id"], r["variant"]) for r in report["results"]]
    if set(declared) != expected or len(declared) != len(expected):
        raise ValueError("Missing or duplicate controlled inference matrix")
    fields = {r["id"]: load_features(Path(plan["data"]), r, job["method"]) for r in rows}
    conditions, constants = {}, {}
    for item in report["results"]:
        row, field = lookup[item["id"]], fields[item["id"]]
        pair = next(pair for pair in pairs if pair["update"] == item["update"])
        tokens = torch.from_numpy(field.tokens())[None]
        grid = field.size[1] // 16, field.size[0] // 16
        transformed = research_tokens(tokens, grid, item["variant"], p["inference"]["seed"])
        if (
            item["split"] != row["split"]
            or item["group_id"] != row["group_id"]
            or item["strength"] != float(row["strength"])
            or item["source_sha256"] != row["source_sha256"]
            or item["feature_sha256"] != field.digest
            or item["actual_feature_tokens_sha256"] != tensor_sha(transformed)
            or item["checkpoint_sha256"] != pair["lora_sha256"]
            or item["encoder_sha256"] != pair["encoder"]["sha256"]
            or item["seed"] != sample_seed(p["inference"]["seed"], row["group_id"])
            or item["prompt"] != p["prompt"]
            or any(item[k] != p["inference"][k] for k in ("steps", "guidance"))
            or (item["width"], item["height"]) != field.size
            or sha256(item["path"]) != item["prediction_sha256"]
            or item["fusion"]["calls"] != p["inference"]["steps"] * 2
            or item["fusion"]["source_tokens"] != [grid[0] * grid[1]]
            or item["fusion"]["reference_times"] != [10]
        ):
            raise ValueError(
                "Actual controlled condition, checkpoint, sampling budget or PNG changed"
            )
        verify_teacher(item, row, p)
        if item["teacher"] is not None:
            for teacher in item["teacher"]["records"]:
                conditions[("teacher", row["id"], item["update"], teacher["time"])] = tuple(
                    teacher[k]
                    for k in [
                        "target_latent_sha256",
                        "noise_sha256",
                        "source_tokens_sha256",
                        "text_sha256",
                    ]
                )
        key = row["group_id"], item["update"]
        identity = tuple(
            item[k] for k in ["source_tokens_sha256", "initial_noise_sha256", "text_sha256"]
        )
        if key in conditions and conditions[key] != identity:
            raise ValueError("Same-group conditions/noise differ across strengths or interventions")
        conditions[key] = identity
        if item["variant"] == "original" and (job["method"] == "constant" or item["update"] == 0):
            if key in constants and constants[key] != item["prediction_sha256"]:
                raise ValueError(
                    "Constant/zero-init same-noise strengths no longer have identical PNG"
                )
            constants[key] = item["prediction_sha256"]
    expected_probes = {
        r["id"] for r in rows if case_policy(job["method"], p["steps"], r, rows, p)[1]
    }
    if (
        len(report["probes"]) != len(expected_probes)
        or {r["id"] for r in report["probes"]} != expected_probes
        or len(report["gradient_probes"]) != len(expected_probes)
        or {r["id"] for r in report["gradient_probes"]} != expected_probes
    ):
        raise ValueError("Missing or duplicate fixed-state/gradient probes")
    return report, conditions
