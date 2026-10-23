"""Verify frozen conditions, actual BF16 arrays and separate target archives before scoring."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.backends.direction_probe import geometry, native_cfg
from src.core.io import sha256
from src.core.metadata import tensor_sha
from src.datasets.features import load_features
from src.datasets.scene_manifest import load_role_supervision
from src.evaluation.direction_metrics import bf16_float
from src.evaluation.feature_teacher import TARGET_PREPROCESSING
from src.inference.direction_study import state_specs, verify_job
from src.inference.direction_worker import fields_for


@lru_cache(maxsize=128)
def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}


def verify_report(study, plan, job, path, smoke=False):
    import torch

    verify_job(job)
    report = json.loads(path.read_text())
    execution_path = path.parent.with_name(path.parent.name + "_execution") / "execution.json"
    execution = json.loads(execution_path.read_text())
    if (
        report["status"] != "complete"
        or report["job"] != job
        or report["plan_sha256"] != sha256(study / "plan.json")
        or report["training_updates"] != 0
        or report["test_status"] != "sealed"
        or report["smoke"] != smoke
        or not report["weights_unchanged"]
        or report["trainable_parameters"] != 0
        or execution["status"] != "complete"
        or execution["returncode"] != 0
        or execution["source"] != report["source"]
    ):
        raise ValueError("Incomplete or mismatched frozen direction execution")
    rows = plan["rows"]
    if smoke:
        group = next(r["group_id"] for r in rows if r["split"] == "train")
        rows = [r for r in rows if r["group_id"] == group and float(r["strength"]) <= 0.5]
    lookup = {r["id"]: r for r in rows}
    expected = {(r["id"], f, t) for r in rows for f, t in state_specs(plan["protocol"])}
    declared = [(r["id"], r["state"]["family"], r["state"]["time"]) for r in report["results"]]
    if (
        len(declared) != len(expected)
        or set(declared) != expected
        or set(report["groups"]) != {r["group_id"] for r in rows}
    ):
        raise ValueError("Incomplete, duplicate or non-preregistered direction matrix")
    root = Path(plan["data"])
    checked = set()
    contexts = {}
    supervision_hashes = {}
    for group, meta in report["groups"].items():
        if (
            not meta["native_cfg_pipeline_exact"]
            or not meta["predictions_complete_before_supervision"]
        ):
            raise ValueError("Direction input or target-isolation gate missing")
        input_path = Path(meta["input_path"])
        if sha256(input_path) != meta["input_sha256"]:
            raise ValueError("Source-only input archive changed")
        inputs = arrays(str(input_path))
        row = next(r for r in rows if r["group_id"] == group)
        grid = int(row["height"]) // 16, int(row["width"]) // 16
        positions = geometry(torch.from_numpy(inputs["img_ids"]), grid).numpy()
        if not np.array_equal(positions, inputs["positions"]):
            raise ValueError("Actual spatial alignment changed")
        if (
            tensor_sha(torch.from_numpy(inputs["epsilon"])[None].bfloat16())
            != meta["initial_noise_sha256"]
        ):
            raise ValueError("Archived pure noise differs from the actual initial condition")
        contexts[group] = tuple(
            meta[k]
            for k in [
                "source_tokens_sha256",
                "initial_noise_sha256",
                "text_sha256",
                "negative_text_sha256",
                "anchor_sha256",
            ]
        )
        ids = {r["id"] for r in rows if r["group_id"] == group}
        if set(meta["supervision"]) != ids:
            raise ValueError("Supervision cohort changed")
        for identifier, item in meta["supervision"].items():
            canonical = lookup[identifier]
            if (
                sha256(item["path"]) != item["sha256"]
                or item["target_preprocessing"] != TARGET_PREPROCESSING
                or any(
                    item[k] != canonical[k]
                    for k in ["target_sha256", "source_sha256", "role_sha256", "evidence_sha256"]
                )
            ):
                raise ValueError("Supervision bytes or actual dataset identity changed")
            target = arrays(item["path"])
            if target["target"].shape != inputs["anchor"].shape or target["masks"].shape != (
                3,
                grid[0] * grid[1],
            ):
                raise ValueError("Supervision geometry changed")
            labels = np.asarray(load_role_supervision(root, canonical).role_labels)
            h, w = labels.shape
            expected_masks = np.stack(
                [
                    (labels == k).reshape(h // 16, 16, w // 16, 16).mean((1, 3)).reshape(-1)
                    for k in [0, 1, 2]
                ]
            ).astype(np.float32)
            if not np.array_equal(expected_masks, target["masks"]):
                raise ValueError("Supervised regions differ from actual counterfactual labels")
            if float(canonical["strength"]) == 0 and not np.array_equal(
                target["target"], inputs["anchor"]
            ):
                raise ValueError("Identity supervision differs from the target-free source anchor")
            supervision_hashes[identifier] = tuple(
                sha256_bytes(target[k]) for k in ["target", "masks"]
            )
    fields = {r["id"]: load_features(root, r, job["method"]) for r in rows}
    for record in report["results"]:
        row = lookup[record["id"]]
        state = record["state"]
        meta = report["groups"][record["group_id"]]
        if (
            record["split"] != row["split"]
            or record["strength"] != float(row["strength"])
            or record["feature_sha256"] != fields[record["id"]].digest
            or set(record["variants"]) != set(plan["protocol"]["variants"])
            or state not in meta["states"]
        ):
            raise ValueError("Actual feature, group or declared condition changed")
        if sha256(state["path"]) != state["sha256"]:
            raise ValueError("Actual common state changed")
        x = arrays(state["path"])["x"]
        inputs = arrays(meta["input_path"])
        actual = float(torch.tensor(state["time"], dtype=torch.bfloat16))
        expected_x = (
            (1 - actual) * torch.from_numpy(inputs["anchor"])
            + actual * torch.from_numpy(inputs["epsilon"])
            if state["family"] == "source_bridge" and actual != 1
            else torch.from_numpy(inputs["epsilon"])
        )
        if state["actual_time"] != actual or not np.array_equal(
            expected_x.bfloat16().float().numpy(), x
        ):
            raise ValueError(
                "Common state is not the declared target-free source/noise construction"
            )
        field = fields[record["id"]]
        grid = field.size[1] // 16, field.size[0] // 16
        tokens = torch.from_numpy(field.tokens())[None]
        for variant, receipt in record["variants"].items():
            path = receipt["path"]
            if path not in checked:
                if sha256(path) != receipt["sha256"]:
                    raise ValueError("Raw velocity archive SHA256 changed")
                checked.add(path)
            raw = arrays(path)
            p, n = [
                torch.from_numpy(bf16_float(raw[k])).bfloat16() for k in ["positive", "negative"]
            ]
            if not np.array_equal(
                native_cfg(p, n, plan["inherited"]["inference"]["guidance"]).float().numpy(),
                bf16_float(raw["cfg_native"]),
            ):
                raise ValueError("Archived native CFG differs from the pinned BF16 formula")
            expected_fields = fields_for(
                tokens, grid, variant, plan["inherited"]["inference"]["seed"]
            )
            if (
                not np.array_equal(raw["fields"], expected_fields[0].numpy())
                or tensor_sha(expected_fields) != receipt["fields_sha256"]
            ):
                raise ValueError(
                    "Model silently ignored or changed the declared feature intervention"
                )
            for name in ["positive", "negative", "cfg_native", "fused"]:
                value = bf16_float(raw[name])
                if value.shape != x.shape:
                    raise ValueError("Actual velocity or fusion geometry changed")
                if (
                    name != "fused"
                    and tensor_sha(torch.from_numpy(value)[None].bfloat16())
                    != receipt["velocity_sha256"][name]
                ):
                    raise ValueError("BF16 velocity digest changed")
            expected_fused = (
                (torch.from_numpy(x).bfloat16() + torch.from_numpy(raw["residual"]).bfloat16())
                .float()
                .numpy()
            )
            if not np.array_equal(expected_fused, bf16_float(raw["fused"])):
                raise ValueError("Archived effective BF16 addition differs from the actual fusion")
    return report, contexts, supervision_hashes


def sha256_bytes(value):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
