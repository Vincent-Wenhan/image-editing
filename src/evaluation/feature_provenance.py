"""Verify the actual paired encoder, oracle fields and sealed Dev identity."""

from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.core.config import load_assets
from src.core.io import sha256
from src.datasets.features import load_features
from src.datasets.holdout import require_dev
from src.inference.feature_runner import feature_identity


def verify_features(root: Path, manifest: Path, record: dict, rows: list[dict]):
    config, metadata = record["config"], record["role_features"]
    seal, encoder_path = Path(config["cohort"]), Path(config["encoder"])
    require_dev(root, seal, manifest)
    if (
        metadata["cohort_sha256"] != sha256(seal)
        or metadata["sha256"] != sha256(encoder_path)
        or record["lora_sha256"] != sha256(config["lora"])
        or metadata["mode"] != config["feature_mode"]
        or Path(metadata["cohort"]).resolve() != seal.resolve()
        or Path(metadata["path"]).resolve() != encoder_path.resolve()
        or metadata["image_order"] != "source"
        or metadata["feature_tokens"] != (config["side"] // 16) ** 2
        or config.get("reference_preprocessing") != "training"
        or config.get("control", "none") != "none"
    ):
        raise ValueError("Feature checkpoint/cohort/layout provenance changed")
    encoder, actual = load_encoder(encoder_path, Path(config["lora"]))
    if actual != metadata["checkpoint_metadata"]:
        raise ValueError("Feature checkpoint metadata changed")
    base = load_assets()["models"]["base"]
    if any(record["model"].get(key) != base[key] for key in ("repo_id", "revision")):
        raise ValueError("Feature inference base differs from its paired checkpoint")
    intervention = config.get("residual_intervention")
    if bool(intervention) != bool(record.get("residual_intervention")) or (
        intervention and intervention != record["residual_intervention"]
    ):
        raise ValueError("Residual intervention metadata cannot be omitted or changed")
    for row, item in zip(rows, record["results"], strict=True):
        field = load_features(root, row, config["feature_mode"])
        if item["role_features"] != feature_identity(field):
            raise ValueError("Actual oracle feature SHA256 changed")
        if bool(intervention) != ("residual_condition" in item):
            raise ValueError("Residual intervention provenance cannot be omitted")
        if intervention:
            from src.evaluation.feature_residual_provenance import verify_condition

            verify_condition(item["residual_condition"], encoder, field, intervention)
        if (
            item["fusion"]["tokens"] != metadata["feature_tokens"]
            or item["fusion"]["calls"] < config["steps"]
            or item["fusion"]["source_tokens"] != [metadata["feature_tokens"]]
            or item["fusion"]["reference_times"] != [10]
        ):
            raise ValueError("Incomplete feature fusion in saved inference")
