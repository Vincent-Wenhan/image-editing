"""Strict single-source training recipe for independent oracle features."""

import math
from pathlib import Path

from src.conditioning.features import FEATURE_MODES, FEATURE_SCHEMA, EncoderConfig
from src.conditioning.static import validate_reference_size
from src.core.config import load_assets
from src.core.io import read_rows, sha256
from src.core.paths import PROJECT_ROOT
from src.datasets.holdout import require_dev


def attach_features(config: dict, data: Path, seal: Path, mode: str, encoder: dict, lr: float):
    """Keep experiment information in metadata; the upstream dataset sees only source RGB."""
    config["meta"]["role_features"] = dict(
        mode=mode,
        schema=FEATURE_SCHEMA,
        data_root=str(data.resolve()),
        dataset_sha256=sha256(data / "dataset.json"),
        manifests={
            split: sha256(data / split / "manifest.csv") for split in ("train", "dev", "test")
        },
        cohort=str(seal.resolve()),
        cohort_sha256=sha256(seal),
        encoder=encoder,
        lr=lr,
    )
    dataset = config["config"]["process"][0]["datasets"][0]
    dataset["control_path"] = [str((data / "train/reference").resolve())]
    dataset["full_size_control_images"] = True
    verify_features(config)


def verify_features(config: dict) -> dict | None:
    meta = config.get("meta", {}).get("role_features")
    if meta is None:
        return None
    if config["meta"].get("spatial_control"):
        raise ValueError("Feature and image control recipes cannot be mixed")
    if meta["mode"] not in FEATURE_MODES or meta["schema"] != FEATURE_SCHEMA:
        raise ValueError("Unsupported feature recipe")
    if set(meta["manifests"]) != {"train", "dev", "test"}:
        raise ValueError("Feature study requires three sealed split manifests")
    EncoderConfig(**meta["encoder"])
    if not math.isfinite(meta["lr"]) or not 0 < meta["lr"] <= 0.01:
        raise ValueError("Invalid feature learning rate")
    root, seal = Path(meta["data_root"]), Path(meta["cohort"])
    if (
        sha256(seal) != meta["cohort_sha256"]
        or sha256(root / "dataset.json") != meta["dataset_sha256"]
    ):
        raise ValueError("Feature data/cohort SHA256 changed")
    for split, expected in meta["manifests"].items():
        if sha256(root / split / "manifest.csv") != expected:
            raise ValueError("Feature manifest SHA256 changed")
    require_dev(root, seal)
    process = config["config"]["process"][0]
    if process["model"]["arch"] != "flux2_klein_4b" or len(process["datasets"]) != 1:
        raise ValueError("Features require one fixed FLUX.2 Klein 4B dataset")
    base = (PROJECT_ROOT / load_assets()["models"]["base"]["path"]).resolve()
    if Path(process["model"]["name_or_path"]).resolve() != base:
        raise ValueError("Feature training requires the pinned local Klein Base 4B")
    dataset = process["datasets"][0]
    if (
        Path(dataset["folder_path"]).resolve() != (root / "train/target").resolve()
        or dataset["control_path"] != [str((root / "train/reference").resolve())]
        or not dataset["full_size_control_images"]
        or len(dataset["resolution"]) != 1
        or any(
            dataset.get(k)
            for k in (
                "flip_x",
                "flip_y",
                "random_crop",
                "random_scale",
                "augments",
                "control_path_1",
                "control_path_2",
                "control_path_3",
                "augmentations",
                "mask_path",
                "alpha_mask",
            )
        )
        or dataset.get("scale", 1) != 1
        or process["model"].get("model_kwargs", {}).get("match_target_res", False)
        or process["train"]["noise_scheduler"] != "flowmatch"
        or process["train"]["timestep_type"] != "sigmoid"
    ):
        raise ValueError("Features require unaugmented aligned source-only flowmatch training")
    side = dataset["resolution"][0]
    validate_reference_size((side, side))
    if any(
        (int(r["width"]), int(r["height"])) != (side, side)
        for r in read_rows(root / "train/manifest.csv")
    ):
        raise ValueError("Feature training cannot resize or crop spatial truth")
    return meta
