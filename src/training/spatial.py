"""Fail closed before the pinned trainer can skip or resize a mask reference."""

from pathlib import Path

from src.conditioning.controls import ControlKind
from src.conditioning.static import validate_reference_size
from src.datasets.controls import ControlDataset
from src.datasets.role_controls import RoleControlDataset, load_controls
from src.datasets.validation import validate


def attach_controls(config: dict, data_root: Path, control_root: Path, kind: str) -> None:
    controls = load_controls(control_root, data_root, kind)
    dataset = config["config"]["process"][0]["datasets"][0]
    dataset.update(
        control_path=[
            str((data_root / "train/reference").resolve()),
            str((control_root / "train" / controls.folder).resolve()),
        ],
        full_size_control_images=True,
    )
    config.setdefault("meta", {})["spatial_control"] = controls.metadata()
    verify_training_controls(config, controls=controls)


def verify_training_controls(
    config: dict, *, controls: ControlDataset | RoleControlDataset | None = None
) -> dict | None:
    process = config["config"]["process"][0]
    metadata = config.get("meta", {}).get("spatial_control")
    if not metadata:
        if any(
            isinstance(ds.get("control_path"), list) and len(ds["control_path"]) > 1
            for ds in process.get("datasets", [])
        ):
            raise ValueError("Multi-control training requires spatial provenance")
        return None
    if controls is None:
        validate(Path(metadata["data_root"]))
        controls = load_controls(
            Path(metadata["root"]), Path(metadata["data_root"]), metadata["kind"]
        )
    if metadata != controls.metadata():
        raise ValueError("Training control index SHA256 or configuration changed")
    if controls.kind == ControlKind.NONE or len(process["datasets"]) != 1:
        raise ValueError("Joint conditioning requires exactly one paired dataset")
    dataset = process["datasets"][0]
    if process["model"]["arch"] != "flux2_klein_4b":
        raise ValueError("Joint mask training currently supports FLUX.2 Klein Base 4B only")
    expected = [
        str(controls.data_root / "train/reference"),
        str(controls.root / "train" / controls.folder),
    ]
    if dataset["control_path"] != expected:
        raise ValueError("Training image order must be [source, mask]")
    if Path(dataset["folder_path"]).resolve() != controls.data_root / "train/target":
        raise ValueError("Training target folder differs from manifest")
    if not dataset.get("full_size_control_images"):
        raise ValueError("Joint conditions require full-size raw control images")
    forbidden = (
        "flip_x",
        "flip_y",
        "random_crop",
        "random_scale",
        "augments",
        "augmentations",
        "control_path_1",
        "control_path_2",
        "control_path_3",
        "mask_path",
        "alpha_mask",
    )
    if any(dataset.get(key) for key in forbidden) or dataset.get("scale", 1) != 1:
        raise ValueError("Joint conditions disallow implicit geometry changes and loss masks")
    if process["model"].get("model_kwargs", {}).get("match_target_res"):
        raise ValueError("Implicit bilinear control resizing is not supported")
    resolution = dataset["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 1:
        raise ValueError("Use one fixed spatial training resolution")
    side = resolution[0]
    validate_reference_size((side, side))
    for row in controls.rows["train"].values():
        if (int(row["width"]), int(row["height"])) != (side, side):
            raise ValueError("Training reference/target geometry must match the fixed resolution")
        for key, folder in (("source", "reference"), ("target", "target")):
            if row[f"{key}_path"] != f"train/{folder}/{row['id']}.png":
                raise ValueError("Training image pairing differs from declared paths")
    return controls.metadata()


def verify_joint_batch(batch, latent, *, kind: str = "static_mask") -> None:
    """Inspect actual upstream tensors before encoding, not just recipe declarations."""
    samples = getattr(batch, "control_tensor_list", None)
    if samples is None or len(samples) != latent.shape[0]:
        raise ValueError("Joint training requires a control list for every batch item")
    if getattr(batch, "control_tensor", None) is not None:
        raise ValueError("Joint training cannot fall back to a single control tensor")
    for images in samples:
        if len(images) != 2:
            raise ValueError("Every training sample must contain [source, mask]")
        expected_shape = (3, latent.shape[-2] * 16, latent.shape[-1] * 16)
        for image in images:
            if tuple(image.shape) != expected_shape:
                raise ValueError("Raw control geometry differs from target latent geometry")
            if not image.isfinite().all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Raw controls must be finite RGB in [0,1]")
        mask = images[1]
        if kind == ControlKind.ORACLE_ROLE:
            if not ((mask == 0) | (mask == 1)).all() or not (mask.sum(dim=0) <= 1).all():
                raise ValueError("Role reference must be one-hot T/D/P RGB or black ignore")
        elif not ((mask == 0) | (mask == 1)).all() or not (mask == mask[:1]).all():
            raise ValueError("Second reference must be a three-channel binary mask")
