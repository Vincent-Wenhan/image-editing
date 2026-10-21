"""Generate directly runnable pinned ai-toolkit recipes, retaining legacy YAML support."""

from pathlib import Path

import yaml

from src.conditioning.static import MASK_KINDS
from src.core.config import ProjectSettings, load_yaml
from src.datasets.validation import validate
from src.training.spatial import attach_controls


def build_config(args) -> dict:
    if min(args.steps, args.batch_size, args.rank) < 1:
        raise ValueError("steps, batch-size and rank must be positive")
    validate(Path(args.data_root))
    config = load_yaml(args.template)
    config["config"]["name"] = args.name
    process = config["config"]["process"][0]
    process["training_folder"] = str(Path(args.output_root).resolve())
    process["train"].update(steps=args.steps, batch_size=args.batch_size)
    process["network"].update(linear=args.rank, linear_alpha=args.rank)
    process["save"]["save_every"] = min(
        ProjectSettings.load().process["save"]["save_every"], args.steps
    )
    dataset = process["datasets"][0]
    dataset["folder_path"] = str((Path(args.data_root) / "train/target").resolve())
    dataset["control_path"] = str((Path(args.data_root) / "train/reference").resolve())
    for key in ("name_or_path", "vae_path"):
        path = Path(process["model"][key]).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        process["model"][key] = str(path)
    kind = getattr(args, "control", "none")
    control_root = getattr(args, "control_root", None)
    if kind != "none":
        if not control_root:
            raise ValueError("Mask training requires --control-root from prepare-controls")
        attach_controls(config, Path(args.data_root), Path(control_root), kind)
    elif control_root:
        raise ValueError("No-mask training must not receive a control root")
    return config


def run(args) -> None:
    config = build_config(args)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Config: {path}")


def configure(parser) -> None:
    settings = ProjectSettings.load()
    process = settings.process
    parser.add_argument("--template", default="configs/flux2_edit_lora.yaml")
    parser.add_argument("--data-root", default=settings.dataset["root"])
    parser.add_argument("--output-root", default=process["training_folder"])
    parser.add_argument("--name", default=settings.recipe["config"]["name"])
    parser.add_argument("--steps", type=int, default=process["train"]["steps"])
    parser.add_argument("--batch-size", type=int, default=process["train"]["batch_size"])
    parser.add_argument("--rank", type=int, default=process["network"]["linear"])
    parser.add_argument("--output", default="outputs/configs/flux2_edit_lora.yaml")
    parser.add_argument("--control", choices=["none", *MASK_KINDS], default="none")
    parser.add_argument("--control-root", help="Prepared, hashed mask condition directory")
    parser.set_defaults(func=run)
