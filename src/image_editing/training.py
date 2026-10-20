"""Resolve the pinned ai-toolkit recipe without modifying upstream code."""

from pathlib import Path

import yaml

from image_editing.validation import validate


def run(args):
    if min(args.steps, args.batch_size, args.rank) < 1:
        raise ValueError("steps, batch-size and rank must be positive")
    validate(Path(args.data_root))
    config = yaml.safe_load(Path(args.template).read_text())
    config["config"]["name"] = args.name
    process = config["config"]["process"][0]
    process["training_folder"] = str(Path(args.output_root).resolve())
    process["train"].update(steps=args.steps, batch_size=args.batch_size)
    process["network"].update(linear=args.rank, linear_alpha=args.rank)
    process["save"]["save_every"] = min(50, args.steps)
    dataset = process["datasets"][0]
    dataset["folder_path"] = str((Path(args.data_root) / "train/target").resolve())
    dataset["control_path"] = str((Path(args.data_root) / "train/reference").resolve())
    for key in ("name_or_path", "vae_path"):
        path = Path(process["model"][key]).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        process["model"][key] = str(path)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Config: {path}")


def configure(parser):
    parser.add_argument("--template", default="configs/flux2_edit_lora.yaml")
    parser.add_argument("--data-root", default="data/magicbrush")
    parser.add_argument("--output-root", default="outputs/lora")
    parser.add_argument("--name", default="flux2_magicbrush_smoke")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--output", default="outputs/configs/flux2_edit_lora.yaml")
    parser.set_defaults(func=run)
