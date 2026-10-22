"""Deterministically expand physical scene variations into an explicit pinned recipe."""

import copy
from pathlib import Path

import numpy as np
import yaml

from src.core.config import load_yaml
from src.core.paths import CONFIG_ROOT
from src.datasets.scene_config import finite, validate_scene_config


def validate_variations(variations: dict) -> None:
    if (
        set(variations) != {"schema_version", "shapes", "ranges"}
        or not isinstance(variations["shapes"], list)
        or not isinstance(variations["ranges"], dict)
    ):
        raise ValueError("Invalid physical variation schema")
    bounds = dict(
        object_width=(0.4, 1),
        color=(0, 1),
        source_x=(-1.2, 0.5),
        source_y=(-1.2, 0.5),
        light_x=(-10, 10),
        light_y=(-10, 10),
        light_z=(0.1, 10),
        light_energy=(1, 5000),
        floor_color1=(0, 1),
        floor_color2=(0, 1),
        checker_scale=(4, 24),
    )
    shapes, ranges = variations["shapes"], variations["ranges"]
    if (
        variations["schema_version"] != 1
        or not shapes
        or len(set(shapes)) != len(shapes)
        or not set(shapes) <= {"cube", "cylinder"}
        or set(ranges) != set(bounds)
    ):
        raise ValueError("Invalid physical variation schema/shapes/range names")
    for key, (low, high) in bounds.items():
        pair = ranges[key]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not finite(x, low, high) for x in pair)
            or pair[0] > pair[1]
        ):
            raise ValueError(f"Invalid physical sampling range: {key}")


def expand_scenes(
    template: dict,
    counts: dict[str, int],
    *,
    seed: int,
    resolution: int,
    variations: dict | None = None,
) -> dict:
    if set(counts) != {"train", "dev", "test"} or any(
        not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 1000 for n in counts.values()
    ):
        raise ValueError("Declare positive train/dev/test scene-group counts (at most 1000 each)")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("Use an explicit nonnegative scene sampling seed")
    config = copy.deepcopy(template)
    variations = copy.deepcopy(
        load_yaml(CONFIG_ROOT / "datasets/blender_variations.yaml")
        if variations is None
        else variations
    )
    validate_variations(variations)
    config["renderer"]["resolution"] = resolution
    rng = np.random.default_rng(seed)
    scenes = []

    def sample(key, length=None):
        values = np.round(rng.uniform(*variations["ranges"][key], size=length), 5)
        return float(values) if length is None else values.tolist()

    for split in ("train", "dev", "test"):
        for index in range(counts[split]):
            scenes.append(
                dict(
                    id=f"{split}_{seed}_{index:03d}",
                    split=split,
                    shape=variations["shapes"][index % len(variations["shapes"])],
                    object_width=sample("object_width"),
                    color=sample("color", 3),
                    source_xy=[sample("source_x"), sample("source_y")],
                    light_position=[sample("light_x"), sample("light_y"), sample("light_z")],
                    light_energy=sample("light_energy"),
                    floor_color1=sample("floor_color1", 3),
                    floor_color2=sample("floor_color2", 3),
                    checker_scale=sample("checker_scale"),
                )
            )
    config["scenes"] = scenes
    config["sampling"] = dict(
        seed=seed,
        scene_groups=counts,
        protocol="physical_sweep_v2_configured",
        variations=variations,
    )
    validate_scene_config(config)
    return config


def run(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    config = expand_scenes(
        load_yaml(args.template),
        dict(train=args.train_groups, dev=args.dev_groups, test=args.test_groups),
        seed=args.seed,
        resolution=args.resolution,
        variations=load_yaml(args.variations),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(args.output)


def configure(actions) -> None:
    parser = actions.add_parser("expand")
    parser.add_argument(
        "--template", type=Path, default=CONFIG_ROOT / "datasets/blender_translation.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variations", type=Path, default=CONFIG_ROOT / "datasets/blender_variations.yaml"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--train-groups", type=int, required=True)
    parser.add_argument("--dev-groups", type=int, required=True)
    parser.add_argument("--test-groups", type=int, required=True)
    parser.set_defaults(func=run)
