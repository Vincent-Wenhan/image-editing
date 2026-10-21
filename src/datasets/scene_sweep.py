"""Deterministically expand physical scene variations into an explicit pinned recipe."""

import copy
from pathlib import Path

import numpy as np
import yaml

from src.core.config import load_yaml
from src.core.paths import CONFIG_ROOT
from src.datasets.scene_config import validate_scene_config


def expand_scenes(template: dict, counts: dict[str, int], *, seed: int, resolution: int) -> dict:
    if set(counts) != {"train", "dev", "test"} or any(
        not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 1000 for n in counts.values()
    ):
        raise ValueError("Declare positive train/dev/test scene-group counts (at most 1000 each)")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("Use an explicit nonnegative scene sampling seed")
    config = copy.deepcopy(template)
    config["renderer"]["resolution"] = resolution
    rng = np.random.default_rng(seed)
    scenes = []
    for split, count in counts.items():
        for index in range(count):

            def vector(low, high, length):
                return np.round(rng.uniform(low, high, length), 5).tolist()

            scenes.append(
                dict(
                    id=f"{split}_{seed}_{index:03d}",
                    split=split,
                    shape="cube" if index % 2 == 0 else "cylinder",
                    object_width=round(float(rng.uniform(0.5, 0.8)), 5),
                    color=vector(0.05, 0.75, 3),
                    source_xy=[
                        round(float(rng.uniform(-0.9, -0.5)), 5),
                        round(float(rng.uniform(-0.3, 0.3)), 5),
                    ],
                    light_position=[
                        round(float(rng.uniform(-4, -2)), 5),
                        round(float(rng.uniform(-5, -2.5)), 5),
                        round(float(rng.uniform(5, 7)), 5),
                    ],
                    light_energy=round(float(rng.uniform(650, 1000)), 5),
                    floor_color1=vector(0.30, 0.45, 3),
                    floor_color2=vector(0.50, 0.65, 3),
                    checker_scale=round(float(rng.uniform(10, 14)), 5),
                )
            )
    config["scenes"] = scenes
    config["sampling"] = dict(seed=seed, scene_groups=counts, protocol="physical_sweep_v1")
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--train-groups", type=int, required=True)
    parser.add_argument("--dev-groups", type=int, required=True)
    parser.add_argument("--test-groups", type=int, required=True)
    parser.set_defaults(func=run)
