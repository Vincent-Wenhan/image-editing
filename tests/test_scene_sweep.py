"""Programmatic geometry and held-out physical groups preserve legacy identities."""

import copy

import pytest

from src.core.config import load_yaml
from src.core.paths import CONFIG_ROOT
from src.datasets.scene_config import scene_group, validate_scene_config
from src.datasets.scene_sweep import expand_scenes


def test_sweep_is_deterministic_disjoint_and_resolution_does_not_change_groups():
    template = load_yaml(CONFIG_ROOT / "datasets/blender_translation.yaml")
    counts = dict(train=12, dev=4, test=4)
    first = expand_scenes(template, counts, seed=71, resolution=128)
    assert first == expand_scenes(template, counts, seed=71, resolution=128)
    high = expand_scenes(template, counts, seed=71, resolution=512)
    assert first["scenes"] == high["scenes"]
    groups = [scene_group(first, scene) for scene in first["scenes"]]
    assert len(groups) == len(set(groups)) == 20
    assert {s["split"] for s in first["scenes"]} == set(counts)
    assert len({tuple(s["source_xy"]) for s in first["scenes"]}) == 20
    assert len({s["light_energy"] for s in first["scenes"]}) == 20
    assert [scene_group(high, s) for s in high["scenes"]] == groups
    invalid = copy.deepcopy(first)
    invalid["scenes"][-1] = dict(invalid["scenes"][0], id="duplicate", split="test")
    with pytest.raises(ValueError):
        validate_scene_config(invalid)


def test_explicit_default_overrides_do_not_evade_physical_group_identity():
    config = load_yaml(CONFIG_ROOT / "datasets/blender_translation.yaml")
    scene = config["scenes"][0]
    explicit = dict(
        scene,
        source_xy=config["source_xy"],
        light_energy=config["light_energy"],
        floor_color1=[0.38, 0.42, 0.46],
        floor_color2=[0.65, 0.6, 0.5],
        checker_scale=12,
    )
    assert scene_group(config, scene) == scene_group(config, explicit)
