"""Small explicit scene schema; no registry or implicit renderer defaults."""

import math
import re

from src.core.metadata import config_digest


def finite(value, low: float, high: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and low <= value <= high
    )


def validate_scene_config(config: dict) -> None:
    render = config["renderer"]
    if (
        config["schema_version"] != 1
        or render["version"] != "4.5.3"
        or render["engine"] != "CYCLES"
        or render["device"] != "CPU"
    ):
        raise ValueError("Only pinned Blender 4.5.3 Cycles CPU scenes are supported")
    for key, low, high in (
        ("resolution", 64, 512),
        ("samples", 1, 256),
        ("threads", 1, 32),
        ("seed", 0, 2**31 - 1),
    ):
        value = render[key]
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"Invalid renderer {key}")
    if render["resolution"] % 16:
        raise ValueError("Scene resolution must be a multiple of 16")
    strengths = config["strengths"]
    if (
        not strengths
        or any(not finite(s, 0, 1) for s in strengths)
        or strengths != sorted(set(strengths))
        or strengths[0] != 0
        or strengths[-1] == 0
    ):
        raise ValueError("Use unique ordered strengths including zero and a positive value")
    if not finite(config["max_translation_widths"], 0.1, 3):
        raise ValueError("Invalid physical displacement scale")
    if not finite(config["ortho_scale"], 2, 10) or not finite(config["light_energy"], 1, 5000):
        raise ValueError("Invalid camera/light scale")
    for key, length in (("camera_position", 3), ("camera_target", 3), ("source_xy", 2)):
        if len(config[key]) != length or any(not finite(x, -20, 20) for x in config[key]):
            raise ValueError(f"Invalid {key}")
    policy = config["roles"]
    if (
        not finite(policy["shadow_low"], 0, 1)
        or not finite(policy["shadow_high"], 0, 1)
        or not 0 < policy["shadow_low"] < policy["shadow_high"] < 1
        or not finite(policy["unexplained_tolerance"], 0, 1)
        or not isinstance(policy["boundary_radius"], int)
        or isinstance(policy["boundary_radius"], bool)
        or not 0 <= policy["boundary_radius"] <= 4
    ):
        raise ValueError("Invalid role thresholds or boundary margin")
    scenes = config["scenes"]
    if {scene["split"] for scene in scenes} != {"train", "dev"}:
        raise ValueError("Scene groups must include disjoint train and dev")
    if len({scene["id"] for scene in scenes}) != len(scenes):
        raise ValueError("Duplicate scene IDs")
    groups = set()
    for scene in scenes:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scene["id"]):
            raise ValueError("Unsafe scene ID")
        if scene["shape"] not in {"cube", "cylinder"} or not finite(scene["object_width"], 0.4, 1):
            raise ValueError("Unsupported object geometry")
        if len(scene["color"]) != 3 or any(not finite(x, 0, 1) for x in scene["color"]):
            raise ValueError("Object color must be unit RGB")
        if (
            len(scene["light_position"]) != 3
            or any(not finite(x, -10, 10) for x in scene["light_position"])
            or scene["light_position"][2] <= 0
        ):
            raise ValueError("Invalid point light position")
        group = scene_group(config, scene)
        if group in groups:
            raise ValueError("Duplicate physical scene group, including across splits")
        groups.add(group)


def scene_group(config: dict, scene: dict) -> str:
    return config_digest(
        dict(
            scene={k: v for k, v in scene.items() if k not in {"id", "split"}},
            camera_position=config["camera_position"],
            camera_target=config["camera_target"],
            ortho_scale=config["ortho_scale"],
            source_xy=config["source_xy"],
            light_energy=config["light_energy"],
        )
    )
