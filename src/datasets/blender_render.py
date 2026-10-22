"""Isolated CPU Blender worker; emits linear counterfactual evidence and raw EXRs."""

import argparse
import time
from pathlib import Path

import numpy as np

from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.metadata import config_digest
from src.datasets.blender_scene import build_scene, material
from src.datasets.scene_config import scene_group, validate_scene_config


def render_linear(scene, path: Path) -> np.ndarray:
    import bpy

    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        values = np.asarray(image.pixels[:], dtype=np.float32).reshape(
            scene.render.resolution_y, scene.render.resolution_x, 4
        )
        return values[::-1, :, :3].copy()
    finally:
        bpy.data.images.remove(image)


def occupancy(scene, obj, floor, light, path: Path) -> np.ndarray:
    import bpy

    original = (obj.data.materials[0], floor.data.materials[0], scene.cycles.samples)
    try:
        obj.data.materials[0] = material(bpy, "id_foreground", (1, 1, 1), emission=True)
        floor.data.materials[0] = material(bpy, "id_background", (0, 0, 0), emission=True)
        light.hide_render = True
        scene.cycles.samples = min(16, original[2])
        return np.clip(render_linear(scene, path)[:, :, 0], 0, 1)
    finally:
        obj.data.materials[0], floor.data.materials[0], scene.cycles.samples = original
        light.hide_render = False


def render_group(config: dict, spec: dict, output: Path) -> dict:
    scene, obj, floor, light = build_scene(config, spec)
    folder = output / spec["id"]
    folder.mkdir()
    obj.hide_render = True
    background = render_linear(scene, folder / "background.exr")
    obj.hide_render = False
    source = None
    entries = []
    for index, strength in enumerate(config["strengths"]):
        prefix = folder / f"s{index:03d}"
        displacement = strength * config["max_translation_widths"] * spec["object_width"]
        obj.location.x = spec.get("source_xy", config["source_xy"])[0] + displacement
        obj.visible_shadow = True
        normal = render_linear(scene, prefix.with_name(prefix.name + "_normal.exr"))
        obj.visible_shadow = False
        no_shadow = render_linear(scene, prefix.with_name(prefix.name + "_no_shadow.exr"))
        obj.visible_shadow = True
        coverage = occupancy(scene, obj, floor, light, prefix.with_name(prefix.name + "_id.exr"))
        if source is None:
            source = (normal, no_shadow, coverage)
        path = prefix.with_suffix(".npz")
        np.savez_compressed(
            path,
            source=source[0],
            source_no_shadow=source[1],
            source_occupancy=source[2],
            target=normal,
            target_no_shadow=no_shadow,
            target_occupancy=coverage,
            background=background,
        )
        entries.append(
            dict(
                strength=strength,
                displacement=displacement,
                evidence_path=path.relative_to(output).as_posix(),
                sha256=sha256(path),
            )
        )
    return dict(
        scene=spec,
        group=scene_group(config, spec),
        entries=entries,
        raw_files={p.relative_to(output).as_posix(): sha256(p) for p in folder.glob("*.exr")},
    )


def run(config_path: Path, output: Path) -> None:
    import bpy

    config = load_yaml(config_path)
    validate_scene_config(config)
    numeric_version = ".".join(str(value) for value in bpy.app.version)
    if numeric_version != config["renderer"]["version"]:
        raise ValueError("Installed Blender does not match the pinned scene version")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    index = dict(
        schema_version=1,
        renderer=numeric_version,
        renderer_display=bpy.app.version_string,
        build_hash=bpy.app.build_hash.decode(),
        device="CPU",
        config_sha256=config_digest(config),
        groups=[],
    )
    for spec in config["scenes"]:
        index["groups"].append(render_group(config, spec, output))
        write_json(output / "index.json", index)
    index["seconds"] = time.perf_counter() - started
    write_json(output / "index.json", index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
