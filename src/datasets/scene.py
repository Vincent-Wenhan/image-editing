"""Non-overwriting CPU scene generation and read-only verification CLI."""

import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from src.core.config import load_yaml
from src.core.io import sha256, write_json
from src.core.metadata import now
from src.core.paths import CONFIG_ROOT
from src.datasets.scene_config import validate_scene_config
from src.datasets.scene_pairs import compile_pairs
from src.datasets.scene_validation import validate_scene_dataset


def generate(config_path: Path, output: Path) -> dict:
    config = load_yaml(config_path)
    validate_scene_config(config)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config_copy = output / "scene_config.yaml"
    shutil.copyfile(config_path, config_copy)
    started = time.perf_counter()
    report = dict(
        status="running",
        created=now(),
        device="CPU",
        config_sha256=sha256(config_copy),
        generator_files={
            path.name: sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("blender_render.py"),
                Path(__file__).with_name("blender_scene.py"),
                Path(__file__).with_name("scene_pairs.py"),
                Path(__file__).with_name("scene_validation.py"),
                Path(__file__).with_name("scene_config.py"),
                Path(__file__).parents[1] / "conditioning/role_truth.py",
            )
        },
    )
    write_json(output / "generation.json", report)
    try:
        argv = [
            sys.executable,
            "-m",
            "src.datasets.blender_render",
            "--config",
            str(config_copy),
            "--output",
            str(output / "raw"),
        ]
        report["argv"] = argv
        with (output / "render.log").open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
        compile_pairs(output, config)
        validation = validate_scene_dataset(output)
        write_json(output / "validation.json", validation)
        report.update(
            status="complete", validation=validation, dataset_sha256=sha256(output / "dataset.json")
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_json(output / "generation.json", report)
    return report


def run_generate(args) -> None:
    print(generate(Path(args.config).resolve(), Path(args.output).resolve()))


def run_validate(args) -> None:
    print(validate_scene_dataset(Path(args.root)))


def configure(parser) -> None:
    actions = parser.add_subparsers(dest="scene_action", required=True)
    generate_parser = actions.add_parser("generate")
    generate_parser.add_argument(
        "--config", default=str(CONFIG_ROOT / "datasets/blender_translation.yaml")
    )
    generate_parser.add_argument("--output", required=True)
    generate_parser.set_defaults(func=run_generate)
    validate_parser = actions.add_parser("validate")
    validate_parser.add_argument("--root", required=True)
    validate_parser.set_defaults(func=run_validate)
