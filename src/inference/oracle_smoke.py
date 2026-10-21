"""Check oracle reference consumption only; no role-trained quality upper bound is claimed."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from src.backends.flux2 import Flux2Backend
from src.backends.reference_trace import trace_references
from src.conditioning.controls import ControlBundle
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.manifest import load_edit_input
from src.datasets.scene_manifest import load_oracle_condition
from src.datasets.scene_validation import validate_scene_dataset
from src.infra.gpu import require_lease


def run(args) -> None:
    require_lease()
    validate_scene_dataset(args.data_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(
        status="running",
        scope="protocol/sensitivity only; not a quality upper-bound result",
        results={},
        model=json.loads((Path(args.model) / "download.json").read_text()),
    )
    started = time.perf_counter()
    try:
        row = next(
            row
            for row in read_rows(args.data_root / "dev/manifest.csv")
            if float(row["strength"]) > 0
        )
        edit, oracle = (
            load_edit_input(args.data_root, row),
            load_oracle_condition(args.data_root, row),
        )
        report.update(sample_id=row["id"], conditioning=dict(oracle.metadata))
        options = GenerationOptions(
            *edit.source.size, args.steps, args.guidance, sample_seed(args.seed, row["id"])
        )
        labels = np.array(oracle.maps["roles"])
        direct, dependency = labels == 0, labels == 1
        labels[direct], labels[dependency] = 1, 0
        path = args.output / "swapped_td_labels.png"
        Image.fromarray(labels).save(path)
        altered = ControlBundle(
            oracle.kind,
            {"roles": labels},
            dict(oracle.metadata, role_sha256=sha256(path), diagnostic_intervention="swap_T_D"),
        )
        if args.lora:
            report["lora_sha256"] = sha256(args.lora)
        backend = Flux2Backend(args.model, args.lora)
        for name, controls in (("oracle", oracle), ("swapped_td", altered), ("repeat", oracle)):
            with trace_references(backend.pipe) as trace:
                result = backend.edit_oracle(edit, controls, options, allow_oracle=True)
            if (
                len(trace["vae"]) != 2
                or not trace["vae"][1]["binary"]
                or trace["vae"][1]["positive_channels_max"] != 1
                or trace.get("attention", {}).get("offsets") != [0, 10, 20]
                or len(set(trace["attention"]["counts"])) != 1
            ):
                raise ValueError("Oracle role RGB/reference protocol did not reach the model")
            result.image.save(args.output / f"{name}.png")
            report["results"][name] = dict(
                trace=trace,
                seconds=result.seconds,
                image_sha256=sha256(args.output / f"{name}.png"),
                peak_allocated_gib=result.peak_allocated_gib,
                peak_reserved_gib=result.peak_reserved_gib,
            )
            write_json(args.output / "report.json", report)
        original, swapped, repeat = [
            report["results"][key] for key in ("oracle", "swapped_td", "repeat")
        ]
        for key in ("source_sha256", "noise_sha256"):
            if original["trace"]["attention"][key] != swapped["trace"]["attention"][key]:
                raise ValueError("Role intervention changed source or initial noise")
        if (
            original["image_sha256"] != repeat["image_sha256"]
            or original["image_sha256"] == swapped["image_sha256"]
            or original["trace"]["attention"]["control_sha256"]
            == swapped["trace"]["attention"]["control_sha256"]
        ):
            raise ValueError("Role sensitivity or repeated sampling check failed")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_json(args.output / "report.json", report)


def main() -> None:
    settings = ProjectSettings.load()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=settings.assets["models"]["base"]["path"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lora", help="Optional completed role adapter for reload protocol check")
    parser.add_argument(
        "--guidance", type=float, default=settings.process["sample"]["guidance_scale"]
    )
    parser.add_argument("--seed", type=int, default=settings.process["sample"]["seed"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
