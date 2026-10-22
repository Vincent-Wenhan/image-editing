"""Trace matched single/constant references and the opt-in BF16 preprocessing ablation."""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from src.backends.flux2 import Flux2Backend
from src.backends.reference_trace import trace_references
from src.conditioning.controls import ControlBundle, ControlKind
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import now
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.manifest import load_edit_input
from src.infra.gpu import require_lease


def run(args) -> None:
    require_lease()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status="running", created=now(), results={})
    try:
        row = read_rows(args.data_root / "dev/manifest.csv")[0]
        edit = load_edit_input(args.data_root, row)
        report.update(
            sample_id=row["id"],
            source_sha256=sha256(args.data_root / row["source_path"]),
            lora_sha256=sha256(args.lora),
            steps=args.steps,
        )
        options = GenerationOptions(
            *edit.source.size, args.steps, args.guidance, sample_seed(args.seed, row["id"])
        )
        matched = ControlBundle(metadata={"reference_preprocessing": "training"})
        constant = ControlBundle(
            ControlKind.CONSTANT_MASK, {"mask": Image.new("L", edit.source.size)}
        )
        backend = Flux2Backend(args.model, str(args.lora))
        images = {}
        for name, controls in (
            ("legacy", ControlBundle()),
            ("training", matched),
            ("constant", constant),
            ("training_repeat", matched),
        ):
            with trace_references(backend.pipe) as trace:
                result = backend.edit(edit, controls, options)
            expected = [0, 10, 20] if name == "constant" else [0, 10]
            if (
                len(trace["vae"]) != len(expected) - 1
                or trace.get("attention", {}).get("offsets") != expected
                or len(set(trace["attention"]["counts"])) != 1
            ):
                raise ValueError("Reference/token structure differs from the declared protocol")
            result.image.save(args.output / f"{name}.png")
            images[name] = np.asarray(result.image).astype(np.float32) / 255
            report["results"][name] = dict(
                trace=trace,
                seconds=result.seconds,
                image_sha256=sha256(args.output / f"{name}.png"),
                peak_allocated_gib=result.peak_allocated_gib,
                peak_reserved_gib=result.peak_reserved_gib,
            )
            write_json(args.output / "report.json", report)
        results = report["results"]
        for name in ("legacy", "constant", "training_repeat"):
            if (
                results[name]["trace"]["attention"]["noise_sha256"]
                != results["training"]["trace"]["attention"]["noise_sha256"]
            ):
                raise ValueError("Ablation changed the initial target noise")
        if (
            results["constant"]["trace"]["attention"]["source_sha256"]
            != results["training"]["trace"]["attention"]["source_sha256"]
        ):
            raise ValueError("Matched references changed source tokens")
        if results["legacy"]["trace"]["vae"][0] == results["training"]["trace"]["vae"][0]:
            raise ValueError("Selected sample did not exercise the BF16 cast difference")
        if results["constant"]["trace"]["vae"][1]["max"] != -1:
            raise ValueError("Constant RGB was not -1 at the VAE input")
        if results["training"]["image_sha256"] != results["training_repeat"]["image_sha256"]:
            raise ValueError("Fixed explicit preprocessing is not deterministic")
        report.update(
            status="complete",
            legacy_vs_training_mae=float(np.abs(images["legacy"] - images["training"]).mean()),
        )
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
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=settings.assets["models"]["base"]["path"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--guidance", type=float, default=settings.process["sample"]["guidance_scale"]
    )
    parser.add_argument("--seed", type=int, default=settings.process["sample"]["seed"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
