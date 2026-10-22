"""Predeclared Dev interventions on existing adapters; OOD diagnostics, not trained ablations."""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from src.backends.flux2 import Flux2Backend
from src.backends.reference_trace import trace_references
from src.cli import build_parser
from src.core.config import ProjectSettings, load_yaml
from src.core.io import read_rows, sha256, write_json
from src.core.paths import CONFIG_ROOT
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.geometry import load_geometry_condition
from src.datasets.holdout import require_dev
from src.datasets.manifest import load_edit_input
from src.datasets.role_controls import RoleControlDataset
from src.inference.runner import run as infer
from src.infra.gpu import require_lease


def interventions(kind: str, original: Image.Image, geometry) -> dict[str, Image.Image]:
    values = np.array(original)
    if kind == "oracle_role":
        neutral = values.copy()
        neutral[..., 2] = 0
        return {
            "cycle_palette": Image.fromarray(np.roll(values, 1, axis=-1)),
            "neutral_p": Image.fromarray(neutral),
        }
    if kind != "dynamic_binary":
        raise ValueError("Use existing RGB Oracle or Dynamic Binary checkpoints")
    return {
        name: Image.fromarray(np.asarray(geometry.maps[key], dtype=np.uint8) * 255).convert("RGB")
        for name, key in (("old_only", "old"), ("new_only", "new"))
    }


def run(args) -> dict:
    require_lease()
    seal = require_dev(args.data_root, args.cohort, args.data_root / "dev/manifest.csv")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = dict(
        status="running",
        scope="OOD fixed-checkpoint interventions; no human ratings",
        lora_sha256=sha256(args.lora),
        cohort_sha256=sha256(args.cohort),
        condition_index_sha256=sha256(args.control_root / "controls.json"),
        results=[],
    )
    started = time.perf_counter()
    try:
        data = RoleControlDataset(args.control_root, args.data_root, args.control)
        rows = read_rows(args.data_root / "dev/manifest.csv")
        if [r["id"] for r in rows] != seal["dev_ids"]:
            raise ValueError("Fixed Dev selection changed")
        groups = list(dict.fromkeys(r["group_id"] for r in rows))[:2]
        backend = Flux2Backend(args.model, str(args.lora))
        inference_args = build_parser().parse_args(
            [
                "infer",
                "--data-root",
                str(args.data_root),
                "--manifest",
                str(args.data_root / "dev/manifest.csv"),
                "--control",
                args.control,
                "--control-root",
                str(args.control_root),
                "--allow-oracle",
                "--model",
                str(args.model),
                "--lora",
                str(args.lora),
                "--limit",
                str(len(rows)),
                "--steps",
                str(args.steps),
                "--guidance",
                str(args.guidance),
                "--seed",
                str(args.seed),
                "--side",
                str(rows[0]["width"]),
                "--output",
                str(args.output / "baseline"),
            ]
        )
        infer(inference_args, backend_factory=lambda *_: backend)
        for row in rows:
            if row["group_id"] not in groups:
                continue
            edit = load_edit_input(args.data_root, row)
            geometry = load_geometry_condition(args.data_root, row)
            original = data.image(row)
            options = GenerationOptions(
                *edit.source.size, args.steps, args.guidance, sample_seed(args.seed, row["id"])
            )
            variants = {"original": original, **interventions(args.control, original, geometry)}
            row_traces = {}
            for name, control in variants.items():
                folder = args.output / row["id"] / name
                folder.mkdir(parents=True)
                control.save(folder / "control.png")
                with trace_references(backend.pipe) as trace:
                    result = backend._generate(edit, [edit.source, control], options, matched=True)
                result.image.save(folder / "prediction.png")
                if len(trace["vae"]) != 2 or trace["attention"]["offsets"] != [0, 10, 20]:
                    raise ValueError("Intervention did not reach both VAE references/attention")
                if name == "original":
                    if sha256(folder / "prediction.png") != sha256(
                        args.output / "baseline" / f"{row['id']}.png"
                    ):
                        raise ValueError("Fixed-checkpoint baseline repeat is not byte identical")
                else:
                    for key in ("source_sha256", "noise_sha256"):
                        if trace["attention"][key] != row_traces["original"]["attention"][key]:
                            raise ValueError("Intervention changed source/noise")
                row_traces[name] = trace
                report["results"].append(
                    dict(
                        id=row["id"],
                        group_id=row["group_id"],
                        variant=name,
                        strength=float(row["strength"]),
                        seed=options.seed,
                        trace=trace,
                        control_sha256=sha256(folder / "control.png"),
                        prediction_sha256=sha256(folder / "prediction.png"),
                        seconds=result.seconds,
                        peak_allocated_gib=result.peak_allocated_gib,
                        peak_reserved_gib=result.peak_reserved_gib,
                    )
                )
                write_json(args.output / "report.json", report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_json(args.output / "report.json", report)
    return report


def main() -> None:
    sampling = load_yaml(CONFIG_ROOT / "experiments/role_geometry.yaml")["inference"]
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "cohort", "control-root", "lora", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--control", choices=["oracle_role", "dynamic_binary"], required=True)
    parser.add_argument("--model", default=ProjectSettings.load().assets["models"]["base"]["path"])
    parser.add_argument("--steps", type=int, default=sampling["steps"])
    parser.add_argument("--guidance", type=float, default=sampling["guidance"])
    parser.add_argument("--seed", type=int, default=sampling["seed"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
