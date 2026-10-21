"""Shared sample selection and incremental records, independent of model internals."""

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from PIL import Image

from src.backends.base import EditingBackend
from src.backends.flux2 import Flux2Backend
from src.conditioning.controls import ControlBundle
from src.conditioning.static import MASK_KINDS
from src.core.config import ProjectSettings
from src.core.images import dimensions
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import config_digest, config_snapshot, now, versions
from src.core.paths import data_path
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.controls import ControlDataset
from src.datasets.manifest import load_edit_input
from src.infra.gpu import require_lease

logger = logging.getLogger(__name__)


def run(args, *, backend_factory: Callable[..., EditingBackend] = Flux2Backend) -> None:
    require_lease()
    if args.limit < 1 or args.steps < 1 or args.side < 16:
        raise ValueError("Positive limit/steps and side >=16 required")
    rows = read_rows(args.manifest)[: args.limit]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new run directory: {output}")
    output.mkdir(parents=True)
    model = Path(args.model)
    provenance = json.loads((model / "download.json").read_text())
    run_info = dict(
        created=now(),
        status="running",
        config=config_snapshot(args),
        model=provenance,
        packages=versions(),
        manifest_sha256=sha256(args.manifest),
        results=[],
    )
    run_info["config_sha256"] = config_digest(run_info["config"])
    if args.lora:
        run_info["lora_sha256"] = sha256(args.lora)
    write_json(output / "run.json", run_info)
    try:
        kind = getattr(args, "control", "none")
        control_root = getattr(args, "control_root", None)
        if (kind == "none") != (control_root is None):
            raise ValueError("Mask inference requires both --control and --control-root")
        condition_data = (
            ControlDataset(Path(control_root), Path(args.data_root), kind) if control_root else None
        )
        if condition_data:
            run_info["conditioning"] = condition_data.metadata()
            if any(row != condition_data.rows[row["split"]].get(row["id"]) for row in rows):
                raise ValueError("Inference manifest differs from prepared controls")
        backend = backend_factory(model, args.lora)
        for row in rows:
            if sha256(data_path(args.data_root, row["source_path"])) != row["source_sha256"]:
                raise ValueError(f"Source checksum mismatch: {row['id']}")
            edit_input = load_edit_input(args.data_root, row)
            width, height = dimensions(edit_input.source.size, args.side)
            edit_input = replace(
                edit_input,
                source=edit_input.source.resize((width, height), Image.Resampling.LANCZOS),
            )
            seed = sample_seed(args.seed, row["id"])
            controls = (
                condition_data.bundle(row, (width, height)) if condition_data else ControlBundle()
            )
            controls.require_inference_safe()
            prediction = backend.edit(
                edit_input,
                controls,
                GenerationOptions(width, height, args.steps, args.guidance, seed),
            )
            if prediction.image.size != (width, height):
                raise ValueError(f"Backend output dimensions differ for {row['id']}")
            prediction.image.save(output / f"{row['id']}.png")
            run_info["results"].append(
                dict(
                    id=row["id"],
                    seed=seed,
                    width=width,
                    height=height,
                    seconds=prediction.seconds,
                    source_sha256=row["source_sha256"],
                    prompt=row["instruction"],
                    peak_allocated_gib=prediction.peak_allocated_gib,
                    peak_reserved_gib=prediction.peak_reserved_gib,
                )
            )
            if condition_data:
                run_info["results"][-1]["conditioning"] = dict(controls.metadata)
            write_json(output / "run.json", run_info)
            logger.info("%s: %.1fs", row["id"], prediction.seconds)
        run_info["status"] = "complete"
    except BaseException as error:
        run_info.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(output / "run.json", run_info)


def configure(parser) -> None:
    settings = ProjectSettings.load()
    sample = settings.process["sample"]
    parser.add_argument("--model", default=settings.assets["models"]["base"]["path"])
    parser.add_argument("--manifest", default=settings.manifest)
    parser.add_argument("--data-root", default=settings.dataset["root"])
    parser.add_argument("--output", default="outputs/base_inference")
    parser.add_argument("--limit", type=int, default=settings.experiment["inference_limit"])
    parser.add_argument("--side", type=int, default=sample["width"])
    parser.add_argument("--steps", type=int, default=sample["sample_steps"])
    parser.add_argument("--guidance", type=float, default=sample["guidance_scale"])
    parser.add_argument("--seed", type=int, default=sample["seed"])
    parser.add_argument("--lora")
    parser.add_argument("--control", choices=["none", *MASK_KINDS], default="none")
    parser.add_argument("--control-root", help="Prepared, hashed mask condition directory")
    parser.set_defaults(func=run)
