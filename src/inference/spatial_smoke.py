"""Trace real VAE/attention inputs and repeat controlled ablations on one resident model."""

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src.backends.flux2 import Flux2Backend
from src.backends.flux2_diagnostics import encoding_parity, tensor_digest
from src.conditioning.controls import ControlBundle, ControlKind
from src.core.config import ProjectSettings, load_yaml
from src.core.io import sha256, write_json
from src.core.metadata import now
from src.core.paths import CONFIG_ROOT, PROJECT_ROOT
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.controls import ControlDataset
from src.datasets.manifest import load_edit_input
from src.infra.gpu import require_lease
from src.infra.process import command
from src.training.flux2_compat import SUPPORTED_TRAINER_REVISION


def run(args) -> None:
    require_lease()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(created=now(), status="running", results={})
    write_json(args.output / "report.json", report)
    try:
        import torch

        revision = command(["git", "rev-parse", "HEAD"])
        if (
            Path.cwd().resolve() != (PROJECT_ROOT / "external/ai-toolkit").resolve()
            or revision != SUPPORTED_TRAINER_REVISION
        ):
            raise ValueError("Encoding smoke must run in the pinned ai-toolkit cwd")
        static = ControlDataset(args.static_root, args.data_root, "static_mask")
        shuffled = ControlDataset(args.shuffled_root, args.data_root, "shuffled_mask")
        row = next(
            row
            for row in static.rows["dev"].values()
            if np.asarray(static.image(row)).any()
            and not np.array_equal(static.image(row), shuffled.image(row))
        )
        edit = load_edit_input(args.data_root, row)
        options = GenerationOptions(
            *edit.source.size, args.steps, args.guidance, sample_seed(args.seed, row["id"])
        )
        base = static.bundle(row, edit.source.size)
        wrong = shuffled.bundle(row, edit.source.size)
        zero = ControlBundle(
            ControlKind.STATIC_MASK,
            {"mask": np.zeros((edit.source.height, edit.source.width), dtype=np.float32)},
        )
        report.update(
            sample_id=row["id"],
            seed=options.seed,
            steps=options.steps,
            trainer_revision=revision,
            lora_sha256=sha256(args.lora),
            source_sha256=row["source_sha256"],
            controls=[static.metadata(), shuffled.metadata()],
        )
        backend = Flux2Backend(args.model, str(args.lora))
        report["encoding"] = encoding_parity(
            backend.pipe, [edit.source, static.image(row)], args.vae
        )
        torch.cuda.reset_peak_memory_stats()
        for name, bundle in (
            ("static", base),
            ("zero", zero),
            ("shuffled", wrong),
            ("static_repeat", base),
        ):
            trace = dict(vae_inputs=[])

            def encoder_hook(_module, inputs, trace=trace):
                value = inputs[0]
                trace["vae_inputs"].append(
                    dict(
                        shape=list(value.shape),
                        min=value.min().item(),
                        max=value.max().item(),
                        sha256=tensor_digest(value),
                    )
                )
                if len(trace["vae_inputs"]) == 2 and not ((value == -1) | (value == 1)).all():
                    raise ValueError("Mask is not binary at the actual VAE encoder input")

            def attention_hook(_module, _args, kwargs, trace=trace):
                if "attention" in trace:
                    return
                ids, hidden = kwargs["img_ids"][0], kwargs["hidden_states"]
                offsets, counts = torch.unique(ids[:, 0], return_counts=True)
                if offsets.tolist() != [0, 10, 20] or len(set(counts.tolist())) != 1:
                    raise ValueError("Attention did not receive [noisy_target, source, mask]")
                trace["attention"] = dict(
                    offsets=offsets.tolist(),
                    counts=counts.tolist(),
                    source_sha256=tensor_digest(hidden[:, ids[:, 0] == 10]),
                    mask_sha256=tensor_digest(hidden[:, ids[:, 0] == 20]),
                    noise_sha256=tensor_digest(hidden[:, ids[:, 0] == 0]),
                )

            handles = [
                backend.pipe.vae.encoder.register_forward_pre_hook(encoder_hook),
                backend.pipe.transformer.register_forward_pre_hook(
                    attention_hook, with_kwargs=True
                ),
            ]
            try:
                output = backend.edit(edit, bundle, options)
            finally:
                for handle in handles:
                    handle.remove()
            if len(trace["vae_inputs"]) != 2 or "attention" not in trace:
                raise ValueError(
                    "Expected exactly two encoded image references and joint attention"
                )
            output.image.save(args.output / f"{name}.png")
            report["results"][name] = dict(
                trace=trace,
                seconds=output.seconds,
                image_sha256=sha256(args.output / f"{name}.png"),
                peak_allocated_gib=output.peak_allocated_gib,
                peak_reserved_gib=output.peak_reserved_gib,
            )
            write_json(args.output / "report.json", report)
        results = report["results"]
        if results["static"]["image_sha256"] != results["static_repeat"]["image_sha256"]:
            raise ValueError("Repeated fixed-mask sampling was not deterministic")
        for name in ("zero", "shuffled"):
            left, right = results["static"], results[name]
            for key in ("source_sha256", "noise_sha256"):
                if left["trace"]["attention"][key] != right["trace"]["attention"][key]:
                    raise ValueError("Ablation changed source tokens or initial noise")
            if (
                left["trace"]["attention"]["mask_sha256"]
                == right["trace"]["attention"]["mask_sha256"]
                or left["image_sha256"] == right["image_sha256"]
            ):
                raise ValueError("Mask change did not affect reference tokens and generated output")
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
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--shuffled-root", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / settings.assets["models"]["base"]["path"]
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=PROJECT_ROOT / settings.assets["models"]["training_vae"]["path"] / "ae.safetensors",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=load_yaml(CONFIG_ROOT / "experiments/stage1.yaml")["smoke"]["sampling_steps"],
    )
    parser.add_argument(
        "--guidance", type=float, default=settings.process["sample"]["guidance_scale"]
    )
    parser.add_argument("--seed", type=int, default=settings.process["sample"]["seed"])
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    trainer = (PROJECT_ROOT / "external/ai-toolkit").resolve()
    if Path.cwd().resolve() != trainer:
        argv = [sys.executable, "-m", "src.inference.spatial_smoke"]
        for key, value in vars(args).items():
            argv.extend([f"--{key.replace('_', '-')}", str(value)])
        subprocess.run(argv, cwd=trainer, check=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
