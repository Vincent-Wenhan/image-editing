"""Real resident-GPU intervention images and fixed-state numerical probes."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from src.backends.feature_fusion import feature_fusion
from src.backends.feature_probe import StateCapture, replay_states
from src.backends.flux2_features import Flux2FeatureBackend
from src.conditioning.feature_interventions import intervention_encoder
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import now, versions
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.features import load_features
from src.datasets.manifest import load_edit_input
from src.inference.feature_diagnostic_study import load_plan
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.lifecycle import attempt, gpu_statistics


def run(args):
    require_lease()
    import torch

    plan = load_plan(args.study)
    p, root = plan["protocol"], Path(plan["data"])
    checkpoint = next(
        c
        for c in plan["checkpoints"]
        if c["seed"] == args.train_seed and c["method"] == "full_role"
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(
        status="running",
        created=now(),
        source=source_snapshot(),
        packages=versions(),
        phase=args.phase,
        plan_sha256=sha256(args.study / "plan.json"),
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
        checkpoint=checkpoint,
        protocol=p,
        results=[],
        fixed_states=[],
        training_updates=0,
    )
    write_json(args.output / "report.json", report)
    try:
        model_path = ProjectSettings.load().assets["models"]["base"]["path"]
        backend = Flux2FeatureBackend(model_path, checkpoint["lora"], checkpoint["encoder"])
        encoder, model = backend.encoder, backend.base.pipe.transformer
        encoder.requires_grad_(False)
        model.requires_grad_(False)
        rows = {r["id"]: r for r in read_rows(root / "dev/manifest.csv")}
        smoke = args.phase == "smoke"
        selected = [p["smoke"]["sample"]] if smoke else p["diagnostic_samples"]
        variants = p["smoke"]["variants"] if smoke else p["variants"]
        steps = p["smoke"]["steps"] if smoke else p["inference"]["steps"]
        previous = json.loads((Path(checkpoint["previous_predictions"]) / "run.json").read_text())
        if previous["lora_sha256"] != checkpoint["lora_sha256"]:
            raise ValueError("Original predictions use a different checkpoint")
        for identifier in selected:
            row = rows[identifier]
            field = load_features(root, row, "full_role")
            tokens = torch.from_numpy(field.tokens())[None].to("cuda")
            grid = (field.size[1] // 16, field.size[0] // 16)
            edit = load_edit_input(root, row)
            options = GenerationOptions(
                *field.size,
                steps,
                p["inference"]["guidance"],
                sample_seed(p["inference"]["seed"], identifier),
            )
            captured = StateCapture([0, steps - 1] if smoke else p["probe_steps"])
            for name in variants:
                wrapped = intervention_encoder(
                    encoder, name, shift=p["shift_tokens"], seed=p["shuffle_seed"]
                )
                handle = (
                    model.register_forward_pre_hook(captured, with_kwargs=True)
                    if name == "original"
                    else None
                )
                try:
                    with feature_fusion(model, wrapped, tokens, grid) as stats:
                        generated = backend.base._generate(
                            edit, [edit.source], options, matched=True
                        )
                finally:
                    if handle is not None:
                        handle.remove()
                folder = args.output / identifier / name
                folder.mkdir(parents=True)
                generated.image.save(folder / "prediction.png")
                if name == "original" and not smoke:
                    original = Path(checkpoint["previous_predictions"]) / f"{identifier}.png"
                    if sha256(original) != sha256(folder / "prediction.png"):
                        raise RuntimeError(
                            "Reloaded baseline does not match the original fixed PNG"
                        )
                report["results"].append(
                    dict(
                        id=identifier,
                        group_id=row["group_id"],
                        strength=float(row["strength"]),
                        variant=name,
                        seed=options.seed,
                        source_sha256=row["source_sha256"],
                        feature_sha256=field.digest,
                        prediction_sha256=sha256(folder / "prediction.png"),
                        seconds=generated.seconds,
                        peak_allocated_gib=generated.peak_allocated_gib,
                        peak_reserved_gib=generated.peak_reserved_gib,
                        fusion=stats,
                    )
                )
                write_json(args.output / "report.json", report)
            captured.validate(steps)
            probe_variants = variants if not smoke else ["original", "zero", "swap_old_new"]
            report["fixed_states"].append(
                dict(
                    id=identifier,
                    feature_sha256=field.digest,
                    probes=replay_states(model, encoder, tokens, grid, captured, probe_variants, p),
                )
            )
            if smoke:
                with feature_fusion(model, encoder, tokens, grid):
                    repeated = backend.base._generate(edit, [edit.source], options, matched=True)
                repeated.image.save(args.output / "repeat.png")
                if sha256(args.output / "repeat.png") != sha256(
                    args.output / identifier / "original/prediction.png"
                ):
                    raise RuntimeError("Diagnostic smoke is not deterministic")
                report["repeat_exact"] = True
            del captured
            write_json(args.output / "report.json", report)
        if (
            sha256(checkpoint["lora"]) != checkpoint["lora_sha256"]
            or sha256(checkpoint["encoder"]) != checkpoint["encoder_sha256"]
        ):
            raise RuntimeError("Frozen checkpoint bytes changed")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        errors = {}
        report.update(
            attempt("gpu_statistics", lambda: gpu_statistics(torch, errors), errors) or {}
        )
        report["seconds"] = time.perf_counter() - started
        if errors:
            report["telemetry_errors"] = errors
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "diagnose"], required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
