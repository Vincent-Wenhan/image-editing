"""Frozen residual repair smoke and four-arm Dev comparison, with immutable provenance."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.backends.feature_fusion import feature_fusion
from src.backends.feature_probe import StateCapture, replay_states
from src.backends.flux2_features import Flux2FeatureBackend
from src.conditioning.feature_residuals import cached_encoders
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import config_digest, now, versions
from src.core.random import sample_seed
from src.core.types import GenerationOptions
from src.datasets.features import load_features
from src.datasets.manifest import load_edit_input
from src.evaluation.feature_residual_provenance import save_conditions
from src.inference.feature_repair_study import load_comparison
from src.inference.feature_runner import feature_identity
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.lifecycle import attempt, gpu_statistics


def run_record(plan, checkpoint, backend, model_path, variant, output, options, limit):
    root, p = Path(plan["data"]), plan["protocol"]
    config = dict(
        data_root=str(root),
        manifest=str(root / "dev/manifest.csv"),
        cohort=plan["cohort"],
        model=str(model_path),
        lora=checkpoint["lora"],
        encoder=checkpoint["encoder"],
        feature_mode="full_role",
        allow_oracle=True,
        output=str(output.resolve()),
        limit=limit,
        side=options.width,
        steps=options.steps,
        guidance=options.guidance,
        seed=p["inference"]["seed"],
        control="none",
        reference_preprocessing="training",
        residual_intervention=variant,
    )
    return dict(
        created=now(),
        status="running",
        config=config,
        config_sha256=config_digest(config),
        model=json.loads((Path(model_path) / "download.json").read_text()),
        packages=versions(),
        manifest_sha256=sha256(root / "dev/manifest.csv"),
        lora_sha256=checkpoint["lora_sha256"],
        role_features=dict(
            path=checkpoint["encoder"],
            sha256=checkpoint["encoder_sha256"],
            mode="full_role",
            cohort=plan["cohort"],
            cohort_sha256=plan["cohort_sha256"],
            image_order="source",
            feature_tokens=(options.width // 16) ** 2,
            checkpoint_metadata=backend.metadata,
        ),
        reference_preprocessing="training",
        residual_intervention=variant,
        results=[],
    )


def run(args):
    require_lease()
    import torch

    plan, comparison = load_comparison(args.study)
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
        comparison_sha256=sha256(args.study / "comparison.json"),
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
        checkpoint=checkpoint,
        protocol=p,
        recipe=comparison["recipe"],
        results=[],
        fixed_states=[],
        checks={},
        training_updates=0,
    )
    records = {}
    write_json(args.output / "report.json", report)
    try:
        model_path = ProjectSettings.load().assets["models"]["base"]["path"]
        backend = Flux2FeatureBackend(model_path, checkpoint["lora"], checkpoint["encoder"])
        encoder, model = backend.encoder, backend.base.pipe.transformer
        encoder.requires_grad_(False)
        model.requires_grad_(False)
        rows = read_rows(root / "dev/manifest.csv")
        smoke = args.phase == "repair_smoke"
        if smoke:
            rows = [r for r in rows if r["id"] in comparison["recipe"]["smoke_samples"]]
        steps = p["smoke"]["steps"] if smoke else p["inference"]["steps"]
        for row in rows:
            identifier = row["id"]
            field = load_features(root, row, "full_role")
            tokens = torch.from_numpy(field.tokens())[None].to("cuda")
            grid = field.size[1] // 16, field.size[0] // 16
            edit = load_edit_input(root, row)
            options = GenerationOptions(
                *field.size,
                steps,
                p["inference"]["guidance"],
                sample_seed(p["inference"]["seed"], identifier),
            )
            torch.cuda.synchronize()
            cached_start = time.perf_counter()
            wrappers, arrays, metadata = cached_encoders(encoder, tokens, grid, p["shuffle_seed"])
            torch.cuda.synchronize()
            metadata["precompute_seconds"] = time.perf_counter() - cached_start
            conditions = save_conditions(
                args.output / "conditions" / identifier, wrappers, arrays, metadata
            )
            captured = StateCapture([0, steps - 1])
            digests = {}
            for name, wrapped in wrappers.items():
                folder = args.output / name
                folder.mkdir(exist_ok=True)
                if name not in records:
                    records[name] = run_record(
                        plan, checkpoint, backend, model_path, name, folder, options, len(rows)
                    )
                pre = (
                    model.register_forward_pre_hook(captured, with_kwargs=True)
                    if smoke and name == "original"
                    else None
                )
                post = model.register_forward_hook(captured.capture_output) if pre else None
                try:
                    with feature_fusion(model, wrapped, tokens, grid) as stats:
                        generated = backend.base._generate(
                            edit, [edit.source], options, matched=True
                        )
                finally:
                    if pre:
                        pre.remove()
                        post.remove()
                path = folder / f"{identifier}.png"
                generated.image.save(path)
                digests[name] = sha256(path)
                item = dict(
                    id=identifier,
                    group_id=row["group_id"],
                    strength=float(row["strength"]),
                    variant=name,
                    seed=options.seed,
                    width=options.width,
                    height=options.height,
                    prompt=row["instruction"],
                    source_sha256=row["source_sha256"],
                    feature_sha256=field.digest,
                    role_features=feature_identity(field),
                    prediction_sha256=sha256(path),
                    prediction_path=f"{name}/{identifier}.png",
                    seconds=generated.seconds,
                    peak_allocated_gib=generated.peak_allocated_gib,
                    peak_reserved_gib=generated.peak_reserved_gib,
                    fusion=stats,
                    residual_condition=conditions[name],
                )
                if name == "original" and not smoke:
                    if sha256(Path(checkpoint["previous_predictions"]) / path.name) != sha256(path):
                        raise RuntimeError("Cached original differs from the original fixed PNG")
                report["results"].append(item)
                records[name]["results"].append(item)
                write_json(folder / "run.json", records[name])
                write_json(args.output / "report.json", report)
            if smoke:
                captured.validate(steps)
                report["fixed_states"].append(
                    dict(
                        id=identifier,
                        probes=replay_states(
                            model,
                            encoder,
                            tokens,
                            grid,
                            captured,
                            list(wrappers),
                            p,
                            encoders=wrappers,
                        ),
                    )
                )
                with feature_fusion(model, encoder, tokens, grid):
                    uncached = backend.base._generate(edit, [edit.source], options, matched=True)
                path = args.output / f"{identifier}_uncached.png"
                uncached.image.save(path)
                if sha256(path) != digests["original"]:
                    raise RuntimeError("Cached original differs from the live encoder")
                if float(row["strength"]) == 0:
                    if digests["centered"] != digests["zero"] or metadata["centered_rms"] != 0:
                        raise RuntimeError("Identity-centered condition must equal zero")
                    report["checks"]["identity_zero_exact"] = True
                else:
                    reloaded, _ = load_encoder(
                        Path(checkpoint["encoder"]), Path(checkpoint["lora"]), device="cuda"
                    )
                    fresh, _, _ = cached_encoders(reloaded, tokens, grid, p["shuffle_seed"])
                    with feature_fusion(model, fresh["centered"], tokens, grid):
                        repeated = backend.base._generate(
                            edit, [edit.source], options, matched=True
                        )
                    path = args.output / "reloaded.png"
                    repeated.image.save(path)
                    if (
                        sha256(path) != digests["centered"]
                        or digests["centered"] == digests["zero"]
                    ):
                        raise RuntimeError("Repair reload/effective-condition check failed")
                    report["checks"].update(reload_exact=True, edit_effective=True)
                report["checks"].update(
                    original_cached_exact=True, fixed_replay_exact=True, zero_disabled_exact=True
                )
            write_json(args.output / "report.json", report)
        load_comparison(args.study)
        report["checks"]["original_all_exact"] = True
        report["status"] = "complete"
        for record in records.values():
            record["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        for record in records.values():
            record.update(status="failed", error=repr(error))
        raise
    finally:
        errors = {}
        report.update(
            attempt("gpu_statistics", lambda: gpu_statistics(torch, errors), errors) or {}
        )
        report["seconds"] = time.perf_counter() - started
        if errors:
            report["telemetry_errors"] = errors
        for name, record in records.items():
            write_json(args.output / name / "run.json", record)
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--phase", choices=["repair_smoke", "compare"], required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
