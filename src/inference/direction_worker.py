"""Resident frozen checkpoint, common target-free states, then separate supervision export."""

import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from src.backends.direction_probe import (
    build_state,
    capture_source,
    native_cfg,
    predict,
    state_identity,
)
from src.backends.flux2_features import Flux2FeatureBackend
from src.conditioning.feature_interventions import intervene
from src.conditioning.learnability import research_tokens
from src.core.config import ProjectSettings
from src.core.io import sha256, write_json
from src.core.metadata import now, tensor_sha, versions
from src.core.random import sample_seed
from src.core.types import EditInput, GenerationOptions
from src.datasets.features import load_features
from src.datasets.manifest import load_edit_input
from src.evaluation.direction_supervision import encode_rgb, export_supervision
from src.inference.direction_study import load_direction, state_specs, verify_job
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.lifecycle import attempt, gpu_statistics


def fields_for(tokens, grid, variant, seed):
    return (
        intervene(tokens, grid, "drop_d")
        if variant == "drop_d"
        else research_tokens(tokens, grid, variant, seed)
    )


def archive_velocity(path, outputs, fields, guidance, fusion_arrays):
    import torch

    outputs = dict(
        outputs, cfg_native=native_cfg(outputs["positive"], outputs["negative"], guidance)
    )
    if any(v.dtype != torch.bfloat16 for v in outputs.values()):
        raise TypeError("The frozen pipeline must provide actual BF16 velocities")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{k: v[0].contiguous().view(torch.uint16).cpu().numpy() for k, v in outputs.items()},
        fields=fields[0].cpu().numpy(),
        residual=fusion_arrays["residual"][0].float().cpu().numpy(),
        fused=fusion_arrays["fused"][0].contiguous().view(torch.uint16).cpu().numpy(),
    )
    return dict(
        path=str(path.resolve()),
        sha256=sha256(path),
        velocity_sha256={k: tensor_sha(v) for k, v in outputs.items()},
        fields_sha256=tensor_sha(fields),
    )


def run(args):
    require_lease()
    import torch

    plan = load_direction(args.study)
    job = next(j for j in plan["jobs"] if j["name"] == args.job)
    verify_job(job)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    report = dict(
        status="running",
        created=now(),
        source=source_snapshot(),
        packages=versions(),
        job=job,
        plan_sha256=sha256(args.study / "plan.json"),
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
        training_updates=0,
        smoke=args.smoke,
        test_status="sealed",
        results=[],
        groups={},
    )
    write_json(args.output / "report.json", report)
    try:
        pair, inherited, protocol = job["pair"], plan["inherited"], plan["protocol"]
        backend = Flux2FeatureBackend(
            ProjectSettings.load().assets["models"]["base"]["path"],
            pair["lora"],
            pair["encoder"]["path"],
        )
        model, encoder = backend.base.pipe.transformer, backend.encoder
        expected = copy.deepcopy(inherited["encoder"])
        if job["family"] == "channel_rms":
            expected["normalization"] = "channel_rms"
        if backend.metadata["mode"] != job["method"] or backend.metadata["encoder"] != expected:
            raise ValueError("Frozen encoder mode/config differs from the bound training study")
        model.eval().requires_grad_(False)
        encoder.eval().requires_grad_(False)
        if any(p.requires_grad for p in [*model.parameters(), *encoder.parameters()]):
            raise RuntimeError("Direction diagnostics cannot have trainable parameters")
        before = {k: tensor_sha(v) for k, v in encoder.state_dict().items()}
        rows = plan["rows"]
        if args.smoke:
            group = next(r["group_id"] for r in rows if r["split"] == "train")
            rows = [r for r in rows if r["group_id"] == group and float(r["strength"]) <= 0.5]
        root, settings = Path(plan["data"]), inherited["inference"]
        old = json.loads(
            (Path(job["previous_job"]["directory"]) / "inference/report.json").read_text()
        )
        with torch.inference_mode():
            for group in sorted({r["group_id"] for r in rows}):
                selected = [r for r in rows if r["group_id"] == group]
                row = selected[0]
                edit = EditInput(load_edit_input(root, row).source, inherited["prompt"])
                grid = row["height"], row["width"]
                width, height = int(grid[1]), int(grid[0])
                grid = height // 16, width // 16
                seed = sample_seed(settings["seed"], group)
                states, positions, checks = capture_source(
                    backend,
                    edit,
                    GenerationOptions(width, height, 1, settings["guidance"], seed),
                    grid,
                )
                identity = state_identity(states, positions)
                previous = next(
                    r for r in old["results"] if r["id"] == row["id"] and r["variant"] == "original"
                )
                if any(
                    identity[k] != previous[k]
                    for k in ["source_tokens_sha256", "initial_noise_sha256", "text_sha256"]
                ):
                    raise RuntimeError("Source/prompt/initial noise differs from PR7")
                anchor = encode_rgb(backend.base.pipe, edit.source, "cuda", seed)
                folder = args.output / "inputs" / group
                folder.mkdir(parents=True)
                path = folder / "source.npz"
                np.savez(
                    path,
                    anchor=anchor[0].float().cpu().numpy(),
                    epsilon=states[0]["hidden_states"][:, positions][0].float().cpu().numpy(),
                    positions=positions.cpu().numpy(),
                    img_ids=states[0]["img_ids"].float().cpu().numpy(),
                )
                report["groups"][group] = dict(
                    identity,
                    **checks,
                    anchor_sha256=tensor_sha(anchor),
                    input_path=str(path.resolve()),
                    input_sha256=sha256(path),
                    seed=seed,
                    states=[],
                )
                fields = {r["id"]: load_features(root, r, job["method"]) for r in selected}
                for family, requested in state_specs(protocol):
                    common, x, actual = build_state(states, positions, anchor, requested, family)
                    key = f"{family}_{requested:g}"
                    path = folder / f"{key}.npz"
                    np.savez(path, x=x[0].float().cpu().numpy())
                    descriptor = dict(
                        family=family,
                        time=requested,
                        actual_time=actual,
                        path=str(path.resolve()),
                        sha256=sha256(path),
                        model_state_sha256=tensor_sha(common[0]["hidden_states"]),
                    )
                    report["groups"][group]["states"].append(descriptor)
                    cache = {}
                    for row in selected:
                        tokens = torch.from_numpy(fields[row["id"]].tokens())[None].to("cuda")
                        variants = {}
                        for variant in protocol["variants"]:
                            value = fields_for(tokens, grid, variant, settings["seed"])
                            digest = tensor_sha(value)
                            if digest not in cache:
                                outputs, measurement, fusion_arrays = predict(
                                    model, encoder, value, grid, common, positions
                                )
                                if variant == "zero":
                                    disabled, _, _ = predict(
                                        model, None, value, grid, common, positions
                                    )
                                    if any(
                                        not torch.equal(outputs[b], disabled[b]) for b in outputs
                                    ):
                                        raise RuntimeError("Zero differs from disabled fusion")
                                    if requested == 1 and family == "source_bridge":
                                        native = native_cfg(
                                            outputs["positive"],
                                            outputs["negative"],
                                            settings["guidance"],
                                        )
                                        if (
                                            any(
                                                tensor_sha(outputs[b])
                                                != checks["zero_reference"][b]
                                                for b in outputs
                                            )
                                            or tensor_sha(native)
                                            != checks["zero_reference"]["cfg_native"]
                                        ):
                                            raise RuntimeError(
                                                "Pure-noise zero replay differs from pipeline"
                                            )
                                location = (
                                    args.output / "predictions" / group / key / f"{digest}.npz"
                                )
                                cache[digest] = dict(
                                    archive_velocity(
                                        location,
                                        outputs,
                                        value,
                                        settings["guidance"],
                                        fusion_arrays,
                                    ),
                                    fusion=measurement,
                                )
                            variants[variant] = cache[digest]
                        report["results"].append(
                            dict(
                                id=row["id"],
                                group_id=group,
                                split=row["split"],
                                strength=float(row["strength"]),
                                state=descriptor,
                                variants=variants,
                                feature_sha256=fields[row["id"]].digest,
                            )
                        )
                # All predictions for this group are complete before reading any RGB target.
                report["groups"][group]["predictions_complete_before_supervision"] = True
                supervision = export_supervision(
                    backend.base.pipe, root, selected, args.output / "supervision" / group, seed
                )
                identity_row = next(r for r in selected if float(r["strength"]) == 0)
                with np.load(supervision[identity_row["id"]]["path"], allow_pickle=False) as a:
                    if not np.array_equal(a["target"], anchor[0].float().cpu().numpy()):
                        raise RuntimeError("Identity target latent differs from the source anchor")
                report["groups"][group]["supervision"] = supervision
                write_json(args.output / "report.json", report)
        if before != {k: tensor_sha(v) for k, v in encoder.state_dict().items()}:
            raise RuntimeError("Frozen encoder parameters changed")
        verify_job(job)
        report.update(status="complete", weights_unchanged=True, trainable_parameters=0)
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        errors = {}
        report.update(
            attempt("gpu_statistics", lambda: gpu_statistics(torch, errors), errors) or {}
        )
        if errors:
            report["telemetry_errors"] = errors
        report["seconds"] = time.perf_counter() - start
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
