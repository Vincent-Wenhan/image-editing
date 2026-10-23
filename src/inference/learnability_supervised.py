"""Correct supervised probes using existing weights; primary generated PNGs stay untouched."""

import argparse
import os
import time
import traceback
from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.backends.flux2_features import Flux2FeatureBackend
from src.core.config import ProjectSettings
from src.core.io import sha256, write_json
from src.core.metadata import now, tensor_sha, versions
from src.core.random import sample_seed
from src.core.types import EditInput, GenerationOptions
from src.datasets.features import load_features
from src.datasets.manifest import load_edit_input
from src.evaluation.feature_gradient import gradient_probe
from src.evaluation.feature_teacher import TARGET_PREPROCESSING, score_teacher
from src.evaluation.learnability_provenance import verify_inference
from src.inference.learnability import case_policy, checkpoint_pairs, evaluation_rows
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.learnability_study import completed, load_plan
from src.training.lifecycle import attempt, gpu_statistics


def condition_hashes(state):
    ids = state["img_ids"]
    source = (ids[0] if ids.ndim == 3 else ids)[:, 0] != 0
    return dict(
        source_tokens_sha256=tensor_sha(state["hidden_states"][:, source]),
        initial_noise_sha256=tensor_sha(state["hidden_states"][:, ~source]),
        text_sha256=tensor_sha(state["encoder_hidden_states"]),
    )


def source_state(backend, edit, opts):
    """One source-only step captures the same initial state as the original 20-step run."""
    import torch

    captured = {}

    def capture(_model, _args, kwargs):
        if not captured:
            captured.update(
                {
                    k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
            )

    handle = backend.base.pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        backend.base._generate(edit, [edit.source], opts, matched=True)
    finally:
        handle.remove()
    if not captured or float(captured["timestep"].flatten()[0]) != 1:
        raise RuntimeError("Auxiliary source-only capture must start at model time 1")
    return captured


def run(args):
    require_lease()
    import torch

    plan = load_plan(args.study)
    p, root = plan["protocol"], Path(plan["data"])
    job = next(j for j in plan["jobs"] if j["name"] == args.job)
    trained = completed(job)
    if args.smoke:
        if job["phase"] != "capacity":
            raise ValueError("Use the capacity checkpoint for the corrected-probe smoke")
        primary = None
        pairs = [trained["feature_statistics"]["observation"]["checkpoints"][-1]]
    else:
        primary, _ = verify_inference(args.study, plan, job)
        pairs = checkpoint_pairs(trained, job["method"])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(
        status="running",
        created=now(),
        source=source_snapshot(),
        packages=versions(),
        job=job,
        plan_sha256=sha256(args.study / "plan.json"),
        primary_inference_sha256=sha256(Path(job["directory"]) / "inference/report.json")
        if primary
        else None,
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
        training_updates=0,
        test_status="sealed",
        target_preprocessing=TARGET_PREPROCESSING,
        results=[],
        gradients=[],
        source_captures={},
    )
    write_json(args.output / "report.json", report)
    try:
        backend = Flux2FeatureBackend(
            ProjectSettings.load().assets["models"]["base"]["path"],
            pairs[0]["lora"],
            pairs[0]["encoder"]["path"],
        )
        rows = evaluation_rows(plan, job)
        if args.smoke:
            rows = [next(r for r in rows if r["split"] == "train" and float(r["strength"]) > 0)]
        cache = {}
        originals = (
            {(r["update"], r["id"]): r for r in primary["results"] if r["variant"] == "original"}
            if primary
            else {}
        )
        for index, pair in enumerate(pairs):
            if index:
                backend.base.pipe.unload_lora_weights()
                backend.base.pipe.load_lora_weights(pair["lora"])
            backend.base.pipe.transformer.requires_grad_(False)
            encoder, metadata = load_encoder(
                Path(pair["encoder"]["path"]), Path(pair["lora"]), device="cuda"
            )
            encoder.eval().requires_grad_(False)
            if metadata["mode"] != job["method"] or metadata["encoder"] != p["encoder"]:
                raise ValueError("Corrected probe encoder differs from the protocol")
            selected = (
                rows
                if args.smoke or pair["update"] in {0, p["steps"]}
                else [r for r in rows if r["split"] == "train"]
            )
            for row in selected:
                field = load_features(root, row, job["method"])
                tokens = torch.from_numpy(field.tokens())[None].to("cuda")
                grid = field.size[1] // 16, field.size[0] // 16
                seed = sample_seed(p["inference"]["seed"], row["group_id"])
                if row["group_id"] not in cache:
                    edit = EditInput(load_edit_input(root, row).source, p["prompt"])
                    cache[row["group_id"]] = source_state(
                        backend,
                        edit,
                        GenerationOptions(*field.size, 1, p["inference"]["guidance"], seed),
                    )
                state = cache[row["group_id"]]
                identity = condition_hashes(state)
                if primary:
                    previous = originals[(pair["update"], row["id"])]
                    if any(previous[k] != v for k, v in identity.items()):
                        raise RuntimeError(
                            "Corrected probe source/text/noise differs from primary generation"
                        )
                report["source_captures"][row["group_id"]] = identity
                teacher = score_teacher(
                    backend.base.pipe,
                    encoder,
                    tokens,
                    grid,
                    state,
                    root,
                    row,
                    p["teacher_times"],
                    seed,
                )
                report["results"].append(
                    dict(
                        id=row["id"],
                        split=row["split"],
                        group_id=row["group_id"],
                        strength=float(row["strength"]),
                        update=pair["update"],
                        checkpoint_sha256=pair["lora_sha256"],
                        encoder_sha256=pair["encoder"]["sha256"],
                        feature_sha256=field.digest,
                        source_only_capture=True,
                        **identity,
                        teacher=teacher,
                    )
                )
                probe = args.smoke or case_policy(job["method"], pair["update"], row, rows, p)[1]
                if probe:
                    report["gradients"].append(
                        dict(
                            id=row["id"],
                            split=row["split"],
                            update=pair["update"],
                            gradient=gradient_probe(
                                backend.base.pipe,
                                encoder,
                                tokens,
                                grid,
                                state,
                                root,
                                row,
                                p["teacher_times"],
                                seed,
                            ),
                        )
                    )
                write_json(args.output / "report.json", report)
        if primary:
            verify_inference(args.study, plan, job)
        report.update(status="complete", target_is_reference=False)
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
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
