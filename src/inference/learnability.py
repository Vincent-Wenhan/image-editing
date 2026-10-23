"""Controlled Train/Dev generation; teacher supervision is delegated after generation."""

import argparse
import os
import time
import traceback
from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.backends.feature_fusion import feature_fusion
from src.backends.feature_probe import StateCapture, replay_states
from src.backends.flux2_features import Flux2FeatureBackend
from src.conditioning.learnability import RESEARCH_VARIANTS, research_encoder, research_tokens
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import now, tensor_sha, versions
from src.core.random import sample_seed
from src.core.types import EditInput, GenerationOptions
from src.datasets.features import load_features
from src.datasets.manifest import load_edit_input
from src.datasets.research_view import verify_view
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.learnability_study import completed, load_plan
from src.training.lifecycle import attempt, gpu_statistics


def evaluation_rows(plan, job):
    root = Path(plan["data"])
    view = verify_view(Path(job["directory"]) / "view/view.json", root=root)
    return [
        r for r in read_rows(root / "train/manifest.csv") if r["id"] in view["ids"]
    ] + read_rows(root / "dev/manifest.csv")


def diagnostic_ids(rows):
    """Predeclared first identity plus first full-strength case from two scene groups."""
    groups = sorted({r["group_id"] for r in rows})[:2]
    first = [r for r in rows if r["group_id"] == groups[0] and float(r["strength"]) == 0]
    full = [
        r for group in groups for r in rows if r["group_id"] == group and float(r["strength"]) == 1
    ]
    return {r["id"] for r in [*first, *full]}


def checkpoint_pairs(trained, method):
    saved = trained["feature_statistics"]["observation"]["checkpoints"]
    steps = [0, 20, 60, trained["steps"]] if method == "full_role" else [trained["steps"]]
    result = []
    for update in steps:
        options = [p for p in saved if p["update"] == update]
        if len(options) != 1:
            raise ValueError("Missing or duplicate exact-update checkpoint pair")
        pair = options[0]
        if (
            sha256(pair["lora"]) != pair["lora_sha256"]
            or sha256(pair["encoder"]["path"]) != pair["encoder"]["sha256"]
        ):
            raise ValueError("Overfit checkpoint pair SHA256 changed")
        result.append(pair)
    return result


def case_policy(method, update, row, all_rows, protocol):
    names, probe = ["original"], False
    if update == protocol["steps"]:
        split_rows = [r for r in all_rows if r["split"] == row["split"]]
        selected = row["id"] in diagnostic_ids(split_rows)
        probe = method == "full_role" and row["id"] == next(
            r["id"] for r in split_rows if float(r["strength"]) > 0
        )
        if method == "full_role" and (row["split"] == "train" or selected or probe):
            names = list(RESEARCH_VARIANTS)
        elif row["split"] == "train":
            names += ["zero", "shuffle"]
    return names, probe


def generate_case(backend, encoder, edit, tokens, grid, opts, names, folder, probe, protocol):
    import torch

    model = backend.base.pipe.transformer
    wrappers = {
        name: research_encoder(encoder, name, protocol["inference"]["seed"]) for name in names
    }
    first_state, results = {}, []
    captured = StateCapture(sorted({0, opts.steps // 2 - 1, opts.steps - 1})) if probe else None
    for name in names:
        handles = []
        if name == "original":

            def first(_m, _args, kwargs):
                if not first_state:
                    first_state.update(
                        {
                            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                            for k, v in kwargs.items()
                        }
                    )

            handles.append(model.register_forward_pre_hook(first, with_kwargs=True))
            if captured:
                handles += [
                    model.register_forward_pre_hook(captured, with_kwargs=True),
                    model.register_forward_hook(captured.capture_output),
                ]
        try:
            with feature_fusion(model, wrappers[name], tokens, grid) as fusion:
                generated = backend.base._generate(edit, [edit.source], opts, matched=True)
        finally:
            for handle in handles:
                handle.remove()
        path = folder / name / "prediction.png"
        path.parent.mkdir(parents=True)
        if generated.image.mode != "RGB" or generated.image.size != (opts.width, opts.height):
            raise RuntimeError("Controlled prediction mode/geometry changed")
        generated.image.save(path)
        results.append(
            dict(
                variant=name,
                path=str(path.resolve()),
                prediction_sha256=sha256(path),
                seconds=generated.seconds,
                fusion=fusion,
            )
        )
    probes = None
    if captured:
        captured.validate(opts.steps)
        probes = replay_states(
            model,
            encoder,
            tokens,
            grid,
            captured,
            names,
            protocol,
            encoders=wrappers,
            measure_layers=True,
        )
    return results, first_state, probes


def run(args):
    require_lease()
    import torch

    plan = load_plan(args.study)
    p, root = plan["protocol"], Path(plan["data"])
    job = next(j for j in plan["jobs"] if j["name"] == args.job and j["phase"] == "train")
    trained = completed(job)
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
        plan_sha256=sha256(args.study / "plan.json"),
        job=job,
        training_report_sha256=sha256(Path(job["directory"]) / "train_report.json"),
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
        pairs=pairs,
        results=[],
        probes=[],
        gradient_probes=[],
        prompt=p["prompt"],
        seed_policy="group_id",
        test_status="sealed",
    )
    write_json(args.output / "report.json", report)
    try:
        backend = Flux2FeatureBackend(
            ProjectSettings.load().assets["models"]["base"]["path"],
            pairs[0]["lora"],
            pairs[0]["encoder"]["path"],
        )
        backend.base.pipe.transformer.requires_grad_(False)
        rows = evaluation_rows(plan, job)
        for index, pair in enumerate(pairs):
            if index:
                backend.base.pipe.unload_lora_weights()
                backend.base.pipe.load_lora_weights(pair["lora"])
            encoder, metadata = load_encoder(
                Path(pair["encoder"]["path"]), Path(pair["lora"]), device="cuda"
            )
            encoder.eval().requires_grad_(False)
            if metadata["mode"] != job["method"] or metadata["encoder"] != p["encoder"]:
                raise ValueError("Overfit mode/config differs from its actual encoder")
            chosen = (
                rows
                if pair["update"] in {0, p["steps"]}
                else [r for r in rows if r["split"] == "train"]
            )
            constants = {}
            for row in chosen:
                field = load_features(root, row, job["method"])
                edit = EditInput(load_edit_input(root, row).source, p["prompt"])
                tokens = torch.from_numpy(field.tokens())[None].to("cuda")
                grid = field.size[1] // 16, field.size[0] // 16
                opts = GenerationOptions(
                    *field.size,
                    p["inference"]["steps"],
                    p["inference"]["guidance"],
                    sample_seed(p["inference"]["seed"], row["group_id"]),
                )
                names, probe = case_policy(job["method"], pair["update"], row, rows, p)
                folder = args.output / f"step{pair['update']:03d}" / row["split"] / row["id"]
                predictions, captured, probes = generate_case(
                    backend, encoder, edit, tokens, grid, opts, names, folder, probe, p
                )
                # Explicit evaluation-only module: supervision is not passed to _generate.
                from src.evaluation.feature_teacher import score_teacher

                teacher = score_teacher(
                    backend.base.pipe,
                    encoder,
                    tokens,
                    grid,
                    captured,
                    root,
                    row,
                    p["teacher_times"],
                    opts.seed,
                )
                position_ids = captured["img_ids"]
                source_positions = (position_ids[0] if position_ids.ndim == 3 else position_ids)[
                    :, 0
                ] != 0
                source_tokens_sha = tensor_sha(captured["hidden_states"][:, source_positions])
                initial_noise_sha = tensor_sha(captured["hidden_states"][:, ~source_positions])
                text_sha = tensor_sha(captured["encoder_hidden_states"])
                for item in predictions:
                    report["results"].append(
                        dict(
                            item,
                            id=row["id"],
                            split=row["split"],
                            group_id=row["group_id"],
                            strength=float(row["strength"]),
                            update=pair["update"],
                            seed=opts.seed,
                            steps=opts.steps,
                            guidance=opts.guidance,
                            width=opts.width,
                            height=opts.height,
                            prompt=p["prompt"],
                            source_sha256=row["source_sha256"],
                            feature_sha256=field.digest,
                            checkpoint_sha256=pair["lora_sha256"],
                            encoder_sha256=pair["encoder"]["sha256"],
                            source_tokens_sha256=source_tokens_sha,
                            initial_noise_sha256=initial_noise_sha,
                            text_sha256=text_sha,
                            actual_feature_tokens_sha256=tensor_sha(
                                research_tokens(
                                    tokens, grid, item["variant"], p["inference"]["seed"]
                                )
                            ),
                            teacher=teacher if item["variant"] == "original" else None,
                        )
                    )
                if probes:
                    report["probes"].append(dict(id=row["id"], split=row["split"], probes=probes))
                    from src.evaluation.feature_gradient import gradient_probe

                    report["gradient_probes"].append(
                        dict(
                            id=row["id"],
                            split=row["split"],
                            gradient=gradient_probe(
                                backend.base.pipe,
                                encoder,
                                tokens,
                                grid,
                                captured,
                                root,
                                row,
                                p["teacher_times"],
                                opts.seed,
                            ),
                        )
                    )
                if job["method"] == "constant" or pair["update"] == 0:
                    digest = predictions[0]["prediction_sha256"]
                    key = row["split"], row["group_id"]
                    if key in constants and constants[key] != digest:
                        raise RuntimeError(
                            "Shared-source/prompt/noise Constant or zero-init PNG differs"
                        )
                    constants[key] = digest
                write_json(args.output / "report.json", report)
        completed(job)
        report.update(status="complete", constant_protocol_exact=True)
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
