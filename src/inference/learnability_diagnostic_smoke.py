"""One real checkpoint probe before the full controlled-generation matrix."""

import argparse
import os
import time
import traceback
from pathlib import Path

from src.backends.flux2_features import Flux2FeatureBackend
from src.conditioning.learnability import RESEARCH_VARIANTS
from src.core.config import ProjectSettings
from src.core.io import sha256, write_json
from src.core.metadata import now, versions
from src.core.random import sample_seed
from src.core.types import EditInput, GenerationOptions
from src.datasets.features import load_features
from src.datasets.manifest import load_edit_input
from src.evaluation.feature_gradient import gradient_probe
from src.evaluation.feature_teacher import score_teacher
from src.inference.learnability import evaluation_rows, generate_case
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.learnability_study import completed, load_plan
from src.training.lifecycle import attempt, gpu_statistics


def run(study: Path, output: Path):
    require_lease()
    import torch

    plan = load_plan(study)
    job = next(j for j in plan["jobs"] if j["phase"] == "capacity")
    trained = completed(job)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(
        status="running",
        created=now(),
        source=source_snapshot(),
        packages=versions(),
        plan_sha256=sha256(study / "plan.json"),
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
        training_report_sha256=sha256(Path(job["directory"]) / "train_report.json"),
        checkpoint_sha256=trained["checkpoint_sha256"],
        encoder_sha256=trained["feature_checkpoint"]["sha256"],
        test_status="sealed",
    )
    write_json(output / "report.json", report)
    try:
        p, root = plan["protocol"], Path(plan["data"])
        row = next(
            r
            for r in evaluation_rows(plan, job)
            if r["split"] == "train" and float(r["strength"]) > 0
        )
        field = load_features(root, row, "full_role")
        edit = EditInput(load_edit_input(root, row).source, p["prompt"])
        opts = GenerationOptions(
            *field.size,
            4,
            p["inference"]["guidance"],
            sample_seed(p["inference"]["seed"], row["group_id"]),
        )
        backend = Flux2FeatureBackend(
            ProjectSettings.load().assets["models"]["base"]["path"],
            trained["checkpoint"],
            trained["feature_checkpoint"]["path"],
        )
        encoder, model = backend.encoder, backend.base.pipe.transformer
        encoder.eval().requires_grad_(False)
        model.requires_grad_(False)
        tokens = torch.from_numpy(field.tokens())[None].to("cuda")
        grid = field.size[1] // 16, field.size[0] // 16
        predictions, captured, probes = generate_case(
            backend,
            encoder,
            edit,
            tokens,
            grid,
            opts,
            RESEARCH_VARIANTS,
            output / row["id"],
            True,
            p,
        )
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
        gradient = gradient_probe(
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
        if any(module._forward_pre_hooks for module in [model]) or any(
            m._forward_hooks for m in model.modules()
        ):
            raise RuntimeError("Diagnostic hooks were not removed")
        report.update(
            status="complete",
            id=row["id"],
            predictions=predictions,
            probes=probes,
            teacher=teacher,
            gradient=gradient,
            target_is_reference=False,
            training_updates=0,
        )
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
        write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)


if __name__ == "__main__":
    main()
