"""Real source-only overfit checkpoint smoke, including backward and exact reload."""

import argparse
import time
import traceback
from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.backends.feature_fusion import feature_fusion
from src.backends.flux2_features import Flux2FeatureBackend
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import now, versions
from src.core.random import sample_seed
from src.core.types import EditInput, GenerationOptions
from src.datasets.features import load_features
from src.datasets.manifest import load_edit_input
from src.datasets.research_view import verify_view
from src.inference.feature_smoke import transformer_probes
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.learnability_study import completed, load_plan
from src.training.lifecycle import attempt, gpu_statistics


def run(study: Path, output: Path):
    require_lease()
    import torch

    plan = load_plan(study)
    job = next(j for j in plan["jobs"] if j["phase"] == "smoke")
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
        checkpoint_sha256=trained["checkpoint_sha256"],
        encoder_sha256=trained["feature_checkpoint"]["sha256"],
    )
    write_json(output / "report.json", report)
    try:
        root = Path(plan["data"])
        view = verify_view(Path(job["directory"]) / "view/view.json", root=root)
        row = next(
            r
            for r in read_rows(root / "train/manifest.csv")
            if r["id"] in view["ids"] and float(r["strength"]) > 0
        )
        field = load_features(root, row, "full_role")
        edit = EditInput(load_edit_input(root, row).source, plan["protocol"]["prompt"])
        opts = GenerationOptions(
            *field.size,
            4,
            plan["protocol"]["inference"]["guidance"],
            sample_seed(plan["protocol"]["inference"]["seed"], row["group_id"]),
        )
        backend = Flux2FeatureBackend(
            ProjectSettings.load().assets["models"]["base"]["path"],
            trained["checkpoint"],
            trained["feature_checkpoint"]["path"],
        )
        model, encoder = backend.base.pipe.transformer, backend.encoder
        if encoder.config.as_dict() != plan["protocol"]["encoder"]:
            raise ValueError("Smoke checkpoint has a different feature normalization/config")
        report["encoder_config"] = encoder.config.as_dict()
        tokens = torch.from_numpy(field.tokens())[None].to("cuda")
        grid = field.size[1] // 16, field.size[0] // 16
        captured = {}

        def capture(_m, _args, kwargs):
            if not captured:
                captured.update(kwargs)

        handle = model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            first = backend.edit_features(edit, field, opts, allow_oracle=True)
        finally:
            handle.remove()
        first.image.save(output / "first.png")
        report["probes"] = transformer_probes(model, encoder, tokens, grid, captured)
        reloaded, _ = load_encoder(
            Path(trained["feature_checkpoint"]["path"]), Path(trained["checkpoint"]), device="cuda"
        )
        with feature_fusion(model, reloaded, tokens, grid):
            repeat = backend.base._generate(edit, [edit.source], opts, matched=True)
        repeat.image.save(output / "reloaded.png")
        if sha256(output / "first.png") != sha256(output / "reloaded.png"):
            raise RuntimeError("Overfit smoke checkpoint reload changed PNG")
        report.update(
            status="complete",
            reload_exact=True,
            sample_id=row["id"],
            prediction_sha256=sha256(output / "first.png"),
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        errors = {}
        report.update(
            attempt("gpu_statistics", lambda: gpu_statistics(torch, errors), errors) or {}
        )
        report.update(seconds=time.perf_counter() - started)
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
