"""Real leased FLUX.2 feature probes. Diagnostics are not training-quality evidence."""

import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from src.backends.feature_checkpoint import load_encoder, save_encoder
from src.backends.feature_fusion import aligned_residual, feature_fusion
from src.backends.flux2 import Flux2Backend
from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.feature_encoder import build_encoder
from src.conditioning.features import EncoderConfig
from src.core.config import ProjectSettings
from src.core.io import read_rows, sha256, write_json
from src.core.metadata import now, versions
from src.core.types import GenerationOptions
from src.datasets.features import load_features
from src.datasets.holdout import require_dev
from src.datasets.manifest import load_edit_input
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.training.lifecycle import attempt, gpu_statistics


def transformer_probes(model, encoder, tokens, grid, captured):
    import torch

    # Cloning outside inference_mode makes saved pipeline tensors usable in backward.
    inputs = {
        k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in captured.items()
    }
    ids = inputs["img_ids"]
    if ids.ndim == 2:
        ids = ids[None]
    with torch.no_grad():
        residual = encoder(tokens, grid)
        selected, values = aligned_residual(residual, ids, grid)
        coordinates = ids[0, selected[0]]
        indices = coordinates[:, 1].long() * grid[1] + coordinates[:, 2].long()
        assert torch.equal(values[0], residual[0, indices])
        # Actual GPU IDs are validated even after a nontrivial token permutation.
        permuted = ids.flip(1)
        positions, reordered = aligned_residual(residual, permuted, grid)
        c = permuted[0, positions[0]]
        assert torch.equal(reordered[0], residual[0, c[:, 1].long() * grid[1] + c[:, 2].long()])
        reference = model(**inputs)[0].float()
        with feature_fusion(model, encoder, torch.zeros_like(tokens), grid):
            zero = model(**inputs)[0].float()
        if not torch.equal(reference, zero):
            raise RuntimeError("Trained all-zero features changed the reference transformer")
        variants = dict(
            original=tokens,
            old_new_swap=tokens[..., [1, 0, 2, 3, 4]],
            dependency_preserve_swap=tokens[..., [0, 1, 3, 2, 4]],
            spatial_shift=tokens.reshape(1, *grid, 5).roll(3, 2).reshape_as(tokens),
        )
        predictions = {}
        for name, feature in variants.items():
            with feature_fusion(model, encoder, feature, grid):
                predictions[name] = model(**inputs)[0].float()
        differences = {
            name: float((prediction - predictions["original"]).abs().max())
            for name, prediction in predictions.items()
            if name != "original"
        }
        if any(not np.isfinite(v) or v == 0 for v in differences.values()):
            raise RuntimeError(f"Condition permutation was silently ignored: {differences}")
    del predictions, reference, zero
    model.requires_grad_(False)
    model.train()
    model.enable_gradient_checkpointing()
    encoder.zero_grad(set_to_none=True)
    with feature_fusion(model, encoder, tokens, grid):
        output = model(**inputs)[0]
        # Diagnostic gradient only. Real study training uses the unmodified flow loss.
        output.float().square().mean().backward()
    gradients = {
        name: dict(
            finite=bool(p.grad is not None and p.grad.isfinite().all()),
            max_abs=float(p.grad.abs().max()) if p.grad is not None else None,
        )
        for name, p in encoder.named_parameters()
    }
    if not all(g["finite"] and g["max_abs"] > 0 for g in gradients.values()):
        raise RuntimeError(
            f"Real transformer did not propagate finite learning gradients: {gradients}"
        )
    model.eval()
    return dict(
        zero_condition_exact=True,
        spatial_ids_exact=True,
        permutation_linf=differences,
        gradients=gradients,
        feature_tokens=grid[0] * grid[1],
        source_tokens=int((ids[..., 0] == 10).sum()),
        noise_tokens=int((ids[..., 0] == 0).sum()),
    )


def run(args):
    require_lease()
    import torch

    require_dev(args.data_root, args.cohort, args.data_root / "dev/manifest.csv")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(
        status="running",
        created=now(),
        packages=versions(),
        source=source_snapshot(),
        gpu_uuid=os.environ["IMAGE_EDIT_GPU_UUID"],
        lora_sha256=sha256(args.lora),
        encoder_sha256=sha256(args.encoder),
        cohort_sha256=sha256(args.cohort),
    )
    write_json(args.output / "report.json", report)
    try:
        row = next(
            r for r in read_rows(args.data_root / "dev/manifest.csv") if float(r["strength"]) > 0
        )
        edit, features = (
            load_edit_input(args.data_root, row),
            load_features(args.data_root, row, "full_role"),
        )
        side = edit.source.width
        tokens, grid = (
            torch.from_numpy(features.tokens())[None].to("cuda"),
            (side // 16, side // 16),
        )
        options = GenerationOptions(side, side, 4, 4.0, 42)
        backend = Flux2Backend(args.model)
        model = backend.pipe.transformer
        fresh = build_encoder(EncoderConfig()).to("cuda")
        controls = ControlBundle(metadata={"reference_preprocessing": "training"})
        baseline = backend.edit(edit, controls, options).image
        with feature_fusion(model, fresh, tokens, grid):
            zero = backend.edit(edit, controls, options).image
        if not np.array_equal(baseline, zero):
            raise RuntimeError("Zero initialized features changed the pretrained Base PNG")
        baseline.save(args.output / "base.png")
        zero.save(args.output / "base_zero_initialized.png")
        report["pretrained_zero_init_exact"] = True
        legacy = backend.edit(edit, ControlBundle(), options).image
        with feature_fusion(model, None, None, None):
            legacy_disabled = backend.edit(edit, ControlBundle(), options).image
        if not np.array_equal(legacy, legacy_disabled):
            raise RuntimeError("Disabled features changed the legacy No-mask route")
        report["disabled_no_mask_exact"] = True
        # Explicitly exercise the unchanged legacy Mask route with the feature hook disabled.
        with Image.open(args.data_root / row["mask_path"]) as mask:
            mask_controls = ControlBundle(ControlKind.STATIC_MASK, {"mask": mask.copy()})
        mask_baseline = backend.edit(edit, mask_controls, options).image
        with feature_fusion(model, None, None, None):
            mask_disabled = backend.edit(edit, mask_controls, options).image
        if not np.array_equal(mask_baseline, mask_disabled):
            raise RuntimeError("Disabled features changed the legacy Mask route")
        mask_baseline.save(args.output / "base_static_mask.png")
        report["disabled_mask_exact"] = True
        backend.pipe.load_lora_weights(str(args.lora))
        encoder, metadata = load_encoder(args.encoder, args.lora, device="cuda")
        captured, vae_shapes = {}, []

        def capture(_model, _positional, kwargs):
            if not captured:
                captured.update(kwargs)

        handle = model.register_forward_pre_hook(capture, with_kwargs=True)
        # VAE.encode is a method, not Module.forward; instrument its actual call separately.
        encode = backend.pipe.vae.encode

        def encode_source(sample, *a, **kw):
            vae_shapes.append(list(sample.shape))
            return encode(sample, *a, **kw)

        backend.pipe.vae.encode = encode_source
        try:
            with feature_fusion(model, encoder, tokens, grid) as stats:
                prediction = backend.edit(edit, controls, options).image
        finally:
            handle.remove()
            backend.pipe.vae.encode = encode
        if vae_shapes != [[1, 3, side, side]]:
            raise RuntimeError(f"Expected source-only VAE encoding: {vae_shapes}")
        prediction.save(args.output / "trained.png")
        report["forward"] = dict(stats, vae_shapes=vae_shapes)
        report["probes"] = transformer_probes(model, encoder, tokens, grid, captured)
        paired = args.output / "reloaded.features.safetensors"
        save_encoder(paired, encoder, args.lora, "full_role")
        reloaded, reloaded_meta = load_encoder(paired, args.lora, device="cuda")
        if metadata != reloaded_meta or any(
            not torch.equal(v, reloaded.state_dict()[k]) for k, v in encoder.state_dict().items()
        ):
            raise RuntimeError("Feature checkpoint reload changed parameters")
        with feature_fusion(model, reloaded, tokens, grid):
            repeated = backend.edit(edit, controls, options).image
        if not np.array_equal(prediction, repeated):
            raise RuntimeError("Paired checkpoint reload changed the fixed-seed PNG")
        repeated.save(args.output / "trained_reloaded.png")
        report.update(
            status="complete",
            checkpoint_reload_png_exact=True,
            prediction_sha256=sha256(args.output / "trained.png"),
            sample_id=row["id"],
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        errors = {}
        stats = attempt("gpu_statistics", lambda: gpu_statistics(torch, errors), errors)
        report.update(stats or {})
        report["seconds"] = time.perf_counter() - started
        if errors:
            report["telemetry_errors"] = errors
        write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--model", default=ProjectSettings.load().assets["models"]["base"]["path"])
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
