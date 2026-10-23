"""CPU measurements of existing encoders; no training, targets or model selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.backends.feature_checkpoint import load_encoder
from src.backends.feature_probe import difference, rms
from src.conditioning.feature_encoder import build_encoder, category_inputs
from src.conditioning.feature_interventions import intervene
from src.conditioning.features import EncoderConfig
from src.core.io import read_rows, write_json
from src.core.random import sample_seed
from src.datasets.features import load_features
from src.inference.feature_diagnostic_study import load_plan
from src.infra.experiment import source_snapshot


def input_components(encoder, tokens, grid):
    """Observe the actual category/position sum without duplicating encoder math."""
    captured = {}

    def hidden(_module, args):
        captured["hidden"] = args[0].detach()

    handle = encoder.hidden.register_forward_pre_hook(hidden)
    try:
        residual = encoder(tokens, grid)
    finally:
        handle.remove()
    category = category_inputs(tokens, encoder.config.normalization) @ encoder.categories
    return residual, category, captured["hidden"] - category


def analyze(study: Path, output: Path):
    import torch

    if output.exists():
        raise FileExistsError(output)
    plan = load_plan(study)
    started = time.perf_counter()
    torch.set_num_threads(2)
    root, p = Path(plan["data"]), plan["protocol"]
    result = dict(
        status="running",
        source=source_snapshot(),
        training_updates=0,
        encoders=[],
        precision_scope="Controlled unit Gaussian CPU noise; not real diffusion/GPU states",
    )
    for ckpt in plan["checkpoints"]:
        encoder, metadata = load_encoder(Path(ckpt["encoder"]), Path(ckpt["lora"]))
        encoder.eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(ckpt["seed"])
            initial = build_encoder(EncoderConfig(**metadata["encoder"]))
        initial_sha = hashlib.sha256(
            b"".join(v.detach().numpy().tobytes() for v in initial.state_dict().values())
        ).hexdigest()
        training = json.loads(Path(ckpt["training_report"]).read_text())
        if initial_sha != training["feature_statistics"]["initial_sha256"]:
            raise ValueError("Initial encoder reconstruction differs from the real training seed")
        parameters = {
            name: dict(
                initial_rms=rms(initial.state_dict()[name]),
                trained_rms=rms(v),
                delta_rms=rms(v - initial.state_dict()[name]),
            )
            for name, v in encoder.state_dict().items()
        }
        samples = []
        for row in read_rows(root / "dev/manifest.csv"):
            field = load_features(root, row, ckpt["method"])
            tokens = torch.from_numpy(field.tokens())[None]
            grid = field.size[1] // 16, field.size[0] // 16
            with torch.inference_mode():
                original, category, position = input_components(encoder, tokens, grid)
                generator = torch.Generator(device="cpu").manual_seed(
                    sample_seed(p["inference"]["seed"], row["id"])
                )
                controlled_noise = torch.randn(original.shape, generator=generator).bfloat16()
                original_rounded = (controlled_noise + original.bfloat16()).float()
                variants = {}
                for name in p["variants"]:
                    values = intervene(
                        tokens, grid, name, shift=p["shift_tokens"], seed=p["shuffle_seed"]
                    )
                    residual = encoder(values, grid)
                    record = dict(output_rms=rms(residual), **difference(residual, original))
                    selected = (values - tokens).abs().amax(-1) > 1e-8
                    local = rms((residual - original)[selected]) if selected.any() else None
                    reference = rms(original[selected]) if selected.any() else None
                    record.update(
                        active_token_fraction=float(selected.float().mean()),
                        local_delta_rms=local,
                        local_relative_rms=local / reference if reference else None,
                    )
                    support = (residual - original).abs() > 1e-12
                    rounded = (controlled_noise + residual.bfloat16()).float()
                    record["synthetic_bf16_role_difference_lost_fraction"] = (
                        float((rounded[support] == original_rounded[support]).float().mean())
                        if support.any()
                        else None
                    )
                    variants[name] = record
                samples.append(
                    dict(
                        id=row["id"],
                        strength=float(row["strength"]),
                        feature_sha256=field.digest,
                        category_input_rms=rms(category),
                        position_input_rms=rms(position),
                        residual_rms=rms(original),
                        token_means=tokens.mean((0, 1)).tolist(),
                        variants=variants,
                    )
                )
        result["encoders"].append(
            dict(
                checkpoint=ckpt, initial_sha256=initial_sha, parameters=parameters, samples=samples
            )
        )
    result.update(status="complete", seconds=time.perf_counter() - started)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.study, args.output)


if __name__ == "__main__":
    main()
