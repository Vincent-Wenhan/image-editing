"""Persist actual GPU residual arrays and verify their formula and checkpoint binding."""

from pathlib import Path

import numpy as np

from src.conditioning.feature_interventions import intervene
from src.conditioning.feature_residuals import RESIDUAL_VARIANTS, residual_values
from src.core.io import sha256
from src.core.metadata import tensor_sha


def save_conditions(folder: Path, wrappers: dict, arrays: dict, metadata: dict) -> dict:
    folder.mkdir(parents=True)
    base = folder / "base.npz"
    np.savez(base, **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
    records = {}
    for name, wrapper in wrappers.items():
        path = folder / f"{name}.npy"
        np.save(path, wrapper.residual.cpu().numpy(), allow_pickle=False)
        records[name] = dict(
            schema_version=1,
            variant=name,
            base_path=str(base.resolve()),
            base_sha256=sha256(base),
            path=str(path.resolve()),
            sha256=sha256(path),
            tensor_sha256=tensor_sha(wrapper.residual),
            statistics=metadata,
        )
    return records


def verify_condition(record: dict, encoder, field, variant: str):
    import torch

    if (
        record["schema_version"] != 1
        or variant not in RESIDUAL_VARIANTS
        or record["variant"] != variant
        or sha256(record["base_path"]) != record["base_sha256"]
        or sha256(record["path"]) != record["sha256"]
    ):
        raise ValueError("Residual intervention cache provenance changed")
    with np.load(record["base_path"], allow_pickle=False) as data:
        if set(data.files) != {"tokens", "original", "identity", "permutation"}:
            raise ValueError("Unexpected residual cache arrays")
        arrays = {key: torch.from_numpy(data[key].copy()) for key in data.files}
    tokens = torch.from_numpy(field.tokens())[None]
    stats = record["statistics"]
    grid = field.size[1] // 16, field.size[0] // 16
    if (
        not torch.equal(tokens, arrays["tokens"])
        or stats["grid"] != list(grid)
        or stats["cap"] != encoder.config.max_residual
        or stats["parameters"] != sum(p.numel() for p in encoder.parameters())
        or stats["tokens"] != tokens.shape[1]
        or stats["encoder_evaluations"] != 2
    ):
        raise ValueError("Residual cache input/geometry/budget changed")
    with torch.inference_mode():
        # Cross-device FP32 recomputation: explicitly tolerant, not a bit-exact claim.
        for name, features in [
            ("original", tokens),
            ("identity", intervene(tokens, grid, "identity")),
        ]:
            actual = encoder(features, grid)
            if arrays[name].dtype != torch.float32 or not torch.allclose(
                actual, arrays[name], rtol=1e-5, atol=1e-7
            ):
                raise ValueError("Cached residual differs from the actual paired encoder")
        values, order, _metadata = residual_values(
            arrays["original"], arrays["identity"], tokens, stats["cap"], stats["shuffle_seed"]
        )
    selected = torch.from_numpy(np.load(record["path"], allow_pickle=False))
    if (
        not torch.equal(order, arrays["permutation"])
        or selected.dtype != torch.float32
        or not selected.isfinite().all()
        or tensor_sha(selected) != record["tensor_sha256"]
        or stats["residual_sha256"][variant] != record["tensor_sha256"]
        or not torch.allclose(selected, values[variant], rtol=1e-5, atol=1e-7)
    ):
        raise ValueError("Actual residual intervention formula/permutation changed")
