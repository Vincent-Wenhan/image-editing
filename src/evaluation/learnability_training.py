"""Summarize real optimizer observations and small encoder input scales on CPU."""

from pathlib import Path

import numpy as np

from src.backends.feature_checkpoint import load_encoder
from src.backends.feature_probe import rms
from src.core.io import read_rows
from src.datasets.features import load_features
from src.evaluation.feature_encoder_diagnostics import input_components


def verify_full_batch_trace(trained, ids):
    trace = trained["feature_statistics"]["input_trace"]
    if len(trace) != trained["steps"] or any(
        len(t["ids"]) != len(ids) or set(t["ids"]) != ids for t in trace
    ):
        raise ValueError(
            "Every overfit update must contain each selected Train example exactly once"
        )


def training_diagnostics(trained, root, ids):
    import torch

    verify_full_batch_trace(trained, ids)
    observation = trained["feature_statistics"]["observation"]
    encoder, _ = load_encoder(
        Path(trained["feature_checkpoint"]["path"]), Path(trained["checkpoint"])
    )
    encoder.eval().requires_grad_(False)
    scales = []
    with torch.inference_mode():
        for row in read_rows(root / "train/manifest.csv"):
            if row["id"] not in ids:
                continue
            field = load_features(root, row, trained["role_features"]["mode"])
            tokens = torch.from_numpy(field.tokens())[None]
            residual, category, position = input_components(
                encoder, tokens, (field.size[1] // 16, field.size[0] // 16)
            )
            scales.append(
                dict(
                    id=row["id"],
                    category_rms=rms(category),
                    position_rms=rms(position),
                    content_to_position=rms(category) / rms(position) if rms(position) else None,
                    residual_rms=rms(residual),
                    token_means=tokens.mean((0, 1)).tolist(),
                )
            )
    losses = [r["loss"] for r in observation["loss"]]
    return dict(
        loss_scope=observation["scope"],
        loss_first20=float(np.mean(losses[:20])),
        loss_last20=float(np.mean(losses[-20:])),
        updates=observation["updates"],
        input_scales=scales,
        interpretation=(
            "Finite gradients and time/layer responses are diagnostics; "
            "they do not identify a unique bottleneck causally"
        ),
    )
