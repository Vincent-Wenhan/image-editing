"""One frozen inference intervention, with validity-preserving spatial sham controls."""

from src.conditioning.feature_interventions import intervene
from src.core.metadata import tensor_sha

RESIDUAL_VARIANTS = ("original", "zero", "centered", "centered_shuffled")


def centered_residual(original, identity, cap: float):
    """Per-example energy matching, using only the two existing encoder outputs."""
    import torch

    if (
        original.shape != identity.shape
        or original.ndim != 3
        or cap <= 0
        or not original.isfinite().all()
        or not identity.isfinite().all()
    ):
        raise ValueError("Expected matching finite residuals and a positive cap")
    r, delta = original.float(), original.float() - identity.float()
    scale = r.square().mean((1, 2), keepdim=True).sqrt()
    norm = delta.square().mean((1, 2), keepdim=True).sqrt()
    epsilon = torch.finfo(torch.float32).eps
    gain = torch.where(norm > epsilon, scale / norm.clamp_min(epsilon), 0)
    before = delta * gain
    return before.clamp(-cap, cap), dict(
        gain=gain.flatten().tolist(),
        clipped_fraction=float((before.abs() > cap).float().mean()),
    )


def validity_permutation(tokens, seed: int):
    """Permute within exact pooled-validity strata, preserving gating and energy."""
    import torch

    valid = tokens.detach().cpu()[..., 4]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutations = []
    for row in valid:
        order = torch.arange(len(row))
        for value in row.unique(sorted=True):
            members = (row == value).nonzero().flatten()
            order[members] = members[torch.randperm(len(members), generator=generator)]
        permutations.append(order)
    return torch.stack(permutations).to(tokens.device)


def residual_values(original, identity, tokens, cap: float, seed: int):
    import torch

    candidate, metadata = centered_residual(original, identity, cap)
    order = validity_permutation(tokens, seed)
    shuffled = torch.stack([candidate[b, p] for b, p in enumerate(order)])
    return (
        dict(
            original=original,
            zero=torch.zeros_like(original),
            centered=candidate,
            centered_shuffled=shuffled,
        ),
        order,
        metadata,
    )


def cached_encoders(encoder, tokens, grid, seed: int):
    """Two encoder evaluations for every arm; reject a mismatched cache input."""
    import torch
    from torch import nn

    with torch.inference_mode():
        original = encoder(tokens, grid).float()
        identity = encoder(intervene(tokens, grid, "identity"), grid).float()
        values, order, metadata = residual_values(
            original, identity, tokens, encoder.config.max_residual, seed
        )

    class Cached(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.base, self.config = encoder, encoder.config
            self.register_buffer("condition", tokens.detach().clone(), persistent=False)
            self.register_buffer("residual", value.detach().clone(), persistent=False)
            self.calls = 0

        def forward(self, features, actual_grid):
            if tuple(actual_grid) != tuple(grid) or not torch.equal(features, self.condition):
                raise ValueError("Cached residual input/grid changed")
            self.calls += 1
            return self.residual

    wrappers = {name: Cached(value).eval() for name, value in values.items()}
    metadata.update(
        encoder_evaluations=2,
        parameters=sum(p.numel() for p in encoder.parameters()),
        tokens=tokens.shape[1],
        grid=list(grid),
        shuffle_seed=seed,
        cap=encoder.config.max_residual,
        residual_sha256={name: tensor_sha(value) for name, value in values.items()},
        original_rms=float(original.square().mean().sqrt()),
        centered_rms=float(values["centered"].square().mean().sqrt()),
    )
    return (
        wrappers,
        dict(tokens=tokens, original=original, identity=identity, permutation=order),
        metadata,
    )
