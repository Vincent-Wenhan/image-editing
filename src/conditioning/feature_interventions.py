"""Explicit frozen-checkpoint interventions; never load supervision or change RNG state."""

VARIANTS = (
    "original",
    "zero",
    "identity",
    "swap_old_new",
    "swap_dp",
    "drop_old",
    "drop_new",
    "drop_d",
    "drop_p",
    "shift",
    "shuffle",
)


def intervene(tokens, grid, name: str, *, shift: int = 4, seed: int = 20261023):
    import torch

    h, w = grid
    if name not in VARIANTS or tokens.ndim != 3 or tokens.shape[1:] != (h * w, 5):
        raise ValueError("Unknown intervention or mismatched feature grid")
    if not tokens.isfinite().all() or tokens.min() < 0 or tokens.max() > 1:
        raise ValueError("Interventions require finite unit features")
    result = tokens.clone()
    if name == "zero":
        result.zero_()
    elif name == "identity":
        result[..., 1] = tokens[..., 0]
        result[..., 2] = 0
        result[..., 3] = tokens[..., 4]
    elif name in {"swap_old_new", "swap_dp"}:
        order = [1, 0, 2, 3, 4] if name == "swap_old_new" else [0, 1, 3, 2, 4]
        result = tokens[..., order]
    elif name.startswith("drop_"):
        channel = {"drop_old": 0, "drop_new": 1, "drop_d": 2, "drop_p": 3}[name]
        result[..., channel] = 0
    elif name == "shift":
        if type(shift) is not int or not 0 < shift < w:
            raise ValueError("Shift must leave some feature columns in bounds")
        view = tokens.reshape(-1, h, w, 5)
        result = torch.zeros_like(view)
        result[:, :, shift:] = view[:, :, :-shift]
        result = result.reshape_as(tokens)
    elif name == "shuffle":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(h * w, generator=generator).to(tokens.device)
        result = tokens[:, order]
    return result


def intervention_encoder(encoder, name: str, *, shift: int, seed: int):
    """Same parameters and token count; wrapper is inference-only, never checkpointed."""
    from torch import nn

    if name not in VARIANTS:
        raise ValueError("Unsupported frozen feature intervention")

    class Intervention(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = encoder
            self.config = encoder.config

        def forward(self, tokens, grid):
            return self.base(intervene(tokens, grid, name, shift=shift, seed=seed), grid)

    return Intervention().eval()
