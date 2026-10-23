"""Explicit research interventions with unchanged valid geometry and frozen capacity."""

from src.conditioning.feature_interventions import intervene
from src.conditioning.feature_residuals import validity_permutation

RESEARCH_VARIANTS = ("original", "zero", "swap_old_new", "swap_dp", "shuffle")


def research_tokens(tokens, grid, name: str, seed: int):
    import torch

    if name not in RESEARCH_VARIANTS:
        raise ValueError("Unknown controlled role intervention")
    if name == "shuffle":
        order = validity_permutation(tokens, seed)
        return torch.stack([tokens[b, p] for b, p in enumerate(order)])
    return intervene(tokens, grid, name)


def research_encoder(encoder, name: str, seed: int):
    from torch import nn

    if name not in RESEARCH_VARIANTS:
        raise ValueError("Unknown controlled role intervention")

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base, self.config = encoder, encoder.config

        def forward(self, tokens, grid):
            return self.base(research_tokens(tokens, grid, name, seed), grid)

    return Wrapper().eval()
