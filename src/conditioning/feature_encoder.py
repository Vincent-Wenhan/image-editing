"""Lazy torch construction preserves CPU-only CLI/module discovery."""

from src.conditioning.features import EncoderConfig


def category_inputs(features, normalization: str):
    """Optional parameter-free preconditioning; statistics never depend on targets."""
    if normalization == "none":
        return features
    if normalization != "channel_rms":
        raise ValueError("Unknown role feature normalization")
    import torch

    scale = features.detach().float().square().mean(1, keepdim=True).sqrt()
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return features / scale


def build_encoder(config: EncoderConfig):
    import torch
    from torch import nn

    class RoleFeatureEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.categories = nn.Parameter(torch.empty(5, config.hidden))
            nn.init.normal_(self.categories, std=0.02)
            self.position = nn.Linear(8, config.hidden, bias=False)
            self.hidden = nn.Linear(config.hidden, config.hidden)
            self.output = nn.Linear(config.hidden, config.output)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, features, grid):
            height, width = grid
            if (
                features.ndim != 3
                or features.shape[1:] != (height * width, 5)
                or not features.isfinite().all()
                or features.min() < 0
                or features.max() > 1
            ):
                raise ValueError("Encoder needs Bx(grid H*W)x5 unit features")
            with torch.autocast(device_type=features.device.type, enabled=False):
                y, x = torch.meshgrid(
                    torch.arange(height, device=features.device),
                    torch.arange(width, device=features.device),
                    indexing="ij",
                )
                x, y = (x.flatten().float() + 0.5) / width, (y.flatten().float() + 0.5) / height
                position = torch.stack(
                    [
                        fn(v * frequency * 2 * torch.pi)
                        for frequency in (1, 2)
                        for v in (x, y)
                        for fn in (torch.sin, torch.cos)
                    ],
                    -1,
                )
                f = features.float()
                content = category_inputs(f, config.normalization) @ self.categories
                hidden = content + self.position(position)[None]
                residual = torch.tanh(self.output(torch.nn.functional.silu(self.hidden(hidden))))
                return residual * config.max_residual * f[..., 4:5]

    return RoleFeatureEncoder()
