"""Scoped noise-token residuals for pinned BFL training and Diffusers inference."""

from contextlib import contextmanager


def aligned_residual(residual, ids, grid):
    import torch

    height, width = grid
    if ids.ndim != 3 or ids.shape[-1] != 4 or not ids.isfinite().all():
        raise ValueError("FLUX feature fusion needs finite BxNx4 position IDs")
    positions, aligned = [], []
    for b in range(ids.shape[0]):
        selected = (ids[b, :, 0] == 0).nonzero().flatten()
        coordinates = ids[b, selected]
        if (
            len(selected) != height * width
            or not (coordinates == coordinates.long()).all()
            or not (coordinates[:, 3] == 0).all()
            or (coordinates[:, 1:3] < 0).any()
            or (coordinates[:, 1] >= height).any()
            or (coordinates[:, 2] >= width).any()
        ):
            raise ValueError("Noise position grid differs from feature pixels/stride")
        indices = coordinates[:, 1].long() * width + coordinates[:, 2].long()
        if not torch.equal(indices.sort().values, torch.arange(height * width, device=ids.device)):
            raise ValueError("Noise positions must form a complete unique feature grid")
        positions.append(selected)
        aligned.append(residual[b, indices])
    return positions, aligned


@contextmanager
def feature_fusion(transformer, encoder, tokens, grid, *, flavor="diffusers"):
    """No hook when disabled. A zero residual still keeps its learning gradient."""
    stats = dict(calls=0)
    if encoder is None:
        yield stats
        return
    if flavor not in {"bfl", "diffusers"}:
        raise ValueError("Unsupported feature backend")
    x_key, ids_key = ("x", "x_ids") if flavor == "bfl" else ("hidden_states", "img_ids")

    def inject(_model, args, kwargs):
        import torch

        x, ids = kwargs[x_key], kwargs[ids_key]
        if ids.ndim == 2:
            ids = ids[None].expand(x.shape[0], -1, -1)
        if x.shape[:2] != ids.shape[:2] or x.shape[-1] != encoder.config.output:
            raise ValueError("Transformer feature shape/position mismatch")
        f = tokens.to(device=x.device, dtype=torch.float32)
        if f.shape[0] != x.shape[0]:
            raise ValueError("Feature and image batch sizes differ")
        residual = encoder(f, grid)
        if not residual.isfinite().all():
            raise ValueError("Non-finite role residual")
        positions, values = aligned_residual(residual, ids, grid)
        fused = x.clone()
        for b, (indices, value) in enumerate(zip(positions, values, strict=True)):
            fused[b, indices] = x[b, indices] + value.to(x.dtype)
        stats.update(
            calls=stats["calls"] + 1,
            tokens=residual.shape[1],
            grid=list(grid),
            residual_linf=float(residual.detach().abs().max()),
            source_tokens=[int((position[:, 0] != 0).sum()) for position in ids],
            reference_times=ids[..., 0][ids[..., 0] != 0].unique().tolist(),
        )
        return args, dict(kwargs, **{x_key: fused})

    handle = transformer.register_forward_pre_hook(inject, with_kwargs=True)
    try:
        yield stats
    finally:
        handle.remove()
