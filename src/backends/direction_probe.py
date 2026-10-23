"""Frozen, target-free states and native BF16 CFG replay on pinned FLUX.2."""

from contextlib import nullcontext

from src.backends.feature_fusion import aligned_residual, feature_fusion
from src.backends.feature_probe import StateCapture, fusion_measure
from src.core.metadata import tensor_sha


def native_cfg(positive, negative, guidance):
    return negative + guidance * (positive - negative)


def geometry(ids, grid):
    import torch

    if ids.ndim == 2:
        ids = ids[None]
    # Reuse the actual fusion geometry checks, including uniqueness and all four IDs.
    dummy = torch.zeros((1, grid[0] * grid[1], 128), device=ids.device)
    positions, _ = aligned_residual(dummy, ids, grid)
    selected = positions[0]
    indices = ids[0, selected, 1].long() * grid[1] + ids[0, selected, 2].long()
    return selected[indices.argsort()]


def capture_source(backend, edit, opts, grid):
    """Capture both real pipeline branches and the scheduler's native CFG input."""
    import torch

    model, scheduler = backend.base.pipe.transformer, backend.base.pipe.scheduler
    device = next(model.parameters()).device
    capture, received = StateCapture([0]), []
    old_step, had_step = scheduler.step, "step" in scheduler.__dict__

    def observe_step(prediction, *args, **kwargs):
        received.append(prediction.detach().clone())
        return old_step(prediction, *args, **kwargs)

    handles = [
        model.register_forward_pre_hook(capture, with_kwargs=True),
        model.register_forward_hook(capture.capture_output),
    ]
    scheduler.step = observe_step
    try:
        backend.base._generate(edit, [edit.source], opts, matched=True)
    finally:
        for handle in handles:
            handle.remove()
        if had_step:
            scheduler.step = old_step
        else:
            del scheduler.step
    capture.validate(1)
    states = {
        branch: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        for branch, state in capture.states[0].items()
    }
    p, n = [capture.outputs[0][i].to(device) for i in (0, 1)]
    if len(received) != 1 or not torch.equal(native_cfg(p, n, opts.guidance), received[0]):
        raise RuntimeError("Native CFG does not match the actual pinned scheduler input")
    if not torch.equal(states[0]["hidden_states"], states[1]["hidden_states"]):
        raise RuntimeError("CFG source/noise states differ")
    if float(states[0]["timestep"][0]) != 1:
        raise RuntimeError("Pure-noise capture must start at actual model time 1")
    positions = geometry(states[0]["img_ids"], grid)
    return (
        states,
        positions,
        dict(
            native_cfg_pipeline_exact=True,
            zero_reference={
                "positive": tensor_sha(p),
                "negative": tensor_sha(n),
                "cfg_native": tensor_sha(received[0]),
            },
        ),
    )


def build_state(captured, positions, anchor, time, family):
    import torch

    if family not in {"source_bridge", "noise_only"} or not 0 < time <= 1:
        raise ValueError("Unsupported target-free state")
    stamp = torch.full_like(captured[0]["timestep"], time)
    actual = float(stamp[0])
    initial = captured[0]["hidden_states"]
    epsilon = initial[:, positions]
    x = (
        (1 - actual) * anchor.float() + actual * epsilon.float()
        if family == "source_bridge" and actual != 1
        else epsilon
    ).to(initial.dtype)
    hidden = initial.clone()
    hidden[:, positions] = x
    states = {b: dict(v, hidden_states=hidden, timestep=stamp) for b, v in captured.items()}
    return states, x, actual


def predict(model, encoder, tokens, grid, states, positions):
    """Only source-derived states and role fields enter this function; no target argument."""
    import torch

    outputs, measurements, fusion_arrays = {}, {}, {}
    guard = states[0]["hidden_states"].clone()
    source = torch.ones(guard.shape[1], dtype=torch.bool, device=guard.device)
    source[positions] = False
    for branch in (0, 1):
        kwargs = states[branch]
        text = kwargs["encoder_hidden_states"].clone()
        observed = {}

        def fused(_model, _args, inputs, sink=observed):
            sink["fused"] = inputs["hidden_states"].detach()

        def encoded(_encoder, _args, value, sink=observed):
            sink["residual"] = value.detach()

        hook = encoder.register_forward_hook(encoded) if encoder is not None else None
        try:
            with (
                feature_fusion(model, encoder, tokens, grid),
                model.cache_context("cond" if branch == 0 else "uncond")
                if hasattr(model, "cache_context")
                else nullcontext(),
            ):
                handle = model.register_forward_pre_hook(fused, with_kwargs=True)
                try:
                    result = model(**kwargs)[0][:, positions]
                finally:
                    handle.remove()
        finally:
            if hook:
                hook.remove()
        if (
            not torch.equal(observed["fused"][:, source], guard[:, source])
            or not torch.equal(kwargs["hidden_states"], guard)
            or not torch.equal(text, kwargs["encoder_hidden_states"])
        ):
            raise RuntimeError("Direction intervention changed source, text or common state")
        if not result.isfinite().all():
            raise FloatingPointError("Non-finite direction velocity")
        outputs["positive" if branch == 0 else "negative"] = result.detach()
        if encoder is not None:
            measurements[str(branch)] = fusion_measure(
                guard, observed["fused"], observed["residual"], kwargs["img_ids"], grid
            )
        arrays = dict(
            residual=observed.get("residual", torch.zeros_like(guard[:, positions]).float()),
            fused=observed["fused"][:, positions],
        )
        if branch == 0:
            fusion_arrays = arrays
        elif any(not torch.equal(fusion_arrays[k], arrays[k]) for k in arrays):
            raise RuntimeError("CFG branches received different role residuals")
    return outputs, measurements, fusion_arrays


def state_identity(states, positions):
    mask = states[0]["img_ids"]
    source = (mask[0] if mask.ndim == 3 else mask)[:, 0] != 0
    return dict(
        source_tokens_sha256=tensor_sha(states[0]["hidden_states"][:, source]),
        initial_noise_sha256=tensor_sha(states[0]["hidden_states"][:, positions]),
        text_sha256=tensor_sha(states[0]["encoder_hidden_states"]),
        negative_text_sha256=tensor_sha(states[1]["encoder_hidden_states"]),
    )
