"""Observe the existing fusion and replay identical positive/negative CFG states."""

import hashlib
from contextlib import nullcontext

from src.backends.feature_fusion import aligned_residual, feature_fusion
from src.conditioning.feature_interventions import intervention_encoder


def tensor_sha(tensor) -> str:
    data = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + data).hexdigest()


def rms(value) -> float:
    return float(value.detach().float().square().mean().sqrt())


def difference(left, right) -> dict:
    delta = left.float() - right.float()
    scale = rms(right)
    return dict(
        rms=rms(delta),
        linf=float(delta.abs().max()),
        relative_rms=rms(delta) / scale if scale else None,
    )


def fusion_measure(before, after, residual, ids, grid) -> dict:
    """Measure actual BF16 addition; support excludes exactly zero gated residuals."""
    import torch

    if ids.ndim == 2:
        ids = ids[None].expand(before.shape[0], -1, -1)
    positions, values = aligned_residual(residual, ids, grid)
    x = torch.stack([before[b, p] for b, p in enumerate(positions)])
    y = torch.stack([after[b, p] for b, p in enumerate(positions)])
    r = torch.stack(values).float()
    actual = y.float() - x.float()
    ideal_cast = (x.float() + r).to(x.dtype).float() - x.float()
    support = r.abs() > 1e-12
    rr, xx = rms(r), rms(x)
    return dict(
        input_dtype=str(x.dtype),
        input_rms=xx,
        residual_rms=rr,
        residual_linf=float(r.abs().max()),
        relative_rms=rr / xx if xx else None,
        actual_delta_rms=rms(actual),
        support_elements=int(support.sum()),
        lost_fraction=float((actual[support] == 0).float().mean()) if support.any() else None,
        quantization_relative_error=rms(actual - r) / rr if rr else None,
        fp32_sum_cast_error=rms(ideal_cast - r) / rr if rr else None,
        operand_rounding_error=rms(r.to(x.dtype).float() - r) / rr if rr else None,
    )


class StateCapture:
    """Cache selected pre-injection states on CPU; first call is positive CFG branch."""

    def __init__(self, selected_steps):
        self.selected = set(selected_steps)
        self.times, self.states = [], {}
        self.branches = {}
        self.outputs, self.pending = {}, None

    def __call__(self, _model, _args, kwargs):
        import torch

        timestep = float(kwargs["timestep"].flatten()[0])
        if not self.times or timestep != self.times[-1]:
            self.times.append(timestep)
        step = len(self.times) - 1
        branch = self.branches.get(step, 0)
        self.branches[step] = branch + 1
        self.pending = step, branch
        if step in self.selected:
            self.states.setdefault(step, {})[branch] = {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }

    def capture_output(self, _model, _args, output):
        step, branch = self.pending
        if step in self.selected:
            ids = self.states[step][branch]["img_ids"]
            noise = (ids[0] if ids.ndim == 3 else ids)[:, 0] == 0
            self.outputs.setdefault(step, {})[branch] = (
                output[0][:, noise.to(output[0].device)].detach().cpu().clone()
            )

    def validate(self, steps: int):
        if len(self.times) != steps or any(v != 2 for v in self.branches.values()):
            raise RuntimeError("Expected paired positive/negative calls at each real CFG step")
        if set(self.states) != self.selected:
            raise RuntimeError("Selected diffusion states were not captured")
        if set(self.outputs) != self.selected or any(
            set(v) != {0, 1} for v in self.outputs.values()
        ):
            raise RuntimeError("Actual CFG outputs were not captured with their input states")


def replay_branch(model, encoder, tokens, grid, inputs: dict, variants, protocol) -> dict:
    import torch

    inputs = {
        k: v.to(tokens.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    ids, x = inputs["img_ids"], inputs["hidden_states"]
    if x.shape[0] != 1:
        raise ValueError("Fixed-state diagnostics require one example per inference")
    noise = (ids[0] if ids.ndim == 3 else ids)[:, 0] == 0
    original_sha = tensor_sha(x)
    context_sha = tensor_sha(inputs["encoder_hidden_states"])
    guard_x, guard_context = x.clone(), inputs["encoder_hidden_states"].clone()
    outputs, embedded, measurements, effective, residuals = {}, {}, {}, {}, {}
    for name in ["disabled", *variants]:
        wrapped = (
            None
            if name == "disabled"
            else intervention_encoder(
                encoder, name, shift=protocol["shift_tokens"], seed=protocol["shuffle_seed"]
            )
        )
        observed = {}

        def encoded(_module, _args, output, sink=observed):
            sink["residual"] = output.detach()

        def fused(_module, _args, kwargs, sink=observed):
            sink["fused"] = kwargs["hidden_states"].detach()

        def embed(_module, _args, output, sink=observed):
            sink["embedded"] = output[:, noise].detach()

        eh = wrapped.register_forward_hook(encoded) if wrapped is not None else None
        handle = model.x_embedder.register_forward_hook(embed)
        try:
            with feature_fusion(model, wrapped, tokens, grid):
                fh = model.register_forward_pre_hook(fused, with_kwargs=True)
                try:
                    outputs[name] = model(**inputs)[0][:, noise].float()
                finally:
                    fh.remove()
        finally:
            handle.remove()
            if eh is not None:
                eh.remove()
        embedded[name] = observed["embedded"]
        effective[name] = observed["fused"][:, noise].float()
        if name != "disabled":
            measurements[name] = fusion_measure(
                x, observed["fused"], observed["residual"], ids, grid
            )
            positions = ids[None] if ids.ndim == 2 else ids
            _, values = aligned_residual(observed["residual"], positions, grid)
            residuals[name] = torch.stack(values).float()
        if not torch.equal(observed["fused"][:, ~noise], x[:, ~noise]):
            raise RuntimeError("Role intervention changed source reference tokens")
        if not torch.equal(x, guard_x) or not torch.equal(
            inputs["encoder_hidden_states"], guard_context
        ):
            raise RuntimeError("Fixed-state probe mutated noise/source or text inputs")
        if not outputs[name].isfinite().all():
            raise FloatingPointError("Non-finite fixed-state prediction")
    if not torch.equal(outputs["zero"], outputs["disabled"]):
        raise RuntimeError("Zero condition differs from disabled feature fusion")
    reference = outputs["original"]
    records = {}
    for name in variants:
        record = dict(
            fusion=measurements[name],
            output_vs_original=difference(outputs[name], reference),
            output_vs_disabled=difference(outputs[name], outputs["disabled"]),
            embedding_vs_disabled=difference(embedded[name], embedded["disabled"]),
            embedding_vs_original=difference(embedded[name], embedded["original"]),
        )
        different = effective[name] != effective["original"]
        record["fraction_input_elements_different_from_original"] = float(different.float().mean())
        support = (residuals[name] - residuals["original"]).abs() > 1e-12
        record["role_difference_lost_fraction"] = (
            float((~different[support]).float().mean()) if support.any() else None
        )
        record["residual_vs_original"] = difference(residuals[name], residuals["original"])
        records[name] = record
    return dict(
        input_sha256=original_sha, text_sha256=context_sha, records=records, outputs=outputs
    )


def replay_states(model, encoder, tokens, grid, capture: StateCapture, variants, protocol):
    import torch

    result = []
    with torch.inference_mode():
        for step, branches in sorted(capture.states.items()):
            if set(branches) != {0, 1}:
                raise RuntimeError("Missing CFG branch in fixed-state replay")
            measured = []
            for branch in (0, 1):
                context = (
                    model.cache_context("cond" if branch == 0 else "uncond")
                    if hasattr(model, "cache_context")
                    else nullcontext()
                )
                with context:
                    measured.append(
                        replay_branch(
                            model, encoder, tokens, grid, branches[branch], variants, protocol
                        )
                    )
            positive, negative = measured
            if positive["input_sha256"] != negative["input_sha256"]:
                raise RuntimeError("CFG branches have different noise/source states")
            p, n = positive.pop("outputs"), negative.pop("outputs")
            for branch, predictions in enumerate([p, n]):
                expected = capture.outputs[step][branch].to(tokens.device).float()
                if not torch.equal(predictions["original"], expected):
                    raise RuntimeError(
                        "Replayed original differs from its actual trajectory prediction"
                    )
            guidance = protocol["inference"]["guidance"]
            reference = n["original"] + guidance * (p["original"] - n["original"])
            cfg = {}
            for name in variants:
                pred = n[name] + guidance * (p[name] - n[name])
                dp, dn = p[name] - p["original"], n[name] - n["original"]
                norm = rms(dp) * rms(dn)
                cfg[name] = dict(
                    **difference(pred, reference),
                    positive_negative_delta_cosine=float((dp * dn).mean()) / norm if norm else None,
                )
            result.append(
                dict(
                    step=step,
                    timestep=capture.times[step],
                    positive=positive,
                    negative=negative,
                    cfg=cfg,
                    zero_disabled_exact=True,
                    replay_original_exact=True,
                )
            )
    return result
