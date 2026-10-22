"""Scoped real VAE/attention diagnostics; never change the model or its inputs."""

from contextlib import contextmanager

from src.backends.flux2_diagnostics import tensor_digest


@contextmanager
def trace_references(pipe):
    trace = dict(vae=[])
    handles = []

    def encoder(_module, inputs):
        value = inputs[0]
        trace["vae"].append(
            dict(
                sha256=tensor_digest(value),
                shape=list(value.shape),
                min=value.min().item(),
                max=value.max().item(),
                binary=bool(((value == -1) | (value == 1)).all().item()),
                positive_channels_max=int((value > 0).sum(dim=1).max().item()),
            )
        )

    def attention(_module, _inputs, kwargs):
        if "attention" in trace:
            return
        import torch

        ids, hidden = kwargs["img_ids"][0], kwargs["hidden_states"]
        offsets, counts = torch.unique(ids[:, 0], return_counts=True)
        trace["attention"] = dict(
            offsets=offsets.tolist(),
            counts=counts.tolist(),
            source_sha256=tensor_digest(hidden[:, ids[:, 0] == 10]),
            noise_sha256=tensor_digest(hidden[:, ids[:, 0] == 0]),
        )
        if (ids[:, 0] == 20).any():
            trace["attention"]["control_sha256"] = tensor_digest(hidden[:, ids[:, 0] == 20])

    try:
        handles.append(pipe.vae.encoder.register_forward_pre_hook(encoder))
        handles.append(pipe.transformer.register_forward_pre_hook(attention, with_kwargs=True))
        yield trace
    finally:
        for handle in handles:
            handle.remove()
