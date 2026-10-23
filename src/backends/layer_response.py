"""Scoped paired layer measurements on the pinned Diffusers FLUX.2 implementation."""

from contextlib import contextmanager


class LayerResponses:
    def __init__(self, model, noise, text_tokens):
        self.noise, self.text_tokens = noise, text_tokens
        self.references, self.records = {}, {}
        self.modules = [("image_projection", model.x_embedder, False)]
        for name in ["transformer_blocks", "single_transformer_blocks"]:
            blocks = getattr(model, name)
            for index in sorted({0, (len(blocks) - 1) // 2, len(blocks) - 1}):
                self.modules.append(
                    (f"{name}.{index}", blocks[index], name == "single_transformer_blocks")
                )

    @contextmanager
    def observe(self, variant: str):
        handles = []
        self.records[variant] = {}
        for name, module, combined in self.modules:

            def capture(_m, _args, output, label=name, single=combined):
                # Double stream returns (text, image); single stream is concatenated.
                value = output[-1] if isinstance(output, tuple) else output
                if single and value.shape[1] != self.noise.numel():
                    value = value[:, self.text_tokens :]
                value = value[:, self.noise].detach()
                if not value.isfinite().all():
                    raise FloatingPointError("Non-finite layer diagnostic")
                if variant in {"original", "disabled"}:
                    self.references.setdefault(variant, {})[label] = value.clone()
                record = dict(rms=float(value.float().square().mean().sqrt()))
                for reference, layers in self.references.items():
                    scale = float(layers[label].float().square().mean().sqrt())
                    delta = float((value.float() - layers[label].float()).square().mean().sqrt())
                    record[f"relative_to_{reference}"] = delta / scale if scale else None
                    record[f"delta_to_{reference}"] = delta
                self.records[variant][label] = record

            handles.append(module.register_forward_hook(capture))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
