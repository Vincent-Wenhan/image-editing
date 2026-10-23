"""Read-only probes on the pinned trainer's real loss, inputs and learning gradients."""

from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np

from src.backends.feature_probe import fusion_measure
from src.core.io import read_rows
from src.datasets.scene_manifest import load_oracle_condition


def norm(value):
    return float(value.detach().float().square().mean().sqrt())


def step_hook_optimizer(optimizer):
    """Accelerate's Optimizer subclass stores step hooks on its inner optimizer."""
    seen = set()
    while hasattr(optimizer, "optimizer"):
        if id(optimizer) in seen:
            raise ValueError("Cyclic optimizer wrapper")
        seen.add(id(optimizer))
        optimizer = optimizer.optimizer
    return optimizer


class TrainingObserver:
    def __init__(self, state):
        import torch

        self.state, self.update = state, 0
        self.selected = set(state.metadata["observation"]["updates"])
        self.active, self.tensor_hooks = None, []
        self.record = dict(
            loss=[], updates=[], checkpoints=[], scope="actual unchanged upstream loss"
        )
        self.roles = {}
        for row in read_rows(state.root / "train/manifest.csv"):
            if row["id"] not in state.tokens:
                continue
            labels = np.asarray(load_oracle_condition(state.root, row).maps["roles"])
            h, w = labels.shape
            self.roles[row["id"]] = torch.from_numpy(
                np.stack(
                    [
                        (labels == k).reshape(h // 16, 16, w // 16, 16).mean((1, 3))
                        for k in (0, 1, 2)
                    ]
                ).astype(np.float32)
            )
        state.stats["observation"] = self.record

    def begin(self):
        self.update += 1
        self.active = (
            dict(update=self.update, layers={}, parameters={})
            if self.update in self.selected
            else None
        )

    def watch(self, tensor, sink, key, *, channels=False):
        if not tensor.requires_grad:
            return

        def gradient(g):
            if not g.isfinite().all():
                raise FloatingPointError("Non-finite observed learning gradient")
            sink[key] = dict(rms=norm(g), max_abs=float(g.abs().max()))
            if channels:
                sink[key]["channel_rms"] = g.detach().float().square().mean(1).sqrt().cpu().tolist()
                sink[key]["gradient_times_input"] = (
                    (g.detach().float() * tensor.detach().float()).sum(1).cpu().tolist()
                )

        self.tensor_hooks.append(tensor.register_hook(gradient))

    @contextmanager
    def forward(self, model, tokens, grid, ids):
        if self.active is None:
            yield tokens
            return
        n = grid[0] * grid[1]
        sample = self.active
        sample["ids"] = ids
        fields = tokens.detach().clone().requires_grad_(True)
        self.watch(fields, sample, "field_gradient", channels=True)
        observed, handles = {}, []

        def before(_m, _args, kwargs):
            observed.update(
                x=kwargs["x"].detach(), ids=kwargs["x_ids"].detach(), text=kwargs["ctx"].shape[1]
            )
            sample["timestep"] = kwargs["timesteps"].detach().float().cpu().tolist()

        def residual(_m, _args, output):
            observed["residual"] = output.detach()
            self.watch(output, sample, "residual_gradient")

        def projection(_m, args):
            sample["fusion"] = fusion_measure(
                observed["x"], args[0].detach(), observed["residual"], observed["ids"], grid
            )

        handles += [
            model.register_forward_pre_hook(before, with_kwargs=True),
            self.state.encoder.register_forward_hook(residual),
            model.img_in.register_forward_pre_hook(projection),
        ]
        modules = [("image_projection", model.img_in, False)]
        for name in ["double_blocks", "single_blocks"]:
            blocks = getattr(model, name)
            for index in sorted({0, (len(blocks) - 1) // 2, len(blocks) - 1}):
                modules.append((f"{name}.{index}", blocks[index], name == "single_blocks"))
        for name, module, single in modules:

            def layer(_m, _args, output, label=name, combined=single):
                value = output[0] if isinstance(output, tuple) else output
                start = observed["text"] if combined else 0
                value = value[:, start : start + n]
                sink = sample["layers"].setdefault(label, dict(feature_rms=norm(value)))
                # Register on the actual output, before slicing, so its gradient is on the graph.
                actual = output[0] if isinstance(output, tuple) else output
                if actual.requires_grad:

                    def grad(g, target=sink, offset=start):
                        target["gradient_rms"] = norm(g[:, offset : offset + n])

                    self.tensor_hooks.append(actual.register_hook(grad))

            handles.append(module.register_forward_hook(layer))
        try:
            yield fields
        finally:
            for handle in handles:
                handle.remove()

    def loss(self, process, prediction, noise, times, batch):
        import torch

        if self.active is None:
            return
        with torch.no_grad():
            target = (
                process.sd.get_loss_target(noise=noise, batch=batch, timesteps=times)
                if hasattr(process.sd, "get_loss_target")
                else noise - batch.latents
            )
            squared = (prediction.detach().float() - target.detach().float()).square().mean(1)
            masks = torch.stack([self.roles[Path(item.path).stem] for item in batch.file_items]).to(
                squared.device
            )
            denominator = masks.sum((2, 3))
            values = (squared[:, None] * masks).sum((2, 3)) / denominator.clamp_min(1)
            self.active["unweighted_region_velocity_mse"] = [
                [float(v) if float(d) > 0 else None for v, d in zip(row, den, strict=True)]
                for row, den in zip(values, denominator, strict=True)
            ]

    def gradients(self, process):
        import torch

        if self.active is None:
            return
        groups = {}
        for index, group in enumerate(process.optimizer.param_groups):
            values = [
                p.grad.detach().float().square().sum()
                for p in group["params"]
                if p.grad is not None
            ]
            groups[str(index)] = dict(
                post_clip_l2=float(torch.stack(values).sum().sqrt()) if values else 0.0,
                parameters=sum(p.numel() for p in group["params"]),
            )
        self.active["optimizer_groups"] = groups
        self.active["parameters"] = {
            name: dict(
                gradient_rms=norm(p.grad),
                **(
                    dict(channel_rms=p.grad.detach().float().square().mean(1).sqrt().cpu().tolist())
                    if name == "categories"
                    else {}
                ),
            )
            for name, p in self.state.encoder.named_parameters()
            if p.grad is not None
        }

    def finish(self, loss):
        self.record["loss"].append(dict(update=self.update, **loss))
        if self.active is not None:
            self.record["updates"].append(self.active)
        self.active = None
        for handle in self.tensor_hooks:
            handle.remove()
        self.tensor_hooks.clear()


def install_observer(state, trainer_type):
    """Wrappers delegate before recording, preserve return tensors, RNG and exceptions."""
    from src.backends.feature_checkpoint import save_encoder
    from src.core.io import sha256

    observer = TrainingObserver(state)
    for name in [
        "hook_train_loop",
        "calculate_loss",
        "post_save_hook",
        "hook_before_train_loop",
        "end_step_hook",
    ]:
        original = getattr(trainer_type, name)
        state.restore.append((trainer_type, name, original))

        def wrapper(process, *args, __name=name, __original=original, **kwargs):
            if __name == "hook_train_loop":
                observer.begin()
                handle = step_hook_optimizer(process.optimizer).register_step_pre_hook(
                    lambda opt, a, kw: observer.gradients(process)
                )
                try:
                    result = __original(process, *args, **kwargs)
                finally:
                    handle.remove()
                observer.finish(result)
                return result
            result = __original(process, *args, **kwargs)
            if __name == "calculate_loss":
                import inspect

                values = inspect.signature(__original).bind(process, *args, **kwargs).arguments
                observer.loss(
                    process,
                    values["noise_pred"],
                    values["noise"],
                    values["timesteps"],
                    values["batch"],
                )
            elif __name == "post_save_hook":
                lora = Path(args[0])
                saved = save_encoder(
                    lora.with_suffix(".features.safetensors"),
                    state.encoder,
                    lora,
                    state.metadata["mode"],
                )
                observer.record["checkpoints"].append(
                    dict(
                        update=observer.update,
                        lora=str(lora),
                        lora_sha256=sha256(lora),
                        encoder=saved,
                    )
                )
            elif __name == "hook_before_train_loop":
                process.save(step=0)
            elif __name == "end_step_hook":
                policy = state.metadata["observation"]
                if (
                    observer.update % policy["save_every"] == 0
                    and observer.update < policy["steps"]
                ):
                    process.save(step=observer.update)
            return result

        setattr(trainer_type, name, wraps(original)(wrapper))
    return observer
