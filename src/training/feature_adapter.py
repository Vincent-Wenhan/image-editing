"""Versioned extra-optimizer/forward hooks; the pinned trainer's loss is unchanged."""

import hashlib
from functools import wraps
from pathlib import Path

from src.backends.feature_checkpoint import save_encoder
from src.backends.feature_fusion import feature_fusion
from src.conditioning.feature_encoder import build_encoder
from src.conditioning.features import EncoderConfig
from src.core.io import read_rows
from src.datasets.features import load_features
from src.training.flux2_compat import SUPPORTED_TRAINER_REVISION


class FeatureTraining:
    def __init__(self, metadata: dict, seed: int):
        import torch

        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Feature training needs an explicit nonnegative 32-bit seed")
        self.metadata = metadata
        self.seed = seed
        self.root = Path(metadata["data_root"])
        view = metadata.get("training_view")
        self.training_folder = Path(view).parent / "target" if view else self.root / "train/target"
        if view:
            from src.datasets.research_view import verify_view

            selected = set(verify_view(Path(view), root=self.root)["ids"])
        else:
            selected = None
        self.tokens = {
            row["id"]: torch.from_numpy(load_features(self.root, row, metadata["mode"]).tokens())
            for row in read_rows(self.root / "train/manifest.csv")
            if selected is None or row["id"] in selected
        }
        self.encoder = None
        self.handles, self.restore = [], []
        self.stats = dict(
            batches=0, fusion_calls=0, gradients={}, input_trace=[], initial_seed=seed
        )
        self.observer = None

    def initialize(self, process, params):
        import torch

        if self.encoder is not None:
            raise RuntimeError("Feature optimizer already initialized")
        params = [dict(group, params=list(group["params"])) for group in params]
        if self.observer is not None:
            fingerprint = hashlib.sha256()
            for group in params:
                for parameter in group["params"]:
                    fingerprint.update(parameter.detach().float().cpu().numpy().tobytes())
            self.stats["network_initial_sha256"] = fingerprint.hexdigest()
        # Do not change the pretrained LoRA/data-loader random stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.encoder = build_encoder(EncoderConfig(**self.metadata["encoder"]))
        self.stats["initial_sha256"] = hashlib.sha256(
            b"".join(
                value.detach().numpy().tobytes() for value in self.encoder.state_dict().values()
            )
        ).hexdigest()
        self.encoder.to(device=process.device_torch, dtype=torch.float32)
        for name, parameter in self.encoder.named_parameters():

            def observe(gradient, key=name):
                if not gradient.isfinite().all():
                    raise FloatingPointError(f"Non-finite feature gradient: {key}")
                record = self.stats["gradients"].setdefault(key, dict(calls=0, max_abs=0.0))
                record["calls"] += 1
                record["max_abs"] = max(record["max_abs"], float(gradient.abs().max()))
                return gradient

            self.handles.append(parameter.register_hook(observe))
        parameters = list(self.encoder.parameters())
        self.stats["parameters"] = sum(p.numel() for p in parameters)
        self.stats["total_trainable_parameters"] = self.stats["parameters"] + sum(
            p.numel() for group in params for p in group["params"]
        )
        return [*params, dict(params=parameters, lr=self.metadata["lr"])]

    def inputs(self, batch, latent):
        import torch

        items = batch.file_items
        if len(items) != latent.shape[0]:
            raise ValueError("Feature metadata and training batch differ")
        selected = []
        for item in items:
            identifier = Path(item.path).stem
            expected = self.training_folder / f"{identifier}.png"
            if (
                identifier not in self.tokens
                or Path(item.path).resolve() != expected.resolve()
                or item.flip_x
                or item.flip_y
                or item.crop_x
                or item.crop_y
            ):
                raise ValueError("Unknown, flipped or cropped feature example")
            selected.append(self.tokens[identifier])
        controls = batch.control_tensor_list
        if controls is None:
            controls = [[item] for item in batch.control_tensor]
        shape = (3, latent.shape[-2] * 16, latent.shape[-1] * 16)
        if len(controls) != len(items) or any(
            len(c) != 1
            or tuple(c[0].shape) != shape
            or not c[0].isfinite().all()
            or c[0].min() < 0
            or c[0].max() > 1
            for c in controls
        ):
            raise ValueError("Features require one aligned unit-range source reference")
        self.stats["batches"] += 1
        self.stats["input_trace"].append(
            dict(
                ids=[Path(item.path).stem for item in items],
                latent_sha256=hashlib.sha256(
                    latent.detach().float().cpu().numpy().tobytes()
                ).hexdigest(),
            )
        )
        return torch.stack(selected), tuple(latent.shape[-2:])

    def save(self, path: Path, lora: Path):
        gradients = self.stats["gradients"]
        if (
            self.encoder is None
            or len(gradients) != len(list(self.encoder.parameters()))
            or not any(
                value["max_abs"] > 0
                for key, value in gradients.items()
                if not key.startswith("output.")
            )
        ):
            raise RuntimeError("Feature encoder did not receive finite upstream learning gradients")
        if self.observer is not None and path.exists():
            from src.core.io import sha256

            owned = [
                r["encoder"]
                for r in self.observer.record["checkpoints"]
                if Path(r["encoder"]["path"]).resolve() == path.resolve()
                and r["lora_sha256"] == sha256(lora)
            ]
            if len(owned) != 1 or owned[0]["sha256"] != sha256(path):
                raise ValueError("Existing observed checkpoint differs from the saved pair")
            return owned[0]
        return save_encoder(path, self.encoder, lora, self.metadata["mode"])

    def close(self):
        if self.observer is not None:
            for handle in self.observer.tensor_hooks:
                handle.remove()
            self.observer.tensor_hooks.clear()
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        self.restore.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_features(model_type, metadata: dict, revision: str, *, seed: int, trainer_type=None):
    if revision != SUPPORTED_TRAINER_REVISION:
        raise ValueError("Feature adapter requires the pinned ai-toolkit version")
    if getattr(model_type.get_noise_prediction, "_image_edit_feature_adapter", False):
        raise RuntimeError("Feature training adapter already installed")
    if trainer_type is None:
        from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

        trainer_type = DiffusionTrainer
    state = FeatureTraining(metadata, seed)
    extra = trainer_type.hook_add_extra_train_params
    noise = model_type.get_noise_prediction

    @wraps(extra)
    def add(process, params):
        return state.initialize(process, extra(process, params))

    @wraps(noise)
    def predict(model, latent_model_input, *args, batch, **kwargs):
        tokens, grid = state.inputs(batch, latent_model_input)
        timestep = args[0] if args else kwargs.get("timestep")
        if timestep is not None:
            state.stats["input_trace"][-1]["timestep"] = timestep.detach().float().cpu().tolist()
        if state.encoder is None:
            raise RuntimeError("Feature encoder optimizer was not initialized")
        from contextlib import nullcontext

        context = (
            state.observer.forward(
                model.transformer, tokens, grid, state.stats["input_trace"][-1]["ids"]
            )
            if state.observer
            else nullcontext(tokens)
        )
        with context as actual_tokens:
            with feature_fusion(
                model.transformer, state.encoder, actual_tokens, grid, flavor="bfl"
            ) as stats:
                result = noise(model, latent_model_input, *args, batch=batch, **kwargs)
        if (
            stats["calls"] != 1
            or stats["source_tokens"] != [grid[0] * grid[1]] * tokens.shape[0]
            or stats["reference_times"] != [10]
        ):
            raise RuntimeError("The trainer silently skipped role feature fusion")
        state.stats["fusion_calls"] += stats["calls"]
        state.stats["feature_tokens"] = stats["tokens"]
        state.stats["source_tokens"] = stats["source_tokens"]
        return result

    predict._image_edit_feature_adapter = True
    state.restore = [
        (trainer_type, "hook_add_extra_train_params", extra),
        (model_type, "get_noise_prediction", noise),
    ]
    if metadata.get("observation"):
        from src.training.feature_observer import install_observer

        state.observer = install_observer(state, trainer_type)
    trainer_type.hook_add_extra_train_params = add
    model_type.get_noise_prediction = predict
    return state
