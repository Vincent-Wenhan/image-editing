"""Research selection, neutral captions and loss-preserving observations, CPU only."""

import copy
import json
from types import SimpleNamespace

import pytest

from src.core.io import read_rows, sha256, write_json, write_rows
from src.datasets.research_view import create_view, selected_rows, verify_view
from tests.test_feature_training import seal_fixture


def three_strengths(data):
    path = data / "train/manifest.csv"
    rows = read_rows(path)
    middle = dict(rows[1], id=rows[1]["id"] + "_half", strength="0.5")
    # Use real fixture pixels/hashes; this is a selection unit fixture, not a new render.
    write_rows(path, [rows[0], middle, rows[1]])
    dataset = json.loads((data / "dataset.json").read_text())
    dataset["manifests"]["train"] = sha256(path)
    write_json(data / "dataset.json", dataset)


def test_view_cannot_train_on_dev_or_change_pairing_caption_or_parent(scene_data, tmp_path):
    three_strengths(scene_data)
    seal = seal_fixture(scene_data, tmp_path)
    view = tmp_path / "view"
    record = create_view(scene_data, seal, view, "Follow the spatial condition.", 1)
    assert len(record["ids"]) == 3 and all(s.startswith("train_") for s in record["ids"])
    assert verify_view(view / "view.json") == record
    with pytest.raises(FileExistsError):
        create_view(scene_data, seal, view, "same", 1)
    caption = view / "target" / f"{record['ids'][0]}.txt"
    caption.write_text("Leaked target location")
    with pytest.raises(ValueError, match="caption"):
        verify_view(view / "view.json")
    caption.write_text(record["prompt"])
    source = view / "reference" / f"{record['ids'][0]}.png"
    source.write_bytes(b"wrong reference")
    with pytest.raises(ValueError, match="pairing"):
        verify_view(view / "view.json")
    with pytest.raises(ValueError, match="complete Train"):
        selected_rows(scene_data, 2)


def test_observation_preserves_real_loss_gradients_rng_and_cleans_hooks(scene_data):
    torch = pytest.importorskip("torch")
    from src.backends.feature_fusion import feature_fusion
    from src.conditioning.feature_encoder import build_encoder
    from src.conditioning.features import EncoderConfig
    from src.datasets.features import load_features
    from src.training.feature_observer import TrainingObserver

    class Double(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(16, 16)

        def forward(self, image, text):
            return self.layer(image).tanh(), text

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.img_in = torch.nn.Linear(128, 16)
            self.double_blocks = torch.nn.ModuleList([Double(), Double()])
            self.single_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)]
            )
            self.out = torch.nn.Linear(16, 128)

        def forward(self, *, x, x_ids, ctx, timesteps):
            image = self.img_in(x)
            for layer in self.double_blocks:
                image, ctx = layer(image, ctx)
            image = torch.cat([ctx, image], 1)
            for layer in self.single_blocks:
                image = layer(image).tanh()
            return self.out(image[:, 2:18]).transpose(1, 2).reshape(1, 128, 4, 4)

    model, encoder = Model(), build_encoder(EncoderConfig())
    with torch.no_grad():
        encoder.output.weight.normal_(std=0.01)
    reference, ref_encoder = copy.deepcopy(model), copy.deepcopy(encoder)
    row = read_rows(scene_data / "train/manifest.csv")[1]
    tokens = torch.from_numpy(load_features(scene_data, row, "full_role").tokens())[None]
    state = SimpleNamespace(
        root=scene_data,
        tokens={row["id"]: tokens[0]},
        encoder=encoder,
        metadata=dict(observation=dict(updates=[1])),
        stats={},
    )
    observer = TrainingObserver(state)
    ids = torch.cartesian_prod(
        torch.tensor([0, 10]), torch.arange(4), torch.arange(4), torch.tensor([0])
    )[None]
    inputs = dict(
        x=torch.randn(1, 32, 128),
        x_ids=ids,
        ctx=torch.randn(1, 2, 16),
        timesteps=torch.tensor([0.5]),
    )
    rng = torch.random.get_rng_state()
    with feature_fusion(reference, ref_encoder, tokens, (4, 4), flavor="bfl"):
        plain = reference(**inputs)
    plain_loss = plain.square().mean()
    plain_loss.backward()
    observer.begin()
    with observer.forward(model, tokens, (4, 4), [row["id"]]) as fields:
        with feature_fusion(model, encoder, fields, (4, 4), flavor="bfl"):
            observed = model(**inputs)
    loss = observed.square().mean()
    batch = SimpleNamespace(
        latents=torch.zeros_like(observed), file_items=[SimpleNamespace(path=row["target_path"])]
    )
    process = SimpleNamespace(
        sd=SimpleNamespace(),
        optimizer=torch.optim.SGD([*model.parameters(), *encoder.parameters()], lr=0.01),
    )
    observer.loss(process, observed, torch.zeros_like(observed), inputs["timesteps"], batch)
    loss.backward()
    observer.gradients(process)
    observer.finish(dict(loss=float(loss.detach())))
    assert torch.equal(loss, plain_loss)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for left, right in zip(
        [*model.parameters(), *encoder.parameters()],
        [*reference.parameters(), *ref_encoder.parameters()],
        strict=True,
    ):
        assert torch.equal(left.grad, right.grad)
    update = observer.record["updates"][0]
    assert len(update["layers"]) == 5
    assert all("gradient_rms" in v for v in update["layers"].values())
    assert len(update["field_gradient"]["channel_rms"][0]) == 5
    assert "unweighted_region_velocity_mse" in update
    assert not observer.tensor_hooks and not model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in model.modules())


def test_overfit_budget_is_bounded_and_test_cannot_be_declared():
    from src.core.config import load_yaml
    from src.core.paths import CONFIG_ROOT
    from src.training.learnability_study import verify_protocol

    p = load_yaml(CONFIG_ROOT / "experiments/role_learnability.yaml")
    verify_protocol(p)
    for key, value in [
        ("steps", 10000),
        ("train_groups", 20),
        ("methods", ["full_role"]),
        ("train_seeds", [307, 307]),
    ]:
        wrong = copy.deepcopy(p)
        wrong[key] = value
        with pytest.raises(ValueError, match="preregistered"):
            verify_protocol(wrong)


def test_step_observer_unwraps_accelerated_optimizer_without_changing_step():
    torch = pytest.importorskip("torch")
    from src.training.feature_observer import step_hook_optimizer

    class Wrapped(torch.optim.Optimizer):
        def __init__(self, optimizer):
            self.optimizer = (
                optimizer  # Accelerate does not initialize Optimizer hook dictionaries.
            )

        def step(self):
            return self.optimizer.step()

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    raw = torch.optim.SGD([parameter], lr=0.1)
    wrapped = Wrapped(raw)
    observed = []
    handle = step_hook_optimizer(wrapped).register_step_pre_hook(
        lambda *args: observed.append(float(parameter.grad))
    )
    parameter.square().backward()
    wrapped.step()
    handle.remove()
    assert observed == [2.0] and torch.allclose(parameter.detach(), torch.tensor(0.8))


def test_checkpoint_pair_uses_actual_lora_write_not_optimizer_callback(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from src.backends.feature_checkpoint import load_encoder
    from src.conditioning.feature_encoder import build_encoder
    from src.conditioning.features import EncoderConfig
    from src.training.feature_observer import install_pair_saver

    class Network:
        def save_weights(self, path, *, dtype, metadata):
            path.write_bytes(b"LoRA fixture, not actual trained weights")
            return metadata

    state = SimpleNamespace(
        encoder=build_encoder(EncoderConfig()),
        metadata=dict(mode="full_role"),
        observer=SimpleNamespace(update=0, record=dict(checkpoints=[])),
        restore=[],
    )
    network = Network()
    original = network.save_weights
    install_pair_saver(state, network)
    lora = tmp_path / "network_000000000.safetensors"
    assert network.save_weights(lora, dtype="fixture", metadata={"step": 0}) == {"step": 0}
    (tmp_path / "optimizer.pt").write_bytes(b"optimizer fixture")
    record = state.observer.record["checkpoints"][0]
    assert record["lora_sha256"] == sha256(lora)
    assert not (tmp_path / "optimizer.features.safetensors").exists()
    _, meta = load_encoder(tmp_path / "network_000000000.features.safetensors", lora)
    assert meta["lora_sha256"] == sha256(lora)
    owner, name, saved = state.restore.pop()
    setattr(owner, name, saved)
    assert network.save_weights == original
