"""Numerical contracts for the pinned CJT preprocessing/flow path."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def test_dimensions_match_pinned_area_rule_before_rounding():
    from src.qwen.role_data import _dimensions

    for ratio in (1, 1.7, 2.5, 0.4):
        width = math.sqrt(1024**2 * ratio)
        expected = round(width / 32) * 32, round(width / ratio / 32) * 32
        assert _dimensions(1024**2, ratio) == expected


def test_vae_cache_uses_sample_and_preserves_channel_temporal_axes():
    torch = pytest.importorskip("torch")
    from src.qwen.role_data import _encode_vae

    raw = torch.arange(16 * 4 * 4).reshape(1, 16, 1, 4, 4).float()
    seen = {}

    def preprocess(image, height, width):
        seen["size"] = width, height
        return torch.zeros(1, 3, height, width)

    def encode(image):
        assert image.shape == (1, 3, 1, 1024, 1024)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: raw))

    pipeline = SimpleNamespace(image_processor=SimpleNamespace(preprocess=preprocess))
    vae = SimpleNamespace(config=SimpleNamespace(z_dim=16), encode=encode)
    actual = _encode_vae(pipeline, vae, Image.new("RGB", (512, 512)), "cpu", torch.float32)
    assert torch.equal(actual, raw[0]) and seen["size"] == (1024, 1024)


def test_batch_schedule_is_repeatable_and_each_epoch_covers_every_sample():
    from src.qwen.role_train import batch_schedule

    sequence = batch_schedule(6, 42, 240)
    assert sequence[:120] == batch_schedule(6, 42, 120)
    assert all(sorted(sequence[i : i + 6]) == list(range(6)) for i in range(0, 240, 6))


def test_lora_targets_match_pinned_inter_edit():
    from src.qwen.role_train import TARGET_MODULES

    path = Path("external/Inter-Edit/train_qwen_edit_lora.py")
    if not path.exists():
        pytest.skip("Pinned external checkout is optional in lean CI")
    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LoraConfig"
    ]
    assert len(calls) == 1
    target = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "target_modules")
    assert set(ast.literal_eval(target)) == set(TARGET_MODULES)


def test_role_flow_matches_upstream_packing_and_velocity_with_cpu_timestep_table():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from src.qwen.role_flow import flow_error, packed_inputs

    # Execute only the two pure packing functions from the locally pinned upstream.
    path = Path("external/Inter-Edit/pipeline_qwenimage_edit_plus.py")
    if not path.exists():
        pytest.skip("Pinned external checkout is optional in lean CI")
    tree = ast.parse(path.read_text())
    functions = [
        node
        for parent in ast.walk(tree)
        if isinstance(parent, ast.ClassDef)
        for node in parent.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_pack_latents", "_unpack_latents"}
    ]
    scope = {}
    for node in functions:
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    pipeline = SimpleNamespace(**scope)
    batch = {
        "target": torch.zeros(1, 2, 1, 4, 4),
        "controls": torch.stack([torch.ones(2, 1, 4, 4), torch.full((2, 1, 4, 4), 3)])[None],
        "prompt": torch.ones(1, 3, 2),
        "prompt_mask": torch.ones(1, 3),
    }
    mean = torch.tensor([1, 2]).view(1, 2, 1, 1, 1)
    scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=4),
        timesteps=torch.tensor([1000.0, 750.0, 500.0, 250.0]),
        sigmas=torch.tensor([1.0, 0.75, 0.5, 0.25]),
    )
    torch.manual_seed(42)
    values = packed_inputs(
        batch, pipeline, mean, torch.ones_like(mean), scheduler, "cpu", torch.float32
    )
    assert values["noisy"].shape == (1, 12, 8)
    expected_controls = torch.cat(
        [pipeline._pack_latents((batch["controls"][:, i] - mean), 1, 2, 4, 4) for i in range(2)],
        dim=1,
    )
    assert torch.equal(values["noisy"][:, 4:], expected_controls)
    expected_velocity = values["noise"] - values["target"]
    prediction = pipeline._pack_latents(expected_velocity, 1, 2, 4, 4)

    def transformer(**kwargs):
        return (torch.cat([prediction, expected_controls], dim=1),)

    assert torch.count_nonzero(flow_error(transformer, pipeline, values, 8)) == 0
