"""A small real Qwen transformer verifies updates and the public LoRA reload path."""

import copy
from types import SimpleNamespace

import pytest


def test_qwen_lora_update_save_reload_and_rng_continuation(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("diffusers")
    from pathlib import Path

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
    from peft import LoraConfig

    from src.qwen.inference import _pipeline_class
    from src.qwen.role_checkpoint import restore_checkpoint, save_checkpoint, update_audit
    from src.qwen.role_train import TARGET_MODULES

    if not Path("external/Inter-Edit/pipeline_qwenimage_edit_plus.py").exists():
        pytest.skip("Pinned upstream pipeline is optional in lean CI")
    torch.manual_seed(42)
    model = QwenImageTransformer2DModel(
        in_channels=8,
        out_channels=2,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
    )
    base = copy.deepcopy(model)
    model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=list(TARGET_MODULES)))
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name
    model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    pipeline_class = _pipeline_class()
    before = save_checkpoint(accelerator, model, pipeline_class, tmp_path, 0, optimizer, scheduler)
    kwargs = dict(
        hidden_states=torch.randn(1, 12, 8),
        timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.randn(1, 3, 32),
        encoder_hidden_states_mask=torch.ones(1, 3),
        img_shapes=[[(1, 2, 2)] * 3],
        return_dict=False,
    )
    model.train()
    loss = model(**kwargs)[0].float().square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    final = save_checkpoint(accelerator, model, pipeline_class, tmp_path, 1, optimizer, scheduler)
    expected_rng = torch.rand(4)
    assert restore_checkpoint(tmp_path / "checkpoint-1", model, optimizer, scheduler) == 1
    assert torch.equal(expected_rng, torch.rand(4))
    audit = update_audit(Path(before["checkpoint"]), Path(final["checkpoint"]))
    assert audit["changed_parameters"] > 0 and audit["parameter_update_l2"] > 0
    pipe = pipeline_class(
        transformer=base,
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        processor=None,
    )
    pipe.load_lora_weights(final["checkpoint"])
    model.eval()
    pipe.transformer.eval()
    with torch.no_grad():
        assert torch.equal(model(**kwargs)[0], pipe.transformer(**kwargs)[0])
