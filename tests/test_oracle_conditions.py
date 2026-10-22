"""Oracle is a separate explicit model interface, never an ordinary editing input."""

import numpy as np
import pytest
from PIL import Image

from src.backends.flux2 import Flux2Backend
from src.backends.flux2_oracle import oracle_references
from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.oracle import oracle_bundle, role_rgb
from src.core.types import EditInput, EditOutput, GenerationOptions


def bundle():
    labels = np.full((64, 64), 2, dtype=np.uint8)
    labels[:16, :16], labels[16:32, :16], labels[32:48, :16] = 0, 1, 255
    valid = (labels != 255).astype(np.uint8)
    return oracle_bundle(
        labels,
        valid,
        metadata={
            key: "a" * 64
            for key in ("role_sha256", "valid_sha256", "evidence_sha256", "source_sha256")
        },
    )


def test_oracle_channels_ignore_and_values_are_explicit():
    control = bundle()
    image = np.asarray(role_rgb(control.maps["roles"]))
    assert image[0, 0].tolist() == [255, 0, 0]
    assert image[16, 0].tolist() == [0, 255, 0]
    assert image[48, 0].tolist() == [0, 0, 255]
    assert image[32, 0].tolist() == [0, 0, 0]
    assert control.metadata["purpose"] == "oracle_upper_bound"
    with pytest.raises(NotImplementedError, match="hard"):
        role_rgb(np.full((64, 64, 3), 1 / 3, dtype=np.float32))
    with pytest.raises(ValueError, match="SHA256"):
        oracle_bundle(np.ones((64, 64), dtype=np.uint8), np.ones((64, 64)), metadata={})
    with pytest.raises(ValueError, match="Ignored"):
        oracle_bundle(control.maps["roles"], np.ones((64, 64)), metadata={})


def test_default_backend_rejects_oracle_and_opt_in_forwards_only_source_and_roles():
    control = bundle()
    edit = EditInput(Image.new("RGB", (64, 64), "blue"), "move right")
    options = GenerationOptions(64, 64, 4, 4, 42)
    backend = object.__new__(Flux2Backend)
    calls = []

    def generate(edit_input, images, options, *, matched):
        calls.append((images, matched))
        assert edit_input is edit and not hasattr(edit_input, "target")
        return EditOutput(images[0], 0, 0, 0)

    backend._generate = generate
    with pytest.raises(ValueError, match="ordinary inference"):
        backend.edit(edit, control, options)
    with pytest.raises(ValueError, match="allow_oracle"):
        backend.edit_oracle(edit, control, options)
    backend.edit_oracle(edit, control, options, allow_oracle=True)
    images, matched = calls[0]
    assert matched and images[0] is edit.source and len(images) == 2
    assert np.array_equal(images[1], role_rgb(control.maps["roles"]))
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["provenance", "geometry", "strength", "kind"])
def test_oracle_model_rejects_inconsistent_or_unused_inputs(change):
    control = bundle()
    edit = EditInput(
        Image.new("RGB", (64, 64)), "move right", strength=0.5 if change == "strength" else None
    )
    options = GenerationOptions(128 if change == "geometry" else 64, 64, 4, 4, 42)
    if change == "provenance":
        control = ControlBundle(ControlKind.ORACLE_ROLE, control.maps)
    elif change == "kind":
        control = ControlBundle()
    with pytest.raises((ValueError, NotImplementedError)):
        oracle_references(edit, control, options, allow_oracle=True)


def test_reference_trace_captures_real_cpu_tensors_and_removes_hooks_on_failure():
    from types import SimpleNamespace

    from src.backends.reference_trace import trace_references

    torch = pytest.importorskip("torch")

    class Transformer(torch.nn.Module):
        def forward(self, hidden_states, img_ids):
            return hidden_states

    pipe = SimpleNamespace(
        vae=SimpleNamespace(encoder=torch.nn.Identity()), transformer=Transformer()
    )
    with pytest.raises(RuntimeError, match="generation failed"), trace_references(pipe) as trace:
        pipe.vae.encoder(torch.tensor([[[[1.0]], [[-1.0]], [[-1.0]]]]))
        ids = torch.tensor([[[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]]])
        hidden = torch.arange(6.0).reshape(1, 3, 2)
        pipe.transformer(hidden_states=hidden, img_ids=ids)
        assert trace["vae"][0]["binary"] and trace["vae"][0]["positive_channels_max"] == 1
        assert trace["attention"]["offsets"] == [0, 10, 20]
        assert trace["attention"]["source_sha256"] != trace["attention"]["control_sha256"]
        raise RuntimeError("generation failed")
    assert not pipe.vae.encoder._forward_pre_hooks and not pipe.transformer._forward_pre_hooks
