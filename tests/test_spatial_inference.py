"""Real adapter forwarding and run provenance without CUDA or model loading."""

import contextlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.backends.flux2 import Flux2Backend
from src.backends.flux2_conditions import reference_images
from src.cli import build_parser
from src.conditioning.controls import ControlBundle, ControlKind
from src.conditioning.static import MaskSpec
from src.core.io import read_rows
from src.core.types import EditInput, EditOutput, GenerationOptions
from src.datasets.control_export import export_controls
from src.evaluation.provenance import verify_run
from src.inference import runner


def test_actual_adapter_forwards_both_images_in_order(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        Generator=lambda _: SimpleNamespace(manual_seed=lambda _: object()),
        inference_mode=contextlib.nullcontext,
        cuda=SimpleNamespace(
            synchronize=lambda: None, max_memory_allocated=lambda: 1, max_memory_reserved=lambda: 2
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    backend = object.__new__(Flux2Backend)
    source = Image.new("RGB", (64, 64), "blue")

    class Pipeline:
        image_processor = SimpleNamespace(preprocess=lambda *args, **kwargs: None)

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[source])

    backend.pipe = Pipeline()
    edit = EditInput(source, "edit")
    options = GenerationOptions(64, 64, 2, 4, 42)
    controls = ControlBundle(ControlKind.STATIC_MASK, {"mask": Image.new("L", (64, 64), 255)})
    backend.edit(edit, controls, options)
    assert calls[0]["image"][0] is source
    assert len(calls[0]["image"]) == 2 and np.asarray(calls[0]["image"][1]).min() == 255
    backend.edit(edit, ControlBundle(), options)
    assert calls[1]["image"] is source  # Keep the exact legacy single-image call.
    with pytest.raises(ValueError, match="Compile"):
        reference_images(
            EditInput(source, "edit", mask=Image.new("L", source.size)), ControlBundle(), options
        )
    with pytest.raises(ValueError, match="dimensions"):
        reference_images(edit, controls, GenerationOptions(128, 64, 2, 4, 42))


def test_mask_runner_never_opens_target_and_records_consumed_conditions(
    spatial_data, tmp_path, monkeypatch
):
    control_root = tmp_path / "controls"
    export_controls(spatial_data, control_root, MaskSpec(ControlKind.STATIC_MASK))
    model = tmp_path / "model"
    model.mkdir()
    (model / "download.json").write_text('{"revision":"fixed"}')
    args = build_parser().parse_args(
        [
            "infer",
            "--data-root",
            str(spatial_data),
            "--manifest",
            str(spatial_data / "dev/manifest.csv"),
            "--model",
            str(model),
            "--output",
            str(tmp_path / "run"),
            "--side",
            "64",
            "--limit",
            "1",
            "--control",
            "static_mask",
            "--control-root",
            str(control_root),
        ]
    )
    monkeypatch.setattr(runner, "require_lease", lambda: None)
    monkeypatch.setattr(runner, "versions", lambda: {"test": "cpu"})
    calls = []

    class Backend:
        def __init__(self, *args):
            pass

        def edit(self, edit, controls, options):
            calls.append(controls)
            assert controls.kind == ControlKind.STATIC_MASK
            assert not hasattr(edit, "target")
            return EditOutput(edit.source.copy(), 1, 2, 3)

    # Forbid supervision reads even during control snapshot validation.
    original_open = Image.open

    def open_without_target(path, *args, **kwargs):
        if "/target/" in str(path):
            raise AssertionError("Target leaked into inference")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Image, "open", open_without_target)
        runner.run(args, backend_factory=Backend)
    record = json.loads((tmp_path / "run/run.json").read_text())
    verify_run(spatial_data, args.manifest, record)
    assert len(calls) == 1
    assert record["results"][0]["conditioning"] == dict(calls[0].metadata)
    row = read_rows(spatial_data / "dev/manifest.csv")[0]
    Image.new("RGB", (64, 64)).save(control_root / "dev/mask_rgb" / f"{row['id']}.png")
    with pytest.raises(ValueError, match="SHA256"):
        verify_run(spatial_data, args.manifest, record)
