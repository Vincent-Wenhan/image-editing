"""Run fake model outputs through the real runner, evaluator and legacy file schema."""

import json

import pytest
from PIL import Image

from src.cli import build_parser
from src.core.io import sha256
from src.core.types import EditInput, EditOutput
from src.evaluation.evaluate import run as evaluate
from src.evaluation.gallery import run as gallery
from src.inference import runner


@pytest.fixture
def inference_args(paired_data, tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "download.json").write_text('{"revision": "locked-test-revision"}')
    monkeypatch.setattr(runner, "require_lease", lambda: None)
    monkeypatch.setattr(runner, "versions", lambda: {"test": "cpu"})
    return build_parser().parse_args(
        [
            "infer",
            "--model",
            str(model),
            "--data-root",
            str(paired_data),
            "--manifest",
            str(paired_data / "dev/manifest.csv"),
            "--side",
            "48",
            "--output",
            str(tmp_path / "inference"),
        ]
    )


class TargetColorBackend:
    def __init__(self, model, lora):
        pass

    def edit(self, edit_input, controls, options):
        assert isinstance(edit_input, EditInput)
        assert not controls.maps
        assert not hasattr(edit_input, "target")
        return EditOutput(Image.new("RGB", (options.width, options.height), "red"), 1.0, 2.0, 3.0)


def test_records_metrics_and_gallery_keep_legacy_format(inference_args, paired_data, tmp_path):
    args = inference_args
    runner.run(args, backend_factory=TargetColorBackend)
    output = tmp_path / "inference"
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "complete"
    assert set(record) == {
        "created",
        "status",
        "config",
        "model",
        "packages",
        "manifest_sha256",
        "results",
    }
    assert record["manifest_sha256"] == sha256(args.manifest)
    assert "func" not in record["config"]
    assert set(record["results"][0]) == {
        "id",
        "seed",
        "width",
        "height",
        "seconds",
        "source_sha256",
        "prompt",
        "peak_allocated_gib",
        "peak_reserved_gib",
    }
    evaluation = build_parser().parse_args(
        [
            "evaluate",
            "--data-root",
            str(paired_data),
            "--manifest",
            args.manifest,
            "--predictions",
            str(output),
        ]
    )
    evaluate(evaluation)
    summary = json.loads((output / "metrics.json").read_text())
    assert summary["count"] == summary["manifest_count"] == 1
    assert summary["target_mae"] == 0
    assert summary["outside_mask_drift_proxy"] is None
    assert (output / "metrics.csv").read_text().startswith("id,target_mae,identity_target_mae,")
    gallery(
        build_parser().parse_args(
            [
                "gallery",
                "--data-root",
                str(paired_data),
                "--manifest",
                args.manifest,
                "--predictions",
                str(output),
                "--output",
                str(tmp_path / "gallery"),
            ]
        )
    )
    page = (tmp_path / "gallery/index.html").read_text()
    assert "&lt;square&gt;" in page and "&amp;" in page
    assert (tmp_path / "gallery/contact_00.jpg").is_file()
    with pytest.raises(FileExistsError):
        runner.run(args, backend_factory=TargetColorBackend)


def test_bad_backend_output_records_failure(inference_args, tmp_path):
    class WrongSizeBackend(TargetColorBackend):
        def edit(self, edit_input, controls, options):
            return EditOutput(Image.new("RGB", (1, 1)), 0, 0, 0)

    with pytest.raises(ValueError, match="dimensions"):
        runner.run(inference_args, backend_factory=WrongSizeBackend)
    record = json.loads((tmp_path / "inference/run.json").read_text())
    assert record["status"] == "failed" and record["results"] == []
    assert "dimensions" in record["error"]


def test_incomplete_legacy_run_cannot_be_evaluated(tmp_path):
    (tmp_path / "run.json").write_text('{"status":"failed","results":[]}')
    args = build_parser().parse_args(["evaluate", "--predictions", str(tmp_path)])
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(args)
