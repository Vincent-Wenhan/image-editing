"""Qwen manifests preserve the project split and the source/mask control contract."""

import json
from pathlib import Path

import pytest

from src.core.io import read_rows, sha256, write_json
from src.qwen.data import verify_condition_manifest, verify_manifest, write_manifest


def make_dataset_record(root: Path) -> None:
    write_json(
        root / "dataset.json",
        {
            "repo_id": "fixture",
            "revision": "fixed",
            "splits": ["train", "dev"],
            "status": "complete",
            "manifests": {
                split: sha256(root / split / "manifest.csv") for split in ("train", "dev")
            },
        },
    )


def test_qwen_manifest_matches_canonical_rows_and_is_target_free_in_controls(paired_data, tmp_path):
    make_dataset_record(paired_data)
    output = tmp_path / "train.json"
    record = write_manifest(paired_data, "train", output)

    payload = json.loads(output.read_text())
    row = read_rows(paired_data / "train/manifest.csv")[0]
    assert payload == [
        {
            "id": row["id"],
            "original_image_path": row["source_path"],
            "edited_image_path": row["target_path"],
            "mask_image_path": row["mask_path"],
            "instruction": row["instruction"],
        }
    ]
    assert record["control_order"] == ["original_image_path", "mask_image_path"]
    assert verify_manifest(output, root=paired_data)["ids"] == [row["id"]]

    payload[0]["mask_image_path"] = payload[0]["edited_image_path"]
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance changed"):
        verify_manifest(output, root=paired_data)


def test_qwen_manifest_rejects_test_split(paired_data, tmp_path):
    make_dataset_record(paired_data)
    (paired_data / "test").mkdir()
    with pytest.raises(ValueError, match="sealed Test"):
        write_manifest(paired_data, "dev", tmp_path / "dev.json")


def test_qwen_condition_verification_never_opens_target(paired_data, tmp_path):
    make_dataset_record(paired_data)
    output = tmp_path / "train.json"
    write_manifest(paired_data, "train", output)
    row = read_rows(paired_data / "train/manifest.csv")[0]
    (paired_data / row["target_path"]).unlink()
    assert verify_condition_manifest(output, root=paired_data)["ids"] == [row["id"]]


def test_qwen_train_refuses_to_start_without_a_gpu_lease(monkeypatch, tmp_path):
    from src.qwen import runner

    monkeypatch.delenv("IMAGE_EDIT_GPU_UUID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = type("Args", (), {"run": str(tmp_path), "lock": "configs/assets.lock.json"})()
    with pytest.raises(RuntimeError, match="image-edit gpu"):
        runner.train(args)


def test_qwen_seed_wrapper_repeats_cpu_random_state():
    import torch

    from src.qwen.seeded_train import _seed

    _seed(42)
    first = torch.rand(4)
    _seed(42)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_qwen_runner_uses_current_python_for_accelerate(tmp_path):
    import sys

    from src.qwen.runner import _accelerate_command

    command = _accelerate_command(tmp_path, "/tmp/train.yaml", 42)
    assert command[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--config") + 1] == "/tmp/train.yaml"


def test_qwen_inference_requires_a_gpu_lease(monkeypatch, tmp_path):
    from src.qwen import inference

    monkeypatch.delenv("IMAGE_EDIT_GPU_UUID", raising=False)
    with pytest.raises(RuntimeError, match="image-edit gpu"):
        inference.run(type("Args", (), {"run": str(tmp_path)})())
