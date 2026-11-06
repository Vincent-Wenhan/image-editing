"""Qwen manifests preserve the project split and the source/mask control contract."""

import json
from pathlib import Path

import pytest
from PIL import Image

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
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


def test_qwen_manifest_rejects_unsupported_requested_split(paired_data, tmp_path):
    make_dataset_record(paired_data)
    with pytest.raises(ValueError, match="train or dev"):
        write_manifest(paired_data, "test", tmp_path / "test.json")


def test_qwen_condition_verification_ignores_sealed_test_directory(paired_data, tmp_path):
    make_dataset_record(paired_data)
    output = tmp_path / "train.json"
    write_manifest(paired_data, "train", output)
    (paired_data / "test").mkdir()
    assert verify_condition_manifest(output, root=paired_data)["split"] == "train"


def test_qwen_condition_verification_never_opens_target(paired_data, tmp_path):
    make_dataset_record(paired_data)
    output = tmp_path / "train.json"
    write_manifest(paired_data, "train", output)
    row = read_rows(paired_data / "train/manifest.csv")[0]
    (paired_data / row["target_path"]).unlink()
    assert verify_condition_manifest(output, root=paired_data)["ids"] == [row["id"]]


def make_control_snapshot(data_root: Path, output: Path) -> Path:
    entries = {}
    for split in ("train", "dev"):
        row = read_rows(data_root / split / "manifest.csv")[0]
        relative = f"{split}/mask_rgb/{row['id']}.png"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(data_path(data_root, row["mask_path"])) as image:
            image.convert("RGB").save(path)
        entries[split] = {
            "manifest_sha256": sha256(data_root / split / "manifest.csv"),
            "entries": {
                row["id"]: {
                    "path": relative,
                    "sha256": sha256(path),
                    "source_sha256": row["source_sha256"],
                }
            },
        }
    write_json(
        output / "controls.json",
        {
            "schema_version": 2,
            "kind": "dynamic_binary",
            "purpose": "oracle_upper_bound",
            "order": ["source", "mask_rgb"],
            "splits": entries,
        },
    )
    return output


def test_qwen_manifest_uses_verified_external_condition_without_target(paired_data, tmp_path):
    make_dataset_record(paired_data)
    controls = make_control_snapshot(paired_data, tmp_path / "controls")
    output = tmp_path / "train.json"
    write_manifest(paired_data, "train", output, control_root=controls)
    payload = json.loads(output.read_text())
    assert payload[0]["mask_image_path"] == str(
        (controls / "train/mask_rgb/train_train_t1.png").resolve()
    )
    assert verify_manifest(output, root=paired_data)["condition"]["kind"] == "dynamic_binary"
    row = read_rows(paired_data / "train/manifest.csv")[0]
    (paired_data / row["target_path"]).unlink()
    assert verify_condition_manifest(output, root=paired_data)["condition"]["kind"] == (
        "dynamic_binary"
    )


def test_qwen_inference_uses_manifest_control_path(paired_data, tmp_path):
    make_dataset_record(paired_data)
    controls = make_control_snapshot(paired_data, tmp_path / "controls")
    manifest = tmp_path / "dev.json"
    write_manifest(paired_data, "dev", manifest, control_root=controls)
    from src.qwen.inference import _row

    sample, checks = _row({"data_root": str(paired_data), "dev_manifest": str(manifest)}, None)
    assert sample["mask_image_path"] == str((controls / "dev/mask_rgb/dev_dev_t1.png").resolve())
    assert checks["id"] == sample["id"]


def test_qwen_train_refuses_to_start_without_a_gpu_lease(monkeypatch, tmp_path):
    from src.qwen import runner

    monkeypatch.delenv("IMAGE_EDIT_GPU_UUID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = type("Args", (), {"run": str(tmp_path), "lock": "configs/assets.lock.json"})()
    with pytest.raises(RuntimeError, match="image-edit gpu"):
        runner.train(args)


def test_qwen_seed_wrapper_repeats_cpu_random_state():
    torch = pytest.importorskip("torch")

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


def test_qwen_inference_selects_latest_checkpoint(tmp_path):
    from src.qwen.inference import _checkpoint_path

    older = tmp_path / "checkpoints/checkpoint-2/pytorch_lora_weights.safetensors"
    latest = tmp_path / "checkpoints/checkpoint-20/pytorch_lora_weights.safetensors"
    ignored = tmp_path / "checkpoints/checkpoint-final/pytorch_lora_weights.safetensors"
    for path in (older, latest, ignored):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    assert _checkpoint_path(tmp_path) == latest
    assert _checkpoint_path(tmp_path, str(older)) == older


def test_qwen_inference_matches_source_geometry():
    from src.qwen.inference import _match_source_size

    image = Image.new("RGB", (1024, 1024), "red")
    resized, changed = _match_source_size(image, (512, 512))
    assert resized.size == (512, 512)
    assert changed is True
    same, changed = _match_source_size(image, (1024, 1024))
    assert same.size == (1024, 1024)
    assert changed is False


def test_qwen_source_only_baseline_is_registered_in_cli():
    from src.cli import build_parser

    args = build_parser().parse_args(
        [
            "qwen",
            "infer",
            "--run",
            "run",
            "--output",
            "out",
            "--source-only",
            "--all-dev",
            "--steps",
            "20",
        ]
    )
    assert args.source_only is True
    assert args.all_dev is True
    assert args.steps == 20


def test_qwen_inference_requires_a_gpu_lease(monkeypatch, tmp_path):
    from src.qwen import inference

    monkeypatch.delenv("IMAGE_EDIT_GPU_UUID", raising=False)
    with pytest.raises(RuntimeError, match="image-edit gpu"):
        inference.run(type("Args", (), {"run": str(tmp_path)})())
