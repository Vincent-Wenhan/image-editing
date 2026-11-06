"""Prepare and launch the pinned Inter-Edit Qwen CJT pilot."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.core.config import load_assets, load_yaml
from src.core.io import sha256, write_json
from src.core.metadata import now, versions
from src.infra.experiment import source_snapshot
from src.infra.gpu import require_lease
from src.qwen.data import verify_manifest, write_manifest
from src.qwen.evaluation import run as score
from src.qwen.inference import run as infer
from src.qwen.provenance import verify_external_repo, verify_model_download

PROTOCOL = "configs/experiments/qwen_cjt_pilot.yaml"


def _paths(run: Path) -> dict[str, Path]:
    return {
        "train": run / "train.json",
        "dev": run / "dev.json",
        "config": run / "train_config.yaml",
        "record": run / "run.json",
    }


def prepare(args) -> None:
    run = Path(args.output).resolve()
    if run.exists():
        raise FileExistsError(f"Qwen run already exists: {run}")
    protocol = load_yaml(args.protocol)
    if protocol.get("schema_version") != "qwen_cjt_pilot_v1":
        raise ValueError("Unsupported Qwen pilot protocol")
    data_root = Path(args.data_root).resolve()
    run.mkdir(parents=True)
    try:
        paths = _paths(run)
        control_root = Path(args.control_root).resolve() if args.control_root else None
        train_record = write_manifest(
            data_root,
            protocol["train_split"],
            paths["train"],
            limit=protocol.get("train_limit"),
            lock_path=args.lock,
            control_root=control_root,
        )
        dev_record = write_manifest(
            data_root,
            protocol["dev_split"],
            paths["dev"],
            limit=protocol.get("dev_limit"),
            lock_path=args.lock,
            control_root=control_root,
        )
        if train_record["condition"] != dev_record["condition"]:
            raise ValueError("Qwen train/dev manifests use different control snapshots")
        expected_condition = protocol.get("condition_kind")
        if expected_condition and train_record["condition"]["kind"] != expected_condition:
            raise ValueError(
                f"Qwen protocol expects {expected_condition}, "
                f"got {train_record['condition']['kind']}"
            )
        asset = load_assets(args.lock)["models"][protocol["model_asset"]]
        model_path = Path(asset["path"]).resolve()
        external = Path(args.external).resolve()
        external_record = verify_external_repo(external, args.lock)
        config = _external_config(protocol, paths, data_root, model_path, run)
        paths["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        record = {
            "schema_version": "qwen_cjt_run_v1",
            "status": "prepared",
            "created": now(),
            "protocol": str(Path(args.protocol).resolve()),
            "protocol_sha256": sha256(args.protocol),
            "data_root": str(data_root),
            "external": external_record,
            "model": asset,
            "model_path": str(model_path),
            "seed": protocol["seed"],
            "train_manifest": str(paths["train"]),
            "dev_manifest": str(paths["dev"]),
            "condition": train_record["condition"],
            "train_config": str(paths["config"]),
            "config_sha256": sha256(paths["config"]),
        }
        write_json(paths["record"], record)
        print(record)
    except BaseException:
        shutil.rmtree(run)
        raise


def _external_config(protocol, paths, data_root, model_path, run) -> dict:
    return {
        "pretrained_model_name_or_path": str(model_path),
        "data_config": {
            "json_file": str(paths["train"]),
            "data_root": str(data_root),
            "train_batch_size": protocol["train_batch_size"],
            "num_workers": protocol["num_workers"],
            "img_size": 1024,
            "caption_dropout_rate": 0.0,
            "filter_invalid": False,
            "only_better_data": False,
            "max_samples": None,
        },
        "report_to": None,
        "train_batch_size": protocol["train_batch_size"],
        "output_dir": str(run / "checkpoints"),
        "max_train_steps": protocol["max_train_steps"],
        "seed": protocol["seed"],
        "learning_rate": protocol["learning_rate"],
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "logging_dir": "logs",
        "mixed_precision": protocol["mixed_precision"],
        "checkpointing_steps": protocol["checkpointing_steps"],
        "checkpoints_total_limit": 2,
        "tracker_project_name": "image_editing_qwen_cjt_pilot",
        "resume_from_checkpoint": None,
        "gradient_accumulation_steps": protocol["gradient_accumulation_steps"],
        "cached_embeddings_dir": None,
        "rank": protocol["rank"],
        "precompute_text_embeddings": protocol["precompute_text_embeddings"],
        "precompute_image_embeddings": protocol["precompute_image_embeddings"],
        "quantize": protocol["quantize"],
        "adam8bit": protocol["adam8bit"],
        "max_control_images": protocol["max_control_images"],
    }


def verify(args, *, require_model: bool = False) -> dict:
    run = Path(args.run).resolve()
    record_path = run / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "qwen_cjt_run_v1":
        raise ValueError("Unsupported Qwen run record")
    if sha256(record["protocol"]) != record["protocol_sha256"]:
        raise ValueError("Qwen protocol changed")
    verify_external_repo(record["external"]["path"], args.lock)
    verify_manifest(record["train_manifest"], root=record["data_root"], lock_path=args.lock)
    verify_manifest(record["dev_manifest"], root=record["data_root"], lock_path=args.lock)
    config = Path(record["train_config"])
    if sha256(config) != record["config_sha256"]:
        raise ValueError("Qwen training config changed")
    if require_model or getattr(args, "require_model", False):
        verify_model_download(record["model_path"], args.lock)
    print(record)
    return record


def _accelerate_command(external: Path, config: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(external / "acc_config.yaml"),
        "--module",
        "src.qwen.seeded_train",
        "--seed",
        str(seed),
        "--config",
        config,
    ]


def train(args) -> None:
    require_lease()
    record = verify(args, require_model=True)
    run = Path(args.run).resolve()
    output = run / "execution.json"
    if output.exists():
        raise FileExistsError(f"Qwen run already has an execution receipt: {output}")
    external = Path(record["external"]["path"])
    command = _accelerate_command(external, record["train_config"], record["seed"])
    execution = {
        "status": "running",
        "created": now(),
        "argv": command,
        "cwd": str(external),
        "parent_cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": os.environ.get("IMAGE_EDIT_GPU_UUID"),
        "external": record["external"],
        "versions": versions(),
        "source": source_snapshot(),
    }
    write_json(output, execution)
    started = time.perf_counter()
    try:
        with (run / "train.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=external, stdout=log, stderr=subprocess.STDOUT)
        execution.update(
            returncode=result.returncode,
            status="complete" if result.returncode == 0 else "failed",
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    except BaseException as error:
        execution.update(status="failed", error=repr(error))
        raise
    finally:
        execution["seconds"] = time.perf_counter() - started
        write_json(output, execution)
    print(execution)


def configure(parser):
    subparsers = parser.add_subparsers(dest="qwen_command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data-root", default="data/magicbrush")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--protocol", default=PROTOCOL)
    prepare_parser.add_argument("--external", default="external/Inter-Edit")
    prepare_parser.add_argument("--control-root")
    prepare_parser.add_argument("--lock", default="configs/assets.lock.json")
    prepare_parser.set_defaults(func=prepare)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run", required=True)
    verify_parser.add_argument("--lock", default="configs/assets.lock.json")
    verify_parser.add_argument("--require-model", action="store_true")
    verify_parser.set_defaults(func=verify)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--run", required=True)
    train_parser.add_argument("--lock", default="configs/assets.lock.json")
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--run", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument(
        "--checkpoint",
        help="LoRA safetensors path; defaults to the latest checkpoint-N in the run",
    )
    infer_parser.add_argument("--sample-id")
    infer_parser.add_argument("--limit", type=int, default=1)
    infer_parser.add_argument("--all-dev", action="store_true")
    infer_parser.add_argument("--steps", type=int, default=4)
    infer_parser.add_argument("--seed", type=int, default=42)
    infer_parser.add_argument("--lock", default="configs/assets.lock.json")
    infer_parser.set_defaults(func=infer)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run", required=True)
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--lock", default="configs/assets.lock.json")
    score_parser.set_defaults(func=score)
