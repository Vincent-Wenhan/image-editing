"""Leased, target-free Qwen CJT checkpoint reload and one-sample inference."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from PIL import Image

from src.core.io import sha256, write_json
from src.infra.gpu import require_lease
from src.qwen.conditions import condition_path
from src.qwen.data import verify_condition_manifest
from src.qwen.provenance import verify_model_download


def _pipeline_class():
    path = Path("external/Inter-Edit/pipeline_qwenimage_edit_plus.py").resolve()
    spec = importlib.util.spec_from_file_location("qwen_inter_edit_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned Qwen pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QwenImageEditPlusPipeline


def _checkpoint_path(run_root: Path, requested: str | None = None) -> Path:
    if requested is not None:
        checkpoint = Path(requested).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing requested Qwen LoRA checkpoint: {checkpoint}")
        return checkpoint
    candidates = []
    for path in (run_root / "checkpoints").glob("checkpoint-*/pytorch_lora_weights.safetensors"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.parent.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No Qwen LoRA checkpoint found under {run_root / 'checkpoints'}")
    return max(candidates, key=lambda item: item[0])[1]


def _match_source_size(
    image: Image.Image, source_size: tuple[int, int]
) -> tuple[Image.Image, bool]:
    if image.size == source_size:
        return image.convert("RGB"), False
    return image.convert("RGB").resize(source_size, Image.Resampling.LANCZOS), True


def _rows(
    record: dict, sample_id: str | None, limit: int | None, split: str = "dev"
) -> list[tuple[dict[str, str], dict[str, str]]]:
    if split not in {"train", "dev"}:
        raise ValueError("Qwen inference split must be train or dev")
    root = Path(record["data_root"])
    manifest = Path(record[f"{split}_manifest"])
    provenance = verify_condition_manifest(manifest, root=root)
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    if sample_id is not None:
        matches = [row for row in samples if row["id"] == sample_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown or duplicate {split} sample: {sample_id}")
        selected = matches
    elif limit is None:
        selected = samples
    else:
        if limit < 1 or limit > len(samples):
            raise ValueError(f"Dev inference limit must be in [1, {len(samples)}]")
        selected = samples[:limit]
    checks_by_id = {item["id"]: item for item in provenance["samples"]}
    result = []
    for sample in selected:
        checks = checks_by_id.get(sample["id"])
        if checks is None:
            raise ValueError(f"Missing provenance for Dev sample: {sample['id']}")
        result.append((sample, checks))
    return result


def _row(
    record: dict, sample_id: str | None, split: str = "dev"
) -> tuple[dict[str, str], dict[str, str]]:
    return _rows(record, sample_id, 1, split)[0]


def run(args) -> None:
    require_lease()
    run_root = Path(args.run).resolve()
    record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if record.get("status") != "prepared":
        raise ValueError("Qwen inference requires the prepared run record")
    execution = json.loads((run_root / "execution.json").read_text(encoding="utf-8"))
    if execution.get("status") != "complete" or execution.get("returncode") != 0:
        raise ValueError("Qwen inference requires a completed training execution")
    verify_model_download(record["model_path"], args.lock)
    sample_id = getattr(args, "sample_id", None)
    all_dev = getattr(args, "all_dev", False)
    split = getattr(args, "split", "dev")
    all_split = getattr(args, "all_split", False) or all_dev
    if sample_id is not None and all_split:
        raise ValueError("--sample-id and --all-split cannot be combined")
    limit = None if all_split else getattr(args, "limit", 1)
    rows = _rows(record, sample_id, limit, split)
    root = Path(record["data_root"])
    source_only = getattr(args, "source_only", False)
    checkpoint = (
        None if source_only else _checkpoint_path(run_root, getattr(args, "checkpoint", None))
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new Qwen inference directory: {output}")
    output.mkdir(parents=True)
    result_record = {
        "status": "running",
        "run": str(run_root),
        "control_order": ["source"] if source_only else ["source", "mask"],
        "baseline": "base_source_only" if source_only else "trained_control",
        "split": split,
        "seed": args.seed,
        "steps": args.steps,
        "cpu_offload": True,
        "sample_count": len(rows),
    }
    write_json(output / "inference.json", result_record)
    try:
        import torch

        pipeline_class = _pipeline_class()
        pipeline = pipeline_class.from_pretrained(record["model_path"], torch_dtype=torch.bfloat16)
        if checkpoint is not None:
            pipeline.load_lora_weights(str(checkpoint))
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=False)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        results = []
        for row, checks in rows:
            source_path = condition_path(root, row["original_image_path"])
            mask_path = condition_path(root, row["mask_image_path"])
            with Image.open(source_path) as image:
                source = image.convert("RGB")
            with Image.open(mask_path) as image:
                mask = image.convert("RGB")
            with torch.inference_mode():
                generated = pipeline(
                    image=[source] if source_only else [source, mask],
                    prompt=row["instruction"],
                    negative_prompt=" ",
                    true_cfg_scale=4.0,
                    guidance_scale=1.0,
                    num_inference_steps=args.steps,
                    generator=generator,
                    num_images_per_prompt=1,
                ).images[0]
            native_size = generated.size
            prediction, resized = _match_source_size(generated, source.size)
            prediction_path = output / f"{row['id']}.png"
            prediction.save(prediction_path)
            results.append(
                {
                    "sample_id": row["id"],
                    "source_sha256": checks["source_sha256"],
                    "target_sha256": checks["target_sha256"],
                    "native_size": list(native_size),
                    "output_size": list(prediction.size),
                    "resized_to_source": resized,
                    "prediction_sha256": sha256(prediction_path),
                }
            )
            if not source_only:
                results[-1]["mask_sha256"] = checks["mask_sha256"]
        result_record.update(status="complete", results=results)
        if len(results) == 1:
            result_record.update(results[0])
        if checkpoint is not None:
            result_record["checkpoint"] = str(checkpoint)
            result_record["checkpoint_sha256"] = sha256(checkpoint)
    except BaseException as error:
        result_record.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(output / "inference.json", result_record)
    print(result_record)


def configure(parser) -> None:
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", default="configs/assets.lock.json")
    parser.set_defaults(func=run)
