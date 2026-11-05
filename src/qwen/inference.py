"""Leased, target-free Qwen CJT checkpoint reload and one-sample inference."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from PIL import Image

from src.core.io import read_rows, sha256, write_json
from src.core.paths import data_path
from src.infra.gpu import require_lease
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


def _row(record: dict, sample_id: str | None) -> dict[str, str]:
    root = Path(record["data_root"])
    verify_condition_manifest(record["dev_manifest"], root=root)
    rows = read_rows(root / "dev/manifest.csv")
    if sample_id is None:
        return rows[0]
    matches = [row for row in rows if row["id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate Dev sample: {sample_id}")
    return matches[0]


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
    row = _row(record, args.sample_id)
    root = Path(record["data_root"])
    source_path = data_path(root, row["source_path"])
    mask_path = data_path(root, row["mask_path"])
    checkpoint = _checkpoint_path(run_root, getattr(args, "checkpoint", None))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new Qwen inference directory: {output}")
    output.mkdir(parents=True)
    result_record = {
        "status": "running",
        "run": str(run_root),
        "sample_id": row["id"],
        "source_sha256": row["source_sha256"],
        "mask_sha256": row["mask_sha256"],
        "target_sha256": row["target_sha256"],
        "control_order": ["source", "mask"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": args.seed,
        "steps": args.steps,
        "cpu_offload": True,
    }
    write_json(output / "inference.json", result_record)
    try:
        import torch

        pipeline_class = _pipeline_class()
        pipeline = pipeline_class.from_pretrained(record["model_path"], torch_dtype=torch.bfloat16)
        pipeline.load_lora_weights(str(checkpoint))
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=False)
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(mask_path) as image:
            mask = image.convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.inference_mode():
            generated = pipeline(
                image=[source, mask],
                prompt=row["instruction"],
                negative_prompt=" ",
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                num_inference_steps=args.steps,
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]
        generated.convert("RGB").save(output / f"{row['id']}.png")
        result_record.update(
            status="complete", prediction_sha256=sha256(output / f"{row['id']}.png")
        )
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
