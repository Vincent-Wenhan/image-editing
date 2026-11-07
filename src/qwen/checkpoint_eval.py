"""Generate and score every requested checkpoint with one resident Qwen pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.core.io import sha256, write_json
from src.infra.gpu import require_lease
from src.qwen.evaluation import run as score
from src.qwen.inference import _pipeline_class
from src.qwen.inference import run as infer


def run(args) -> None:
    require_lease()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    run_root = Path(args.run).resolve()
    record = json.loads((run_root / "run.json").read_text())
    checkpoints = [int(step) for step in args.checkpoints.split(",")]
    if not checkpoints or len(set(checkpoints)) != len(checkpoints) or min(checkpoints) < 0:
        raise ValueError("Supply unique nonnegative checkpoint numbers")
    paths = [
        run_root / f"checkpoints/checkpoint-{step}/pytorch_lora_weights.safetensors"
        for step in checkpoints
    ]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Requested Qwen checkpoint is missing")
    output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "run": str(run_root),
        "split": args.split,
        "steps": args.steps,
        "seed": args.seed,
        "results": [],
    }
    write_json(output / "evaluation.json", receipt)
    try:
        import torch

        pipeline = _pipeline_class().from_pretrained(
            record["model_path"], torch_dtype=torch.bfloat16, local_files_only=True
        )
        pipeline.enable_model_cpu_offload()
        for step, checkpoint in zip(checkpoints, paths, strict=True):
            directory = output / f"checkpoint-{step}"
            infer_args = SimpleNamespace(
                run=str(run_root),
                output=str(directory / "predictions"),
                checkpoint=str(checkpoint),
                split=args.split,
                all_split=True,
                all_dev=False,
                sample_id=None,
                source_only=False,
                seed=args.seed,
                seed_policy="per_sample",
                steps=args.steps,
                lock=args.lock,
            )
            infer(infer_args, pipeline=pipeline)
            score(
                SimpleNamespace(
                    run=str(run_root),
                    predictions=infer_args.output,
                    output=str(directory / "score"),
                    split=args.split,
                    lock=args.lock,
                )
            )
            receipt["results"].append(
                {
                    "checkpoint": step,
                    "checkpoint_sha256": sha256(checkpoint),
                    "metrics": str(directory / "score/metrics.json"),
                }
            )
            write_json(output / "evaluation.json", receipt)
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(output / "evaluation.json", receipt)


def configure(parser) -> None:
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", default="20,40,60,80,100,120")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", default="configs/assets.lock.json")
    parser.set_defaults(func=run)
