"""Fixed-sample FLUX.2 Klein Base editing with incremental provenance records."""

import json
import time
from pathlib import Path

from PIL import Image

from image_editing.common import now, read_rows, sample_seed, sha256, versions, write_json
from image_editing.gpu import require_lease


def dimensions(size, side):
    scale = side / max(size)
    return tuple(max(16, round(value * scale / 16) * 16) for value in size)


def run(args):
    require_lease()
    import torch
    from diffusers import Flux2KleinPipeline

    if args.limit < 1 or args.steps < 1 or args.side < 16:
        raise ValueError("Positive limit/steps and side >=16 required")
    rows = read_rows(args.manifest)[: args.limit]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new run directory: {output}")
    output.mkdir(parents=True)
    model = Path(args.model)
    provenance = json.loads((model / "download.json").read_text())
    run_info = dict(
        created=now(),
        status="running",
        config=vars(args).copy(),
        model=provenance,
        packages=versions(),
        manifest_sha256=sha256(args.manifest),
        results=[],
    )
    run_info["config"].pop("func")
    if args.lora:
        run_info["lora_sha256"] = sha256(args.lora)
    write_json(output / "run.json", run_info)
    try:
        torch.cuda.reset_peak_memory_stats()
        pipe = Flux2KleinPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        if args.lora:
            pipe.load_lora_weights(args.lora)
        for row in rows:
            with Image.open(Path(args.data_root) / row["source_path"]) as original:
                source = original.convert("RGB")
            width, height = dimensions(source.size, args.side)
            source = source.resize((width, height), Image.Resampling.LANCZOS)
            seed = sample_seed(args.seed, row["id"])
            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = pipe(
                    image=source,
                    prompt=row["instruction"],
                    width=width,
                    height=height,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                ).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            prediction.save(output / f"{row['id']}.png")
            run_info["results"].append(
                dict(
                    id=row["id"],
                    seed=seed,
                    width=width,
                    height=height,
                    seconds=elapsed,
                    source_sha256=row["source_sha256"],
                    prompt=row["instruction"],
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                )
            )
            write_json(output / "run.json", run_info)
            print(f"{row['id']}: {elapsed:.1f}s", flush=True)
        run_info["status"] = "complete"
    except BaseException as error:
        run_info["status"] = "failed"
        run_info["error"] = repr(error)
        raise
    finally:
        write_json(output / "run.json", run_info)


def configure(parser):
    parser.add_argument("--model", default="models/FLUX.2-klein-base-4B")
    parser.add_argument("--manifest", default="data/magicbrush/dev/manifest.csv")
    parser.add_argument("--data-root", default="data/magicbrush")
    parser.add_argument("--output", default="outputs/base_inference")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora")
    parser.set_defaults(func=run)
