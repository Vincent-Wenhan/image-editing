"""Wait for verified assets, then run the bounded Stage 0 engineering acceptance."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.core.config import ProjectSettings
from src.core.io import write_json
from src.core.metadata import now
from src.core.paths import PROJECT_ROOT
from src.evaluation.provenance import verify_run


def main():
    settings = ProjectSettings.load()
    budget = settings.experiment["acceptance"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=settings.dataset["root"])
    parser.add_argument("--wait-hours", type=float, default=budget["wait_hours"])
    parser.add_argument("--output", default="outputs/stage0")
    parser.add_argument(
        "--base-run", type=Path, help="Reuse a verified complete fixed-sample Base run"
    )
    args = parser.parse_args()
    root = PROJECT_ROOT
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        raise FileExistsError("Use a new output directory for a new acceptance run")
    state = dict(created=now(), status="waiting_for_assets", completed=[], data_root=args.data_root)
    started = time.perf_counter()
    write_json(status_path, state)
    python = sys.executable
    cli = [python, "-m", "src"]
    deadline = time.monotonic() + args.wait_hours * 3600
    lock = json.loads((root / "configs/assets.lock.json").read_text())
    assets = lock["models"]
    markers = [root / item["path"] / "download.json" for item in assets.values()]
    dataset = root / args.data_root / "dataset.json"
    try:
        while True:
            available = all(path.is_file() for path in markers) and dataset.is_file()
            if available:
                for asset, marker in zip(assets.values(), markers, strict=True):
                    downloaded = json.loads(marker.read_text())
                    for key in ("repo_id", "revision", "derived_from"):
                        if key in asset and downloaded.get(key) != asset[key]:
                            raise ValueError(f"Asset provenance mismatch: {marker}")
                data_state = json.loads(dataset.read_text())
                if data_state.get("revision") != lock["dataset"]["revision"]:
                    raise ValueError("Dataset revision differs from the asset lock")
                if data_state.get("status") == "complete":
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Assets remain incomplete; inspect download logs before retrying"
                )
            time.sleep(budget["poll_seconds"])

        def run(name, command, gpu=False):
            state.update(status="running", current=name, updated=now())
            write_json(status_path, state)
            if gpu:
                command = cli + ["gpu", "--"] + command
            step_started = time.perf_counter()
            try:
                with (output / f"{name}.log").open("w") as log:
                    subprocess.run(
                        command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
            finally:
                state.setdefault("timings", {})[name] = time.perf_counter() - step_started
                write_json(status_path, state)
            state["completed"].append(name)
            write_json(status_path, state)

        manifest = str(root / args.data_root / "dev/manifest.csv")
        data_args = ["--data-root", str(root / args.data_root), "--manifest", manifest]
        run("data_validation", cli + ["validate", "--root", str(root / args.data_root)])
        run(
            "environment",
            cli + ["env", "--cuda", "--output", str(output / "environment.json")],
            True,
        )
        base = output / "base"
        if args.base_run:
            original = args.base_run.resolve()
            record = json.loads((original / "run.json").read_text())
            verify_run(root / args.data_root, manifest, record)
            if record.get("lora_sha256") or record["config"].get("lora"):
                raise ValueError("A LoRA run cannot serve as the Base reference")
            for key in ("repo_id", "revision"):
                if record["model"].get(key) != assets["base"][key]:
                    raise ValueError(f"Reused Base model differs from the lock: {key}")
            expected = settings.process["sample"]
            for key, value in dict(
                side=expected["width"],
                steps=expected["sample_steps"],
                guidance=expected["guidance_scale"],
                seed=expected["seed"],
                limit=settings.experiment["inference_limit"],
            ).items():
                if record["config"][key] != value:
                    raise ValueError(f"Reused Base configuration differs: {key}")
            base.mkdir()
            shutil.copy2(original / "run.json", base / "run.json")
            for item in record["results"]:
                shutil.copy2(original / f"{item['id']}.png", base / f"{item['id']}.png")
            state["base_reused_from"] = str(original)
            state["completed"].append("base_inference")
            write_json(status_path, state)
        else:
            run("base_inference", cli + ["infer", *data_args, "--output", str(base)], True)
        run("base_metrics", cli + ["evaluate", *data_args, "--predictions", str(base)])
        train_root = output / "lora"

        def train(name, steps, batch_size, reference=None):
            config = output / f"{name}.yaml"
            report = output / f"{name}_report.json"
            run(
                f"{name}_config",
                cli
                + [
                    "train-config",
                    "--data-root",
                    str(root / args.data_root),
                    "--steps",
                    str(steps),
                    "--batch-size",
                    str(batch_size),
                    "--name",
                    name,
                    "--output-root",
                    str(train_root),
                    "--output",
                    str(config),
                ],
            )
            training = [python, "-m", "src.training", str(config), "--report", str(report)]
            if reference:
                training += ["--memory-reference", str(reference)]
            state.setdefault("training_attempts", []).append(dict(name=name, report=str(report)))
            run(f"{name}_training", training, True)
            return report

        debug = train("debug", budget["debug_steps"], budget["debug_batch_size"])
        smoke = output / "smoke_report.json"
        try:
            train("smoke", settings.process["train"]["steps"], budget["smoke_batch_size"], debug)
        except subprocess.CalledProcessError:
            failed = json.loads(smoke.read_text()) if smoke.is_file() else {}
            if (
                not failed.get("out_of_memory")
                or failed.get("batch_size", 0) <= budget["debug_batch_size"]
            ):
                raise
            smoke = train(
                "smoke_retry", settings.process["train"]["steps"], budget["debug_batch_size"], debug
            )
        state["training_reports"] = dict(debug=str(debug), smoke=str(smoke))
        checkpoint = Path(json.loads(smoke.read_text())["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Expected final LoRA checkpoint: {checkpoint}")
        lora = output / "lora_inference"
        run(
            "lora_reload",
            cli
            + [
                "infer",
                *data_args,
                "--lora",
                str(checkpoint),
                "--output",
                str(lora),
            ],
            True,
        )
        run(
            "lora_metrics",
            cli
            + ["evaluate", *data_args, "--predictions", str(lora), "--reference-run", str(base)],
        )
        for name, predictions in (("base", base), ("lora", lora)):
            run(
                f"{name}_gallery",
                cli
                + [
                    "gallery",
                    *data_args,
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output / f"{name}_gallery"),
                ],
            )
        state.update(
            status="complete",
            finished=now(),
            conclusion="Engineering acceptance only; no performance improvement claim",
        )
    except BaseException as error:
        state.update(status="failed", error=repr(error), updated=now())
        raise
    finally:
        state["seconds"] = time.perf_counter() - started
        write_json(status_path, state)


if __name__ == "__main__":
    main()
