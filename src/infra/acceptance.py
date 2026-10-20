"""Wait for verified assets, then run the bounded Stage 0 engineering acceptance."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.core.config import ProjectSettings
from src.core.io import write_json
from src.core.metadata import now
from src.core.paths import PROJECT_ROOT


def main():
    settings = ProjectSettings.load()
    budget = settings.experiment["acceptance"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=settings.dataset["root"])
    parser.add_argument("--wait-hours", type=float, default=budget["wait_hours"])
    parser.add_argument("--output", default="outputs/stage0")
    args = parser.parse_args()
    root = PROJECT_ROOT
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.exists():
        raise FileExistsError("Use a new output directory for a new acceptance run")
    state = dict(created=now(), status="waiting_for_assets", completed=[], data_root=args.data_root)
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
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
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
        run(
            "base_inference",
            cli + ["infer", *data_args, "--output", str(base)],
            True,
        )
        run("base_metrics", cli + ["evaluate", *data_args, "--predictions", str(base)])
        train_root = output / "lora"
        batch_size = budget["debug_batch_size"]
        for name, steps in (
            ("debug", budget["debug_steps"]),
            ("smoke", settings.process["train"]["steps"]),
        ):
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
            run(
                f"{name}_training",
                [python, "-m", "src.training", str(config), "--report", str(report)],
                True,
            )
            if name == "debug":
                measured = json.loads(report.read_text())
                batch_size = (
                    budget["smoke_batch_size"]
                    if measured["peak_reserved_gib"] < budget["scale_up_reserved_gib"]
                    else budget["debug_batch_size"]
                )
        checkpoint = train_root / "smoke/smoke.safetensors"
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
        run("lora_metrics", cli + ["evaluate", *data_args, "--predictions", str(lora)])
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
        write_json(status_path, state)


if __name__ == "__main__":
    main()
