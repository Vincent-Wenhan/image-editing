"""Download only locked assets; keep all weights outside Git."""

import json
from pathlib import Path

from image_editing.common import now, write_json


def run(args):
    from huggingface_hub import snapshot_download

    lock = json.loads(Path(args.lock).read_text())
    selected = [args.asset] if args.asset != "all" else list(lock["models"])
    for name in selected:
        asset = lock["models"][name]
        destination = Path(asset["path"])
        snapshot_download(
            repo_id=asset["repo_id"],
            revision=asset["revision"],
            local_dir=destination,
            allow_patterns=asset["allow_patterns"],
            max_workers=args.workers,
        )
        write_json(destination / "download.json", dict(asset, completed=now()))
        print(f"Downloaded {name}: {destination}", flush=True)


def configure(parser):
    parser.add_argument("--lock", default="configs/assets.lock.json")
    parser.add_argument(
        "--asset", choices=["base", "training_vae", "training_text", "all"], default="base"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.set_defaults(func=run)
