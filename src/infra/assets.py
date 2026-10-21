"""Download only locked assets; keep all weights outside Git."""

from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

from src.core.config import load_assets
from src.core.io import write_json
from src.core.metadata import now


def run(args):
    from huggingface_hub import snapshot_download

    lock = load_assets(args.lock)
    selected = [args.asset] if args.asset != "all" else list(lock["models"])
    for name in selected:
        asset = lock["models"][name]
        destination = Path(asset["path"])
        if "derived_from" in asset:
            base = Path(lock["models"][asset["derived_from"]]["path"])
            if not (base / "download.json").exists():
                raise FileNotFoundError("Download the complete base model first")
            destination.mkdir(parents=True, exist_ok=True)
            for folder in ("text_encoder", "tokenizer"):
                for source in (base / folder).iterdir():
                    if source.is_file():
                        target = destination / source.name
                        if not target.exists():
                            target.symlink_to(source.resolve())
        elif args.transport == "ranges":
            from src.infra.transport import download_file, repo_info

            info = repo_info(asset["repo_id"], asset["revision"])
            items = [
                item
                for item in info["siblings"]
                if any(fnmatch(item["rfilename"], p) for p in asset["allow_patterns"])
            ]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(
                        download_file,
                        asset["repo_id"],
                        asset["revision"],
                        item,
                        destination,
                        payload_mirror=args.payload_mirror if name == "base" else None,
                    )
                    for item in items
                ]
                for future in futures:
                    future.result()
        else:
            snapshot_download(
                repo_id=asset["repo_id"],
                revision=asset["revision"],
                local_dir=destination,
                allow_patterns=asset["allow_patterns"],
                max_workers=args.workers,
            )
        write_json(
            destination / "download.json",
            dict(
                asset, completed=now(), transport=args.transport, payload_mirror=args.payload_mirror
            ),
        )
        print(f"Downloaded {name}: {destination}", flush=True)


def configure(parser):
    parser.add_argument("--lock", default="configs/assets.lock.json")
    parser.add_argument(
        "--asset", choices=["base", "training_vae", "training_text", "all"], default="base"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--transport", choices=["hub", "ranges"], default="hub")
    parser.add_argument("--payload-mirror", choices=["modelscope"])
    parser.set_defaults(func=run)
