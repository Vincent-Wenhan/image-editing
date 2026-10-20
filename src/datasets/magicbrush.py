"""Stream or download pinned MagicBrush shards and reuse only validated exports."""

import fcntl
import json
import shutil
from pathlib import Path

from src.core.config import ProjectSettings, load_assets
from src.core.io import write_json
from src.core.metadata import now
from src.datasets.export import export_rows


def streaming_rows(lock, split):
    from datasets import Image as DatasetImage
    from datasets import load_dataset

    rows = load_dataset(
        lock["repo_id"],
        revision=lock["revision"],
        split=split,
        data_files={split: f"data/{split}-*.parquet"},
        streaming=True,
    )
    for column in ("source_img", "target_img", "mask_img"):
        rows = rows.cast_column(column, DatasetImage(decode=False))
    yield from rows


def parquet_rows(info, split, cache):
    """Download only the successive shards needed to satisfy the subset count."""
    import pyarrow.parquet as parquet

    from src.infra.transport import download_file

    files = sorted(
        (item for item in info["siblings"] if item["rfilename"].startswith(f"data/{split}-")),
        key=lambda item: item["rfilename"],
    )
    for item in files:
        path = download_file(info["id"], info["sha"], item, cache, repo_type="dataset")
        with parquet.ParquetFile(path) as shard:
            for batch in shard.iter_batches(batch_size=16):
                yield from batch.to_pylist()


def prepare_locked(args):
    from concurrent.futures import ThreadPoolExecutor

    from src.datasets.validation import scan, validate
    from src.infra.transport import repo_info

    lock = load_assets(args.lock)["dataset"]
    root = Path(args.root)
    state = dict(
        lock,
        selection="first turn; fixed upstream order",
        requested={"train": args.train, "dev": args.dev},
        side=args.side,
        mask_policy="inverse_alpha_or_white_binary",
    )
    state_path = root / "dataset.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if any(previous[k] != state[k] for k in ("repo_id", "revision", "selection", "requested")):
            raise ValueError("Existing export provenance differs; use a new root")
    elif any(p.name != ".prepare.lock" for p in root.iterdir()):
        raise ValueError("Existing data has no provenance; use a new root")
    if args.side < 16 or min(args.train, args.dev) < 1:
        raise ValueError("Positive sample counts and side >=16 required")
    if state_path.exists() and any((root / x / "manifest.csv").exists() for x in ("train", "dev")):
        if any(previous.get(k) != state[k] for k in ("side", "mask_policy")):
            raise ValueError("Existing export uses different preprocessing")
    state.update(created=now(), status="running", transport=args.transport)
    write_json(state_path, state)
    info = (
        repo_info(lock["repo_id"], lock["revision"], repo_type="dataset")
        if args.transport == "shards"
        else None
    )

    def export(split, count):
        if (root / split / "manifest.csv").exists():
            rows, *_ = scan(root, split)
            if len(rows) != count:
                raise ValueError(f"Unexpected count for {split}")
            return
        cache = root.parent / "raw" / root.name
        rows = parquet_rows(info, split, cache) if info else streaming_rows(lock, split)
        export_rows(rows, root, split, count, args.side)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(export, split, count)
            for split, count in (("train", args.train), ("dev", args.dev))
        ]
        for future in futures:
            future.result()
    write_json(root / "validation.json", validate(root))
    write_json(state_path, state | {"status": "complete"})


def prepare(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another preparation job is using {root}") from error
        for split in ("train", "dev"):
            partial = root / f".{split}.incomplete"
            if partial.exists():
                shutil.rmtree(partial)
        prepare_locked(args)


def configure(parser):
    settings = ProjectSettings.load()
    parser.add_argument("--root", default=settings.dataset["root"])
    parser.add_argument("--lock", default="configs/assets.lock.json")
    parser.add_argument("--train", type=int, default=settings.dataset["train"])
    parser.add_argument("--dev", type=int, default=settings.dataset["dev"])
    parser.add_argument(
        "--side", type=int, default=settings.process["datasets"][0]["resolution"][0]
    )
    parser.add_argument("--transport", choices=["streaming", "shards"], default="streaming")
    parser.set_defaults(func=prepare)
