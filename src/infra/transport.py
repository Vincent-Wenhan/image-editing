"""Pinned Hub downloads with curl metadata, parallel ranges and SHA256 checks."""

import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.core.io import sha256, write_json


def file_matches(path, item):
    if not path.is_file():
        return False
    digest = item.get("lfs", {}).get("sha256")
    if digest:
        return sha256(path) == digest
    if not item.get("blobId"):
        raise ValueError(f"No content digest for {item['rfilename']}")
    hasher = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest() == item["blobId"]


def download_ranges(url, partial, size, connections):
    """Resume verified-size chunks; validate the complete payload against the Hub SHA."""
    chunks = partial.with_name(partial.name + ".chunks")
    chunks.mkdir(exist_ok=True)
    chunk_size = 4 * 1024**2

    def fetch(start):
        end = min(start + chunk_size, size) - 1
        target = chunks / str(start)
        expected = end - start + 1
        if target.exists() and target.stat().st_size == expected:
            return target
        temporary = target.with_suffix(".tmp")
        if temporary.exists() and temporary.stat().st_size > expected:
            temporary.unlink()
        for _ in range(8):
            received = temporary.stat().st_size if temporary.exists() else 0
            if received == expected:
                temporary.replace(target)
                return target
            fragment = chunks / f"{start}.{received}.part"
            if not fragment.exists():
                subprocess.run(
                    [
                        "curl",
                        "-4",
                        "-fLsS",
                        "--connect-timeout",
                        "15",
                        "--max-time",
                        "600",
                        "--range",
                        f"{start + received}-{end}",
                        "--max-filesize",
                        str(expected - received),
                        "-o",
                        str(fragment),
                        url,
                    ],
                    check=False,
                )
            # curl can leave valid prefix bytes after a reset; retain them.
            if fragment.exists():
                if fragment.stat().st_size > expected - received:
                    fragment.unlink()
                    raise ValueError(f"Server returned too many bytes for range {start}-{end}")
                with temporary.open("ab") as output, fragment.open("rb") as source:
                    shutil.copyfileobj(source, output)
                fragment.unlink()
        if temporary.exists() and temporary.stat().st_size == expected:
            temporary.replace(target)
            return target
        raise RuntimeError(f"Range {start}-{end} incomplete; saved bytes are resumable")

    offsets = range(0, size, chunk_size)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        files = list(pool.map(fetch, offsets))
    with partial.open("wb") as output:
        for file in files:
            with file.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024**2)
    return chunks


def endpoint():
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def curl(url, *, headers=False):
    args = ["curl", "-fLsS", "--retry", "3", "--connect-timeout", "15", "--max-time", "120"]
    if headers:
        args = [arg for arg in args if arg != "-fLsS"] + ["-fIsS"]
    return subprocess.check_output([*args, url], text=True)


def repo_info(repo, revision, repo_type="model"):
    cache = Path(".cache/asset-metadata") / f"{repo_type}-{repo.replace('/', '--')}-{revision}.json"
    if cache.exists():
        info = json.loads(cache.read_text())
        if info.get("sha") != revision or info.get("id") != repo:
            raise ValueError(f"Cached asset metadata does not match the lock: {cache}")
        return info
    info = json.loads(curl(f"{endpoint()}/api/{repo_type}s/{repo}/revision/{revision}?blobs=true"))
    if info["sha"] != revision:
        raise ValueError("Hub returned a different revision")
    write_json(cache, info)
    return info


def download_file(
    repo, revision, item, destination, repo_type="model", connections=8, payload_mirror=None
):
    name = item["rfilename"]
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Unsafe remote filename: {name}")
    target = Path(destination) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = item.get("lfs", {}).get("sha256")
    if file_matches(target, item):
        return target
    prefix = "datasets/" if repo_type == "dataset" else ""
    url = f"{endpoint()}/{prefix}{repo}/resolve/{revision}/{name}"
    partial = target.with_name(target.name + ".partial")
    chunks = None
    if digest:
        if payload_mirror == "modelscope":
            url = f"https://modelscope.cn/models/{repo}/resolve/master/{name}"
        # Resolve each range through the stable URL so expiring CDN signatures
        # are refreshed automatically by curl, including after retries.
        chunks = download_ranges(url, partial, item["size"], connections)
    else:
        subprocess.run(
            [
                "curl",
                "-fLsS",
                "--retry",
                "3",
                "--connect-timeout",
                "15",
                "--max-time",
                "180",
                "-o",
                str(partial),
                url,
            ],
            check=True,
        )
    if not file_matches(partial, item):
        if chunks:
            shutil.rmtree(chunks)
        partial.unlink(missing_ok=True)
        raise ValueError(f"Content checksum mismatch: {name}")
    partial.replace(target)
    if chunks:
        shutil.rmtree(chunks)
    print(f"Verified download: {target}", flush=True)
    return target
