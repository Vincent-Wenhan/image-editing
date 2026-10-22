"""Optional pinned Linux shared libraries extracted locally, without system installs."""

import argparse
import json
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path

from src.core.io import sha256, write_json
from src.core.paths import CONFIG_ROOT


def prepare(output: Path, lock_path: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("This optional runtime lock is for Linux x86_64 only")
    lock = json.loads(lock_path.read_text())
    if lock["platform"] != "linux-x86_64":
        raise ValueError("Unexpected runtime platform")
    library = output / "usr/lib/x86_64-linux-gnu"
    if output.exists():
        state = json.loads((output / "runtime.json").read_text())
        if state["lock_sha256"] != sha256(lock_path) or any(
            sha256(output / relative) != digest for relative, digest in state["files"].items()
        ):
            raise ValueError("Local Blender runtime SHA256 changed")
        return library
    packages = output.parent / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for item in lock["packages"]:
        path = packages / f"{item['name']}.deb"
        if not path.exists() or sha256(path) != item["sha256"]:
            with (
                urllib.request.urlopen(item["url"], timeout=30) as source,
                path.open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Runtime package checksum mismatch: {item['name']}")
        subprocess.run(["dpkg-deb", "--extract", str(path), str(output)], check=True)
    write_json(
        output / "runtime.json",
        dict(
            lock_sha256=sha256(lock_path),
            files={
                p.relative_to(output).as_posix(): sha256(p)
                for p in library.iterdir()
                if p.is_file()
            },
        ),
    )
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/blender/runtime"))
    parser.add_argument(
        "--lock", type=Path, default=CONFIG_ROOT / "blender-runtime-linux.lock.json"
    )
    args = parser.parse_args()
    print(prepare(args.output.resolve(), args.lock.resolve()))


if __name__ == "__main__":
    main()
