"""Check staged repository hygiene or validate a Conventional Commit subject."""

import re
import subprocess
import sys
from pathlib import Path

SUBJECT = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\([a-zA-Z0-9_./-]+\))?!?: \S.*$"
)
EXCLUDED = {"data", "models", "outputs", "external", ".venv", ".cache", ".tmp"}
ASSET_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".zip",
    ".parquet",
    ".arrow",
    ".npy",
    ".npz",
    ".mp4",
}
CODE_SUFFIXES = {".py", ".sh", ".js", ".ts", ".tsx", ".jsx"}


def main():
    if len(sys.argv) > 1:
        subject = Path(sys.argv[1]).read_text().splitlines()[0]
        if not SUBJECT.fullmatch(subject):
            raise SystemExit("Use Conventional Commits: type(scope): description")
        return
    paths = (
        subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        )
        .decode()
        .split("\0")
    )
    errors = []
    for name in filter(None, paths):
        path = Path(name)
        if (
            path.parts[0] in EXCLUDED
            or path.suffix.lower() in ASSET_SUFFIXES
            or path.name.startswith(".env")
            and path.name != ".env.example"
        ):
            errors.append(f"Forbidden local asset: {name}")
            continue
        data = subprocess.check_output(["git", "show", f":{name}"])
        if len(data) > 1_000_000:
            errors.append(f"File exceeds 1 MB: {name}")
        if path.suffix in CODE_SUFFIXES and len(data.splitlines()) > 500:
            errors.append(f"Code exceeds 500 lines: {name}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
