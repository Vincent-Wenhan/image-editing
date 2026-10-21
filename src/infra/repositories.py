"""Check or bootstrap locked external repositories without modifying existing clones."""

import json
import subprocess

from src.core.paths import PROJECT_ROOT


def main() -> None:
    lock = json.loads((PROJECT_ROOT / "configs/assets.lock.json").read_text())
    repositories = dict(lock["repositories"])
    repositories["diffusers"] = dict(
        url="https://github.com/huggingface/diffusers.git", revision=lock["diffusers_revision"]
    )
    for name, repo in repositories.items():
        destination = PROJECT_ROOT / "external" / name
        if destination.exists():
            current = subprocess.check_output(
                ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
            ).strip()
            if current != repo["revision"]:
                raise SystemExit(f"Revision mismatch: {destination}; inspect before changing it")
            continue
        subprocess.run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", repo["url"], str(destination)],
            check=True,
        )
        subprocess.run(["git", "-C", str(destination), "checkout", repo["revision"]], check=True)


if __name__ == "__main__":
    main()
