"""Set the project seed before executing the pinned external trainer."""

from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path


def _seed(value: int) -> None:
    import numpy as np
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, required=True)
    args, remaining = parser.parse_known_args()
    if args.seed < 0:
        raise ValueError("Qwen training seed must be non-negative")
    _seed(args.seed)
    sys.argv = [sys.argv[0], *remaining]
    trainer = Path.cwd() / "train_qwen_edit_lora.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"Pinned Inter-Edit trainer is missing: {trainer}")
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
