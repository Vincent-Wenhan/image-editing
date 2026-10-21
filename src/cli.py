"""Stable CLI commands with lightweight imports; parsing never initializes CUDA."""

import argparse
import importlib
from collections.abc import Sequence

from src.core.logging import configure_logging

COMMANDS = {
    "gpu": "src.infra.gpu",
    "env": "src.infra.environment",
    "download": "src.infra.assets",
    "prepare": "src.datasets.magicbrush",
    "prepare-controls": "src.datasets.control_export",
    "validate": "src.datasets.validation",
    "infer": "src.inference.runner",
    "evaluate": "src.evaluation.evaluate",
    "train-config": "src.training.config",
    "gallery": "src.evaluation.gallery",
    "review": "src.evaluation.review",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible image-editing preparation")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    for name, module in COMMANDS.items():
        importlib.import_module(module).configure(subparsers.add_parser(name))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = build_parser().parse_args(argv)
    args.func(args)
