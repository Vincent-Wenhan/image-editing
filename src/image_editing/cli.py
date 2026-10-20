"""Lazy subcommands keep data preparation and GPU selection free of CUDA imports."""

import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description="Reproducible image-editing preparation")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    modules = {
        "gpu": "gpu",
        "env": "environment",
        "download": "assets",
        "prepare": "data",
        "validate": "validation",
        "infer": "inference",
        "evaluate": "evaluation",
        "train-config": "training",
        "gallery": "gallery",
    }
    for name, module in modules.items():
        importlib.import_module(f"image_editing.{module}").configure(subparsers.add_parser(name))
    args = parser.parse_args()
    args.func(args)
