"""Installed package and command contracts work outside the checkout, without CUDA."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from src.cli import COMMANDS, build_parser


def test_all_command_defaults_match_pre_refactor_snapshot():
    legacy = json.loads((Path(__file__).parent / "fixtures/legacy_cli_defaults.json").read_text())
    parser = build_parser()
    assert set(COMMANDS) == set(legacy) | {"prepare-controls"}
    for name, expected in legacy.items():
        actual = vars(parser.parse_args([name]))
        actual.pop("func")
        actual.pop("subcommand")
        if name == "evaluate":
            assert actual.pop("reference_run") is None
        if name in {"train-config", "infer"}:
            assert actual.pop("control") == "none"
            assert actual.pop("control_root") is None
        if name == "infer":
            assert actual.pop("reference_preprocessing") == "auto"
        assert actual == expected, name


def test_installed_entrypoint_and_light_imports_outside_checkout(tmp_path):
    distribution = importlib.metadata.distribution("image-editing-research")
    entry = next(item for item in distribution.entry_points if item.name == "image-edit")
    assert entry.value == "src.cli:main"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    script = """
import contextlib
import importlib
import importlib.util
import io
import pkgutil
import sys
import src
from src.cli import COMMANDS, main
for module in pkgutil.walk_packages(src.__path__, prefix='src.'):
    if not module.name.endswith('.__main__'):
        importlib.import_module(module.name)
for command in [[], *[[name] for name in COMMANDS]]:
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            main([*command, '--help'])
        except SystemExit as error:
            assert error.code == 0
assert 'torch' not in sys.modules
assert 'diffusers' not in sys.modules
import src.datasets
upstream = importlib.util.find_spec('datasets')
assert upstream is None or upstream.origin != src.datasets.__file__
"""
    subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=environment, check=True, timeout=60
    )
    for command in (
        [sys.executable, "-m", "src"],
        [str(Path(sys.executable).parent / "image-edit")],
    ):
        result = subprocess.run(
            [*command, "--help"],
            cwd=tmp_path,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert "train-config" in result.stdout
