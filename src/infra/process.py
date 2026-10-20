"""Bounded small subprocess queries used by environment and revision checks."""

import subprocess


def command(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()
