"""Consistent plain progress messages; experiment records carry structured data."""

import logging
import sys


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
