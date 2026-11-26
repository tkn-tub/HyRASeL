"""Command-line arguments of configs/defaults.yml, as passed by src/pipeline.py."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DEFAULTS_PATH, to_cli_args


def default_args(section: str) -> list[str]:
    """Arguments of one section (postprocess, extract or plots) of configs/defaults.yml."""
    defaults = yaml.safe_load(DEFAULTS_PATH.read_text())
    return to_cli_args(defaults[section])
