"""Evaluation configs: configs/<Config>.yml merged into configs/defaults.yml (see README.md).

The sections postprocess and extract become the command-line options of their
scripts (to_cli_args); plots is read by plot.py.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Evaluation directory: configs, results and working directory of all steps.
ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = ROOT / "configs" / "defaults.yml"


def deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base; nested dicts are merged, everything else replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path, defaults_path: Path = DEFAULTS_PATH) -> dict:
    """Load a config merged into the defaults; its name is the file name."""
    defaults = yaml.safe_load(defaults_path.read_text()) or {}
    config = yaml.safe_load(config_path.read_text()) or {}
    merged = deep_merge(defaults, config)
    merged["name"] = config_path.stem
    # Absolute: the plot job runs with ROOT as working directory.
    merged["path"] = str(config_path.resolve())
    # Raw traces of the simulation config with the same name
    if "data" not in merged:
        merged["data"] = str(Path(merged["data_root"]) / merged["name"])
    # A section or sweep without entries is None in YAML.
    for section in ("postprocess", "extract", "plots"):
        merged[section] = merged.get(section) or {}
    merged["sweep"] = merged.get("sweep") or {}
    # Plot-only config: {source config: {method: name in the figures}}; a list keeps the names.
    merged["sources"] = {
        source: dict(zip(methods, methods)) if isinstance(methods, list) else dict(methods)
        for source, methods in (merged.get("sources") or {}).items()
    }
    return merged


def to_cli_args(options: dict) -> list[str]:
    """Translate a config section into command-line arguments."""
    args = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        if value is True:
            args.append(f"--{key}")
        elif isinstance(value, list):
            args.extend([f"--{key}", *map(str, value)])
        else:
            args.extend([f"--{key}", str(value)])
    return args
