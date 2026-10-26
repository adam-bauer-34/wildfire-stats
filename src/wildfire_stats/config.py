"""Load the yaml files in the repo-level ``config/`` directory.

Nothing is cached, so edits to a yaml file are picked up on the next call.
Set ``WILDFIRE_STATS_CONFIG_DIR`` to read config from somewhere else.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Return the directory holding the config yaml files."""
    return Path(os.environ.get("WILDFIRE_STATS_CONFIG_DIR", REPO_ROOT / "config"))


def load_yaml(name: str) -> dict:
    """Read ``config/<name>`` and return its contents as a dict."""
    with open(config_dir() / name) as f:
        return yaml.safe_load(f) or {}


def load_assumptions() -> dict:
    return load_yaml("default_assumptions.yaml")


def load_dist_bounds() -> dict:
    return load_yaml("dist_bounds_param_clips.yaml")


def load_diff_evo() -> dict:
    return load_yaml("diff_evo.yaml")


def load_poisson() -> dict:
    return load_yaml("poisson.yaml")


def resolve(value, key: str):
    """Return ``value``, or the ``key`` entry of default_assumptions.yaml if ``value`` is None."""
    return load_assumptions()[key] if value is None else value


@dataclass(frozen=True)
class Paths:
    data_head: Path
    modis: Path
    mtbs: Path
    gfa: Path
    idaho: Path
    evt: Path
    logged_output: Path
    output: Path
    figs: Path


def load_paths() -> Paths:
    """Build data/output directories from ``config/paths.yaml``.

    ``*_SUBDIR`` entries are relative to ``DATA_HEAD``; ``FIGS_DIR`` is
    relative to the repo root (unless absolute).
    """
    p = load_yaml("paths.yaml")
    head = Path(p["DATA_HEAD"])
    return Paths(
        data_head=head,
        modis=head / p["MODIS_SUBDIR"],
        mtbs=head / p["MTBS_SUBDIR"],
        gfa=head / p["GFA_SUBDIR"],
        idaho=head / p["IDAHO_SUBDIR"],
        evt=head / p["EVT_SUBDIR"],
        logged_output=head / p["LOGGED_OUTPUT_SUBDIR"],
        output=head / p["OUTPUT_SUBDIR"],
        figs=REPO_ROOT / p["FIGS_DIR"],
    )
