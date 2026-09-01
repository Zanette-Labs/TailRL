"""Loader for ``config.yaml``, the experiment's hyperparameter defaults.

``run.py`` calls :func:`argparse_defaults` and feeds the result into its
``ArgumentParser`` defaults, so precedence is

    command-line flag  >  config.yaml  >  the fallback hardcoded in run.py

Paths are deliberately not part of this file — they are machine-specific and
come from the environment via :mod:`paths`. Everything here is machine-independent.

The YAML is grouped into sections purely for readability; :func:`flatten` maps
each leaf onto the flat argparse destination name that ``run.py`` uses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_CONFIG_PATH", "load", "flatten", "argparse_defaults"]

#: Shipped defaults, alongside this module.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

#: Maps ``<section>.<key>`` in the YAML onto the argparse ``dest`` in run.py.
#: Anything not listed here is passed through under its own leaf name, so adding
#: a key to the YAML and a matching flag to run.py needs no change to this table.
_DEST_OVERRIDES = {
    "eval.n_eval_samples": "N_eval_samples",
}


def load(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Parse the YAML config and return it as nested dicts.

    Args:
        path: config file to read. Defaults to :data:`DEFAULT_CONFIG_PATH`.

    Raises:
        SystemExit: with an actionable message if PyYAML is missing or the file
            cannot be read/parsed — a bad config should not surface as a
            traceback from deep inside argparse construction.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        import yaml
    except ImportError:  # pragma: no cover - exercised only without pyyaml
        raise SystemExit(
            "PyYAML is required to read config.yaml. Install it with\n"
            "    pip install pyyaml\n"
            "or `pip install -r requirements.txt` from the repository root."
        )
    if not cfg_path.is_file():
        raise SystemExit(f"config file not found: {cfg_path}")
    try:
        with open(cfg_path) as fh:
            parsed = yaml.safe_load(fh)
    except Exception as exc:
        raise SystemExit(f"could not parse {cfg_path}: {exc}")
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise SystemExit(f"{cfg_path}: top level must be a mapping of sections")
    return parsed


def flatten(cfg: dict[str, Any]) -> dict[str, Any]:
    """Collapse the section/key nesting into flat argparse destination names.

    Section names are dropped; ``_DEST_OVERRIDES`` renames the few leaves whose
    argparse ``dest`` differs from the YAML key. A duplicate leaf name across two
    sections is a config bug, so it raises rather than silently winning.
    """
    flat: dict[str, Any] = {}
    for section, body in cfg.items():
        if not isinstance(body, dict):
            # A scalar at the top level is already flat.
            flat[section] = body
            continue
        for key, value in body.items():
            dest = _DEST_OVERRIDES.get(f"{section}.{key}", key)
            if dest in flat:
                raise SystemExit(
                    f"config.yaml: duplicate key {dest!r} in section {section!r}; "
                    "leaf names must be unique across sections"
                )
            flat[dest] = value
    return flat


def argparse_defaults(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Return ``{argparse_dest: default}`` ready for ``parser.set_defaults()``."""
    return flatten(load(path))
