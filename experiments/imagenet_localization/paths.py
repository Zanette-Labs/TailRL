"""Environment-driven path resolution for the ImageNet localization experiment.

Nothing in this repository hardcodes an author-specific filesystem. Every path a
fresh clone needs is read from an environment variable that falls back to a
repo-relative default, and an explicit command-line flag always wins over both.

Recognised variables
--------------------
``TAILRL_ROOT``
    Repository root. Defaults to the directory two levels above this file, which
    is correct for an in-place clone; set it explicitly if you run the modules
    from an installed wheel or a copied tree.
``IMAGENET_DIR``
    Root of the ImageNet-1k *localization* release — the directory holding
    ``LOC_train_solution.csv``, ``LOC_val_solution.csv`` and ``ILSVRC/``.
    There is deliberately **no default**: point it at your own copy.
``TAILRL_RESULTS_DIR``
    Where per-run output directories are created.
    Default: ``$TAILRL_ROOT/results/imagenet_localization``.
``TAILRL_FIGURES_DIR``
    Where generated figures are written.
    Default: ``$TAILRL_ROOT/figures/imagenet_localization``.
``TAILRL_SWEEP_DIR``
    Where generated SLURM job scripts are written.
    Default: ``<experiment dir>/sweep_scripts`` (gitignored — it is build output).
``TAILRL_LOG_DIR``
    Where SLURM writes job stdout/stderr. Default: ``$TAILRL_SWEEP_DIR/logs``.
``WANDB_PROJECT`` / ``WANDB_ENTITY`` / ``WANDB_DIR``
    Weights & Biases configuration. ``WANDB_ENTITY`` is intentionally unset by
    default so runs land in *your* default entity. Training works without any of
    these — omit ``--wandb`` and no W&B call is made.

Every getter returns an absolute :class:`str`, so the values drop straight into
``argparse`` defaults and f-string command lines.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "experiment_dir",
    "imagenet_dir",
    "require_imagenet_dir",
    "results_dir",
    "figures_dir",
    "sweep_dir",
    "log_dir",
    "wandb_project",
    "wandb_entity",
    "wandb_dir",
]

#: Name of the environment variable that must point at the ImageNet copy.
IMAGENET_DIR_VAR = "IMAGENET_DIR"

#: Default W&B project when ``WANDB_PROJECT`` is unset.
DEFAULT_WANDB_PROJECT = "tailrl-imagenet-localization"


def _env_path(var: str, default: Path) -> str:
    """Return ``$var`` expanded to an absolute path, or *default*."""
    raw = os.environ.get(var, "").strip()
    if raw:
        return str(Path(raw).expanduser().resolve())
    return str(default)


def repo_root() -> str:
    """Absolute path to the repository root (``$TAILRL_ROOT``)."""
    return _env_path("TAILRL_ROOT", Path(__file__).resolve().parents[2])


def experiment_dir() -> str:
    """Absolute path to ``experiments/imagenet_localization``."""
    return str(Path(repo_root()) / "experiments" / "imagenet_localization")


def imagenet_dir() -> str:
    """Absolute path to the ImageNet root, or ``""`` if ``IMAGENET_DIR`` is unset.

    Returns the empty string rather than raising so that ``argparse`` defaults
    stay importable on a machine without the dataset (the test suite relies on
    this). Call :func:`require_imagenet_dir` at the point of use instead.
    """
    raw = os.environ.get(IMAGENET_DIR_VAR, "").strip()
    return str(Path(raw).expanduser().resolve()) if raw else ""


def require_imagenet_dir() -> str:
    """Like :func:`imagenet_dir`, but raise a actionable error when unset."""
    value = imagenet_dir()
    if not value:
        raise SystemExit(
            f"{IMAGENET_DIR_VAR} is not set and --data_dir was not passed.\n"
            "Point it at your ImageNet localization release, e.g.\n"
            f"    export {IMAGENET_DIR_VAR}=/path/to/imagenet\n"
            "That directory must contain LOC_train_solution.csv, "
            "LOC_val_solution.csv and ILSVRC/ — see the experiment README."
        )
    if not os.path.isdir(value):
        raise SystemExit(f"{IMAGENET_DIR_VAR}={value!r} is not a directory.")
    return value


def results_dir() -> str:
    """Absolute path under which per-run output directories are created."""
    return _env_path(
        "TAILRL_RESULTS_DIR", Path(repo_root()) / "results" / "imagenet_localization"
    )


def figures_dir() -> str:
    """Absolute path where generated figures are written."""
    return _env_path(
        "TAILRL_FIGURES_DIR", Path(repo_root()) / "figures" / "imagenet_localization"
    )


def sweep_dir() -> str:
    """Absolute path where generated SLURM job scripts are written."""
    return _env_path("TAILRL_SWEEP_DIR", Path(experiment_dir()) / "sweep_scripts")


def log_dir() -> str:
    """Absolute path where SLURM job stdout/stderr is written."""
    return _env_path("TAILRL_LOG_DIR", Path(sweep_dir()) / "logs")


def wandb_project() -> str:
    """W&B project name (``$WANDB_PROJECT``)."""
    return os.environ.get("WANDB_PROJECT", "").strip() or DEFAULT_WANDB_PROJECT


def wandb_entity() -> str:
    """W&B entity (``$WANDB_ENTITY``), or ``""`` to use your account default."""
    return os.environ.get("WANDB_ENTITY", "").strip()


def wandb_dir() -> str:
    """Local W&B cache directory (``$WANDB_DIR``)."""
    return _env_path("WANDB_DIR", Path(repo_root()) / "wandb")
