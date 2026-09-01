"""Generate SLURM submission scripts for the ImageNet localization sweep.

Writes one .sh per configuration plus:
  - submit_all.sh            — submits every script (use only if all stages wanted at once)
  - submit_stage{1..4}.sh    — per-stage launchers

Every default is environment-driven (see ``paths.py`` and ``scripts/env.sh``):
``IMAGENET_DIR`` supplies --data_dir, ``TAILRL_SWEEP_DIR`` / ``TAILRL_RESULTS_DIR``
the output roots, and the ``TAILRL_SLURM_*`` variables the scheduler directives.

Default output directory:
  $TAILRL_SWEEP_DIR (default <repo>/experiments/imagenet_localization/sweep_scripts/)

CLI:
  python -m experiments.imagenet_localization.sweep \
      [--out_dir DIR] \
      [--results_dir DIR] \
      [--data_dir DIR] \
      [--partition NAME] \
      [--conda_env NAME] \
      [--train_subsample N] \
      [--epochs N] \
      [--time HH:MM:SS] \
      [--include_mse]
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from experiments.imagenet_localization import paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ("tailrl", "binary_maxrl", "grpo", "rloo", "reinforce",
           "ordinal_ce", "cross_entropy")
K_VALUES = (10, 25, 50, 100)
SEEDS = (42, 43, 44)

N_PRIMARY = 64
N_SWEEP_VALUES = (4, 16, 64, 256)

# ---------------------------------------------------------------------------
# Replica matrix — the exact set of (method, K, N) configurations that was
# executed at seed=42 and that the user wants replicated at new seeds to
# enable multi-seed aggregate plots (and, at seed=43 only, gradient / cosine
# analysis). Keep this list in sync with ``results/*_seed42/`` if the matrix
# is ever expanded.
# ---------------------------------------------------------------------------

REPLICA_MATRIX_K50: tuple[tuple[str, int, int], ...] = (
    ("tailrl",                 50,   16),
    ("tailrl",                 50,   64),
    ("tailrl",                 50,  256),
    ("tailrl",                 50, 1024),
    ("binary_maxrl",        50,   64),
    ("grpo",                50,   64),
    ("grpo",                50,  256),
    ("rloo",                50,   64),
    ("rloo",                50,  256),
    ("reinforce",           50,   64),
    ("reinforce",           50,  256),
    ("ordinal_ce",          50,   64),
    ("cross_entropy",            50,   64),
    ("mse",                 50,   64),
    ("mse_iou_match",       50,   64),
    ("mse_centroid_match",  50,   64),
    ("l1_iou_match",        50,   64),
    ("l1_iou_match",        50,  256),
    ("l1_centroid_match",   50,   64),
    ("tailrl_population",      50,   64),
)
assert len(REPLICA_MATRIX_K50) == 20, "replica matrix should have 20 entries"


def replica_configs(seeds: tuple[int, ...]) -> list[tuple[str, int, int, int]]:
    """Cross REPLICA_MATRIX_K50 with the given seeds -> (method, K, N, seed)."""
    return [
        (method, K, N, seed)
        for (method, K, N) in REPLICA_MATRIX_K50
        for seed in seeds
    ]

REPO_DIR = paths.repo_root()
EXPERIMENT_DIR = paths.experiment_dir()

DEFAULT_OUT_DIR = paths.sweep_dir()
DEFAULT_RESULTS_DIR = paths.results_dir()
# Empty when IMAGENET_DIR is unset — resolved (and diagnosed) in main().
DEFAULT_DATA_DIR = paths.imagenet_dir()
# Cluster-specific knobs; empty partition means "let SLURM pick the default".
DEFAULT_PARTITION = os.environ.get("TAILRL_SLURM_PARTITION", "").strip()
DEFAULT_CONDA_ENV = os.environ.get("TAILRL_CONDA_ENV", "").strip()
DEFAULT_EPOCHS = 30
# 0 means "no subsample" — run.py treats --train_subsample absent as full data.
DEFAULT_TRAIN_SUBSAMPLE = 0
DEFAULT_TIME = os.environ.get("TAILRL_SLURM_TIME", "").strip() or "48:00:00"

# SLURM job script template.  Variables wrapped in {{}} are literal braces
# (Python format escapes); single-brace {} are format placeholders.
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
{partition_line}#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={log_dir}/{job_name}_%j.out

# `set -u` would be nice but breaks conda activation: some conda env-activate
# scripts (e.g. activate-binutils_linux-64.sh) reference unbound vars like
# ADDR2LINE and hard-fail under -u. We keep -e + -o pipefail for safety.
set -eo pipefail

# wandb reads credentials from ~/.netrc (written by `wandb login`), so no
# WANDB_API_KEY env export is needed. The earlier strict `${{VAR:?...}}` check
# was killing jobs before they started.
export WANDB_DIR="{wandb_dir}"
{wandb_entity_line}export WANDB_RUN_GROUP="{wandb_group}"

cd "{repo_dir}"

{conda_activate}

python -m experiments.imagenet_localization.run \\
    --method {method} \\
    --K {K} \\
    --N {N} \\
    --seed {seed} \\
    --epochs {epochs} \\
    --batch_size 128 \\
    --lr 5e-4 \\
    --data_dir "{data_dir}" \\
    --output_dir "{run_output_dir}" \\
    --num_workers 8 \\
{subsample_line}\
    --wandb
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conda_activate_block(conda_env: str) -> str:
    """Return bash snippet that activates *conda_env*.

    If *conda_env* is an empty string, returns an empty string (no
    activation).  Otherwise emits ``source ~/.bashrc; conda activate ENV``.
    """
    if not conda_env:
        return ""
    return f"source ~/.bashrc\nconda activate {conda_env}"


def _partition_line(partition: str) -> str:
    """Return the ``#SBATCH --partition`` directive for *partition*.

    An empty ``--partition=`` is a submission error, so when no partition is
    configured we emit no directive at all and let SLURM pick its default.
    """
    if not partition:
        return ""
    return f"#SBATCH --partition={partition}\n"


def _slurm_resources() -> dict[str, str]:
    """Return the per-job resource directives, read from ``TAILRL_SLURM_*``.

    Defaults match ``scripts/env.sh`` so generating scripts without sourcing it
    produces the same job specification.
    """
    return {
        "gres": os.environ.get("TAILRL_SLURM_GRES", "").strip() or "gpu:1",
        "cpus": os.environ.get("TAILRL_SLURM_CPUS", "").strip() or "8",
        "mem": os.environ.get("TAILRL_SLURM_MEM", "").strip() or "32G",
    }


def _wandb_entity_line() -> str:
    """Return the ``WANDB_ENTITY`` export, or "" to use the account default."""
    entity = paths.wandb_entity()
    return f'export WANDB_ENTITY="{entity}"\n' if entity else ""


def _job_name(method: str, K: int, N: int, seed: int) -> str:
    """Return an opaque SLURM job name — deterministic 8-char hex derived from
    the (method, K, N, seed) tuple. This keeps `squeue` on a shared cluster
    from revealing which experiment / method is being run. The mapping is
    reproducible so logs can still be matched back to configs locally.
    """
    import hashlib
    key = f"{method}|{K}|{N}|{seed}".encode()
    digest = hashlib.md5(key).hexdigest()[:8]
    return f"j_{digest}"


def _script_filename(method: str, K: int, N: int, seed: int) -> str:
    """Return the .sh filename for a given configuration."""
    return f"{method}_K{K}_N{N}_seed{seed}.sh"


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


def generate_script_content(
    method: str,
    K: int,
    N: int,
    seed: int,
    results_dir: str,
    data_dir: str,
    partition: str,
    conda_env: str,
    train_subsample: int,
    epochs: int,
    time: str,
    log_dir: str,
    repo_dir: str,
) -> str:
    """Return the full text of a SLURM submission script for one config."""
    job_name = _job_name(method, K, N, seed)
    # Per-run output dir so W&B / checkpoints never collide between seeds.
    run_name = f"{method}_K{K}_N{N}_seed{seed}"
    run_output_dir = os.path.join(results_dir, run_name)
    # W&B group: group by (method, K, N) so the three seeds appear together.
    wandb_group = f"{method}_K{K}_N{N}"
    conda_activate = _conda_activate_block(conda_env)

    # train_subsample <= 0 means "no subsample" — omit the flag entirely so
    # run.py falls back to its default (None = full dataset).
    if train_subsample and train_subsample > 0:
        subsample_line = f"    --train_subsample {train_subsample} \\\n"
    else:
        subsample_line = ""

    return SLURM_TEMPLATE.format(
        job_name=job_name,
        partition_line=_partition_line(partition),
        **_slurm_resources(),
        time=time,
        log_dir=log_dir,
        wandb_dir=paths.wandb_dir(),
        wandb_entity_line=_wandb_entity_line(),
        wandb_group=wandb_group,
        repo_dir=repo_dir,
        conda_activate=conda_activate,
        method=method,
        K=K,
        N=N,
        seed=seed,
        epochs=epochs,
        data_dir=data_dir,
        run_output_dir=run_output_dir,
        subsample_line=subsample_line,
    )


def write_script(
    out_dir: str,
    method: str,
    K: int,
    N: int,
    seed: int,
    results_dir: str,
    data_dir: str,
    partition: str,
    conda_env: str,
    train_subsample: int,
    epochs: int,
    time: str,
    log_dir: str,
    repo_dir: str,
) -> str:
    """Write one .sh file and make it executable; return its absolute path."""
    content = generate_script_content(
        method=method,
        K=K,
        N=N,
        seed=seed,
        results_dir=results_dir,
        data_dir=data_dir,
        partition=partition,
        conda_env=conda_env,
        train_subsample=train_subsample,
        epochs=epochs,
        time=time,
        log_dir=log_dir,
        repo_dir=repo_dir,
    )
    filename = _script_filename(method, K, N, seed)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as fh:
        fh.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------


def stage_groups(include_mse: bool = False) -> dict[int, list[tuple[str, int, int, int]]]:
    """Return a dict mapping stage number → list of (method, K, N, seed) tuples.

    Stage 1 (headline, 24):        tailrl/binary_maxrl × K × seeds, N=64
    Stage 2 (supervised, 24+12):   ordinal_ce/cross_entropy × K × seeds, N=64
                                   (+mse × K × seeds if include_mse)
    Stage 3 (other RL, 36):        grpo/rloo/reinforce × K × seeds, N=64
    Stage 4 (supplementary, 26):   N-sweep (24) + K=2 sanity (2)
    """
    stages: dict[int, list[tuple[str, int, int, int]]] = {1: [], 2: [], 3: [], 4: []}

    # ---------- Stage 1: headline RL ----------
    for method in ("tailrl", "binary_maxrl"):
        for K in K_VALUES:
            for seed in SEEDS:
                stages[1].append((method, K, N_PRIMARY, seed))

    # ---------- Stage 2: supervised ----------
    for method in ("ordinal_ce", "cross_entropy"):
        for K in K_VALUES:
            for seed in SEEDS:
                stages[2].append((method, K, N_PRIMARY, seed))

    if include_mse:
        for K in K_VALUES:
            for seed in SEEDS:
                stages[2].append(("mse", K, N_PRIMARY, seed))

    # ---------- Stage 3: other RL baselines ----------
    for method in ("grpo", "rloo", "reinforce"):
        for K in K_VALUES:
            for seed in SEEDS:
                stages[3].append((method, K, N_PRIMARY, seed))

    # ---------- Stage 4: supplementary ----------
    # N-sweep: tailrl/binary_maxrl × K=50 × N ∈ {4,16,64,256} × seeds.
    # N=N_PRIMARY (=64) is skipped because Stage 1 already generates
    # (method, K=50, N=64, seed) for every seed — including it here would
    # cause submit_all.sh and submit_stage4.sh to re-submit the same script.
    for method in ("tailrl", "binary_maxrl"):
        for N in N_SWEEP_VALUES:
            if N == N_PRIMARY:
                continue
            for seed in SEEDS:
                stages[4].append((method, 50, N, seed))

    # K=2 sanity: tailrl/binary_maxrl × K=2 × N=64 × seed=42
    for method in ("tailrl", "binary_maxrl"):
        stages[4].append((method, 2, N_PRIMARY, 42))

    return stages


# ---------------------------------------------------------------------------
# Submit-script writers
# ---------------------------------------------------------------------------


def _write_launcher(path: str, script_paths: list[str], header_comment: str) -> None:
    """Write a launcher shell script that sbatches every script in *script_paths*."""
    lines = [
        "#!/bin/bash",
        "# " + header_comment,
        "set -euo pipefail",
        "",
    ]
    for sp in script_paths:
        lines.append(f'sbatch "{sp}"')
    lines.append("")  # trailing newline
    content = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def write_submit_all(out_dir: str, per_stage_scripts: dict[int, list[str]]) -> str:
    """Write submit_all.sh; return its path."""
    all_scripts: list[str] = []
    for stage_num in sorted(per_stage_scripts.keys()):
        all_scripts.extend(per_stage_scripts[stage_num])
    path = os.path.join(out_dir, "submit_all.sh")
    total = len(all_scripts)
    _write_launcher(path, all_scripts, f"Submit all {total} sweep scripts across all stages.")
    return path


def write_submit_stage(out_dir: str, stage_num: int, scripts: list[str]) -> str:
    """Write submit_stage{N}.sh; return its path."""
    path = os.path.join(out_dir, f"submit_stage{stage_num}.sh")
    _write_launcher(
        path,
        scripts,
        f"Submit Stage {stage_num} ({len(scripts)} scripts).",
    )
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where generated .sh scripts are written.",
    )
    parser.add_argument(
        "--results_dir",
        default=DEFAULT_RESULTS_DIR,
        help="Root directory for per-run output (checkpoints, logs).",
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR or None,
        help="Root directory of the ImageNet dataset (default: $IMAGENET_DIR).",
    )
    parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
        help=(
            "SLURM partition name (default: $TAILRL_SLURM_PARTITION; empty "
            "emits no --partition directive). Confirm with your cluster admin "
            "before submitting."
        ),
    )
    parser.add_argument(
        "--conda_env",
        default=DEFAULT_CONDA_ENV,
        help=(
            "Conda environment to activate (default: $TAILRL_CONDA_ENV). "
            "Pass '' to skip activation."
        ),
    )
    parser.add_argument(
        "--train_subsample",
        type=int,
        default=DEFAULT_TRAIN_SUBSAMPLE,
        help="Number of training images to subsample (passed as --train_subsample to run.py).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs per run.",
    )
    parser.add_argument(
        "--time",
        default=DEFAULT_TIME,
        help="SLURM wall-clock time limit per job, HH:MM:SS (default: $TAILRL_SLURM_TIME).",
    )
    parser.add_argument(
        "--repo_dir",
        default=REPO_DIR,
        help="Absolute path to the repository root (used as `cd` target in scripts).",
    )
    parser.add_argument(
        "--include_mse",
        action="store_true",
        default=False,
        help=(
            "Include optional MSE regressor baseline in Stage 2 "
            "(adds 4 K-values × 3 seeds = 12 extra scripts; total 116)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("full_sweep", "replica"),
        default="full_sweep",
        help=(
            "full_sweep: generate the 4-stage headline/supervised/RL/N-sweep "
            "matrix (default). replica: generate only the 21-config "
            "seed-42 replica matrix at --replica_seeds, with no stages."
        ),
    )
    parser.add_argument(
        "--replica_seeds",
        type=int,
        nargs="+",
        default=[43],
        help=(
            "Seeds to instantiate the replica matrix at (only used when "
            "--mode=replica). Default: [43]. The seed-42 runs are already on "
            "disk so we default to seed 43 only."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _generate_batch(
    configs: list[tuple[str, int, int, int]],
    args,
    out_dir: str,
    log_dir: str,
) -> list[str]:
    """Emit one .sh per config and return the list of absolute script paths."""
    script_paths: list[str] = []
    for (method, K, N, seed) in configs:
        path = write_script(
            out_dir=out_dir,
            method=method,
            K=K,
            N=N,
            seed=seed,
            results_dir=args.results_dir,
            data_dir=args.data_dir,
            partition=args.partition,
            conda_env=args.conda_env,
            train_subsample=args.train_subsample,
            epochs=args.epochs,
            time=args.time,
            log_dir=log_dir,
            repo_dir=args.repo_dir,
        )
        script_paths.append(path)
    return script_paths


def main() -> None:
    args = parse_args()

    # The generated scripts bake in an absolute dataset path, so resolve it now
    # rather than emitting jobs that fail on the compute node.
    args.data_dir = args.data_dir or paths.require_imagenet_dir()

    out_dir = args.out_dir
    # Logs sit beside the generated scripts unless TAILRL_LOG_DIR overrides it.
    log_dir = (
        paths.log_dir() if os.environ.get("TAILRL_LOG_DIR", "").strip()
        else os.path.join(out_dir, "logs")
    )
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.mode == "replica":
        # --- Replica-matrix mode: 21 configs × |replica_seeds| ------------
        configs = replica_configs(tuple(args.replica_seeds))
        script_paths = _generate_batch(configs, args, out_dir, log_dir)

        # Per-seed launcher (e.g. submit_replica_seed43.sh)
        for seed in args.replica_seeds:
            seed_scripts = [
                p for p in script_paths if p.endswith(f"_seed{seed}.sh")
            ]
            path = os.path.join(out_dir, f"submit_replica_seed{seed}.sh")
            _write_launcher(
                path, seed_scripts,
                f"Submit replica matrix ({len(seed_scripts)} scripts) at seed {seed}.",
            )

        # Combined launcher across all replica seeds.
        all_path = os.path.join(out_dir, "submit_replica_all.sh")
        _write_launcher(
            all_path, script_paths,
            f"Submit the full replica matrix ({len(script_paths)} scripts) "
            f"across seeds {args.replica_seeds}.",
        )

        print(f"Wrote {len(script_paths)} replica scripts to {out_dir}")
        for seed in args.replica_seeds:
            n = sum(1 for p in script_paths if p.endswith(f"_seed{seed}.sh"))
            print(f"  seed {seed}: {n} scripts")
        seed_list = ",".join(str(s) for s in args.replica_seeds)
        print(
            f"Launchers written: submit_replica_all.sh + "
            f"submit_replica_seed{{{seed_list}}}.sh"
        )
        return

    # --- Full-sweep mode (default) ----------------------------------------
    stages = stage_groups(include_mse=args.include_mse)
    per_stage_scripts: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    for stage_num, configs in stages.items():
        per_stage_scripts[stage_num] = _generate_batch(configs, args, out_dir, log_dir)

    # Write launcher scripts
    write_submit_all(out_dir, per_stage_scripts)
    for stage_num, scripts in per_stage_scripts.items():
        write_submit_stage(out_dir, stage_num, scripts)

    total = sum(len(s) for s in per_stage_scripts.values())
    print(f"Wrote {total} scripts to {out_dir}")
    for stage_num, scripts in sorted(per_stage_scripts.items()):
        print(f"  Stage {stage_num}: {len(scripts)} scripts")
    print(f"Launchers written: submit_all.sh + submit_stage{{1..4}}.sh")


if __name__ == "__main__":
    main()
