#!/bin/bash
# Shared runner for the reproduce/*.sh scripts.
#
# Sourced, not executed. Provides run_one(), which trains a single arm into its
# own per-run directory under $TAILRL_RESULTS_DIR and skips work that is already
# finished, so an interrupted sweep can simply be re-run.
#
# Knobs (export before calling a reproduce script):
#   SEEDS="42 43 44"   seeds to run          (paper uses 3)
#   EPOCHS=30          epochs per run        (paper uses 30)
#   K=50               bins per coordinate   (paper uses 50)
#   BATCH_SIZE=128     per-GPU batch size
#   TRAIN_SUBSAMPLE=   images per epoch; empty = full split. Set e.g. 20000 for a
#                      fast smoke pass over the whole matrix.
#   WANDB=0            1 adds --wandb
#   FORCE=0            1 re-runs arms that already have a metrics.json
#   DRY_RUN=0          1 prints the commands without running them
#   EXTRA_ARGS=        appended verbatim to every run.py invocation

set -eo pipefail

# Resolve the repo and the environment (this file lives in scripts/reproduce/).
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_HERE}/../env.sh"

: "${SEEDS:=42 43 44}"
: "${EPOCHS:=30}"
: "${K:=50}"
: "${BATCH_SIZE:=128}"
: "${TRAIN_SUBSAMPLE:=}"
: "${WANDB:=0}"
: "${FORCE:=0}"
: "${DRY_RUN:=0}"
: "${EXTRA_ARGS:=}"

_RUNS_STARTED=0
_RUNS_SKIPPED=0
_RUNS_FAILED=0

# run_one <method> <N> <seed> [reward_transform]
#
# The output directory name must match what the plotting scripts glob:
#   {method}[_{reward_transform}]_K{K}_N{N}_seed{seed}
run_one() {
    local method="$1" n="$2" seed="$3" xform="${4:-none}"

    local tag=""
    [[ "${xform}" != "none" ]] && tag="_${xform}"
    local name="${method}${tag}_K${K}_N${n}_seed${seed}"
    local out="${TAILRL_RESULTS_DIR}/${name}"

    if [[ -f "${out}/metrics.json" && "${FORCE}" != "1" ]]; then
        echo "  [skip] ${name} (metrics.json exists; FORCE=1 to redo)"
        _RUNS_SKIPPED=$((_RUNS_SKIPPED + 1))
        return 0
    fi

    local -a cmd=(
        "${TAILRL_PYTHON}" -m experiments.imagenet_localization.run
        --method "${method}"
        --K "${K}"
        --N "${n}"
        --seed "${seed}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --data_dir "${IMAGENET_DIR}"
        --output_dir "${out}"
    )
    [[ "${xform}" != "none" ]] && cmd+=(--reward_transform "${xform}")
    [[ -n "${TRAIN_SUBSAMPLE}" ]] && cmd+=(--train_subsample "${TRAIN_SUBSAMPLE}")
    [[ "${WANDB}" == "1" ]] && cmd+=(--wandb)
    # shellcheck disable=SC2206
    [[ -n "${EXTRA_ARGS}" ]] && cmd+=(${EXTRA_ARGS})

    echo "  [run ] ${name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        ( IFS=' '; echo "         ${cmd[*]}" )
        return 0
    fi

    _RUNS_STARTED=$((_RUNS_STARTED + 1))
    # Keep the training log next to the checkpoints it produced, so a run
    # directory is self-contained. Appends, so a resumed arm keeps its history.
    mkdir -p "${out}"
    # A single failing arm should not abandon the rest of the matrix.
    if ! ( cd "${TAILRL_ROOT}" && "${cmd[@]}" 2>&1 | tee -a "${out}/train.log" ); then
        echo "  [FAIL] ${name} — see ${out}/train.log — continuing" >&2
        _RUNS_FAILED=$((_RUNS_FAILED + 1))
    fi
}

# Print a summary and exit non-zero if anything failed.
finish() {
    echo
    echo "=============================================================="
    echo "  ran ${_RUNS_STARTED}, skipped ${_RUNS_SKIPPED}, failed ${_RUNS_FAILED}"
    echo "  results: ${TAILRL_RESULTS_DIR}"
    echo "=============================================================="
    [[ "${_RUNS_FAILED}" -eq 0 ]]
}
