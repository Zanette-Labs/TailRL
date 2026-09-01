#!/usr/bin/env bash
# Shared helpers for the reproduce/ scripts. Source after scripts/env.sh.
# shellcheck shell=bash

# The three arms of the comparison. The untrained model is evaluated too, as step 0.
# Override to narrow a sweep:  METHODS=tailrl bash scripts/reproduce/train_all.sh
METHODS="${METHODS:-tailrl grpo rloo}"
SEEDS="${SEEDS:-0}"

# The checkpoints evaluated in the paper. Each arm is evaluated at the step where it
# had converged; they differ because the arms converge at different rates.
EVAL_STEPS_tailrl="${EVAL_STEPS_tailrl:-300}"
EVAL_STEPS_grpo="${EVAL_STEPS_grpo:-500}"
EVAL_STEPS_rloo="${EVAL_STEPS_rloo:-500}"

DRY="${DRY:-0}"
for arg in "$@"; do
  [ "${arg}" = "--dry-run" ] && DRY=1
done

run() {
  if [ "${DRY}" -eq 1 ]; then
    printf '  '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

run_tag() {   # run_tag <method> <seed>
  local model_tag="${PIE_MODEL_TAG:-$(basename "${BASE_MODEL}" | tr '[:upper:]' '[:lower:]')}"
  echo "${model_tag}_${1}_g${PIE_G:-16}_bs${PIE_B:-64}_s${2}"
}

eval_step_for() {   # eval_step_for <method>
  local v="EVAL_STEPS_${1}"
  echo "${!v}"
}
