#!/usr/bin/env bash
# Generic SLURM wrapper for one training arm. Site-specific values come from the environment, so
# nothing here needs editing for your cluster:
#
#   export TAILRL_SLURM_ACCOUNT=my-account TAILRL_SLURM_PARTITION=gpu \
#          TAILRL_SLURM_GRES=gpu:4 TAILRL_SLURM_TIME=47:00:00
#   sbatch scripts/slurm/train.sbatch --method tailrl --model 3b --seed 1
#
# The full comparison is three submissions differing in --method only:
#   for m in tailrl grpo rloo; do sbatch scripts/slurm/train.sbatch --method $m --model 3b; done
#
#SBATCH --job-name=tailrl_gui
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# --account / --partition / --gres / --time / --cpus-per-task are intentionally NOT #SBATCH
# directives: they are cluster-specific and are passed by the submit line below via sbatch flags,
# which override directives. If your site requires them as directives, add them here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

# A 47h slice that ends before the wall clock does; the run resumes from its last full-state
# checkpoint, so chaining slices with --dependency=afterany costs restarts but never progress.
exec bash scripts/train.sh "$@"
