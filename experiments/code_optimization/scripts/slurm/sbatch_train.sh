#!/usr/bin/env bash
#SBATCH --job-name=codeopt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Everything else -- partition, account, QOS, gres, cpus, mem, time -- comes from the
# TAILRL_SLURM_* variables on the submit line, because those are the values that
# differ between clusters. See scripts/slurm/submit_all.sh.
#
# Not invoked directly; submit_all.sh fills in the sbatch flags. If you do submit it
# by hand, pass the same arguments train.sh takes:
#
#   sbatch --partition=gpu --gres=gpu:4 --time=24:00:00 \
#          scripts/slurm/sbatch_train.sh --method tailrl
#
# Ten minutes before the time limit SLURM sends USR1; the handler requeues the job.
# Combined with verl's resume_mode=auto, a short walltime costs restarts but never
# progress, so this converges under any queue policy.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

requeue() {
  echo "[slurm] USR1 received, requeueing ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}" || true
}
trap requeue USR1

echo "[slurm] job ${SLURM_JOB_ID:-?} on $(hostname), $(date)"
bash "${EXP_ROOT}/scripts/train.sh" "$@" &
wait $!
