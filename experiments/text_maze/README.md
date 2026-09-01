# Text-Maze Navigation

Navigation in 17×17 gridworld mazes written as text. A 3M-parameter decoder is
pretrained from scratch on maze→path pairs, then post-trained with RL against a
*continuous* reward that scores how close a rollout gets to the goal and how its
length compares with the shortest path. A rollout earns reward exactly `1` only
when it reaches the goal along a shortest path — the event we call
**shortest-path success**.

The point of this experiment is the **initialization sweep**. By varying how long
the model is pretrained we get seven starting policies whose shortest-path
success rates span roughly `0.83%` down to `0.012%` — one success in ten thousand
attempts. Then every RL method starts from each of them. This isolates one
question: what happens to an advantage estimator when the high-reward rollouts it
needs to learn from are *attainable but rare*?

---

## 1. Install

This experiment runs RL through a **vendored fork of
[verl](https://github.com/volcengine/verl)** at [`verl/`](verl/). The fork is
where the science lives — the estimators are in
[`verl/trainer/ppo/core_algos.py`](verl/trainer/ppo/core_algos.py) — so it ships
with the experiment rather than being a dependency you install. It is pruned to
what this task actually executes: the FSDP actor, the HuggingFace rollout backend,
and the advantage estimators being compared.

You need a GPU. Everything below is one GPU; nothing here needs more than one.

```bash
git clone https://github.com/Zanette-Labs/TailRL.git
cd TailRL/experiments/text_maze

conda create -n tailrl-maze python=3.10 && conda activate tailrl-maze

# torch first, matched to your CUDA runtime (12.4 shown)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

There is nothing to `pip install -e`. The fork is imported as top-level `verl`
off `PYTHONPATH`, which [`scripts/env.sh`](scripts/env.sh) sets for you — every
script in this directory sources it. Check the wiring:

```bash
source scripts/env.sh
python -c "import verl.trainer.main_ppo, verl.trainer.ppo.core_algos as c; \
           print(sorted(c.ADV_ESTIMATOR_REGISTRY))"
# ['gae', 'grpo', 'grpo_passk', 'maxrl', 'opo', 'pkpo', 'pkpo_continuous',
#  'reinforce', 'reinforce_plus_plus', 'reinforce_plus_plus_baseline',
#  'remax', 'rloo', 'tailrl']
```

`maxrl` there is binary MaxRL. TailRL reduces to it *exactly* when the reward is
binary — that identity is what fixes the leading `N` in the estimator, and it
holds to floating-point round-off in this implementation.

**`transformers>=4.51,<5` is mandatory** — `requirements.txt` pins it and every
launcher re-checks it at startup, because outside that range the checkpoints
load silently wrong rather than failing.

<details>
<summary>Running inside a container</summary>

Set `TAILRL_MAZE_SIF` to an apptainer image and the SLURM wrappers and
`verify_setup.sh` will `apptainer exec --nv` into it; bind your scratch
filesystems with `TAILRL_APPTAINER_BIND`. Any image with the stack above works —
no specific image is required or shipped. Inside a container also set
`PYTHONNOUSERSITE=1` (env.sh does) so `~/.local` cannot shadow the image's
pinned numpy/torch.
</details>

---

## 2. Checkpoints and data

Everything you need is public and ungated on the Hugging Face Hub:
[**checkpoints**](https://huggingface.co/max-rl/maze_v2_sft_ckpts_guanning) (~55 MB)
and the [**dataset**](https://huggingface.co/datasets/max-rl/maze_17x17_diverse_1.3m)
(~4 GB, apache-2.0).

```bash
export HF_CACHE_DIR=$HOME/.cache/tailrl_maze_sft_ckpts
export TAILRL_MAZE_DATA_DIR=/scratch/$USER/tailrl_maze_data   # export FIRST, or ~5 GB lands in the checkout

bash scripts/setup_checkpoints.sh 2450,3000,3250,3350,3400,3450,3550   # download, repair, verify
bash scripts/download_dataset.sh
bash scripts/prepare_dataset.sh
```

Those seven checkpoints are the initialization ladder, in order of shortest-path
success **before** any RL — 0.0122, 0.0244, 0.0427, 0.1556, 0.2197, 0.4822,
0.8270 % — which is the x-axis of the initialization figure. The policy is a
3.9M-parameter Qwen2 decoder (4 layers, 4 heads, hidden 256, vocab 32) trained
from scratch by [`scripts/sft_pretrain.py`](scripts/sft_pretrain.py).

---

## 3. Configure

Only two paths have no sensible default. Everything else is derived; see
[`scripts/env.sh`](scripts/env.sh), which every script sources.

```bash
export TAILRL_MAZE_DATA_DIR=/scratch/$USER/tailrl_maze_data   # built datasets
export HF_CACHE_DIR=$HOME/.cache/tailrl_maze_sft_ckpts        # SFT checkpoints
export TAILRL_MAZE_RUN_DIR=/scratch/$USER/tailrl_maze_runs    # default: ./runs
export WANDB_PROJECT=tailrl-text-maze                         # console logging works alone
```

| Variable | Required | Default |
| --- | --- | --- |
| `TAILRL_MAZE_DATA_DIR` | for training | `<experiment>/data` |
| `HF_CACHE_DIR` | **yes** | `~/.cache/tailrl_maze_sft_ckpts` |
| `TAILRL_MAZE_RUN_DIR` | no | `<experiment>/runs` |
| `WANDB_PROJECT` / `WANDB_ENTITY` | no | `tailrl-text-maze` / your default entity |
| `ROLLOUT_MICRO_BATCH_SIZE` | no | `8000` (use `4000` on ≤48 GB cards) |
| `TAILRL_SLURM_*` | no | generic single-GPU defaults |
| `TAILRL_MAZE_SIF` | no | empty (no container) |

W&B credentials come from `~/.netrc` via `wandb login`. **No API key is read
from, or belongs in, this repository.**

RL checkpoints land in
`$TAILRL_MAZE_RUN_DIR/checkpoints/$WANDB_PROJECT/<experiment_name>/`, where
`<experiment_name>` encodes the full configuration. `max_actor_ckpt_to_keep=3`
bounds each run, but the full 84-run matrix is still tens of GB.

> **`WANDB_PROJECT` at eval time must equal `WANDB_PROJECT` at train time.** The
> evaluation resumes the trained weights from that path. If it does not match,
> the directory does not exist, verl silently falls back to the *base SFT
> checkpoint*, and the job exits 0 with wrong numbers. This invalidated a whole
> evaluation batch once, so `eval_ladder.sh` hard-fails unless the log contains
> `Loaded model from .../global_step_N`.

---

## 4. Reproduce

Four scripts, one per result. Each walks its matrix sequentially on one GPU,
**skips arms that already finished, and resumes interrupted ones** from their last
checkpoint — so an interrupted sweep continues by re-running the same command.
Add `--dry-run` to any of them to print the commands without running anything.

```bash
cd experiments/text_maze

bash scripts/reproduce/sft_pass1_ladder.sh       #  7 evals — the initialization x-axis
bash scripts/reproduce/checkpoint_sweep.sh       # 84 runs  — the headline comparison
bash scripts/reproduce/rollout_budget_sweep.sh   # 48 runs  — the training-budget figure
bash scripts/reproduce/eval_ladder.sh            # Pass@k / Best-of-k, once runs exist
```

**`checkpoint_sweep.sh` is the big one**: 4 estimators × 7 initializations × 3
seeds. Every arm is identical except the estimator and the checkpoint it starts
from.

| `--method` | Estimator |
| --- | --- |
| `tailrl` | TailRL — tail-likelihood, gap-over-survivors |
| `grpo` | Group-relative, z-scored baseline |
| `rloo` | Leave-one-out baseline |
| `pkpo` | Pass@K policy optimization, continuous form (`k_opt = 8`) |

**`rollout_budget_sweep.sh`** holds the initialization at `ckpt-3000` and sweeps
`N ∈ {4, 16, 64, 256}`. The `N = 16` cells are the same runs as the checkpoint
sweep's `ckpt-3000` column, so they are skipped if you already have them.

**`eval_ladder.sh`** re-enters each finished policy with `--val-only` and draws
4096 rollouts per prompt in one pass, giving `best@{2,4,…,4096}` for
`is_shortest` (= Pass@k), `goal_reached`, and `reward` (= Best-of-k reward). It
compiles the console logs into one JSON with per-seed mean and sample std via
[`scripts/eval_logs_to_json.py`](scripts/eval_logs_to_json.py). The headline key
is `val-aux/maze_17_continuous/is_shortest/best@1024`. Console metrics print to
3 decimals — that is the resolution ceiling on everything in `paper_results/`.

Narrow any sweep with environment variables before committing to the full matrix:

```bash
METHODS=tailrl CKPTS=3000 SEEDS=0 bash scripts/reproduce/checkpoint_sweep.sh
```

### One arm, by hand

```bash
bash scripts/train.sh --method tailrl --ckpt-step 3000 --seed 0 --gpu-ids 0 \
  --n-rollouts 16 --batch-size 256 --total-steps 5001 \
  --reward composite_v2 --reward-transform raw \
  --extra-eos-token-ids 7 --val-n 64 --test-freq 1000 --save-freq 250
```

`--dry-run` resolves and prints the whole configuration without starting Ray.
`bash scripts/train.sh --help` lists every flag.

### On a cluster

```bash
export TAILRL_SLURM_PARTITION=gpu TAILRL_SLURM_GRES=gpu:1 TAILRL_SLURM_TIME=24:00:00
bash scripts/slurm/submit_sweep.sh checkpoint_sweep --dry-run   # inspect first
bash scripts/slurm/submit_sweep.sh checkpoint_sweep
```

One job per arm through [`scripts/slurm/sbatch_train.sh`](scripts/slurm/sbatch_train.sh),
which traps `USR1` ten minutes before the time limit and requeues itself.
Combined with verl's `resume_mode=auto` (model, optimizer, RNG and LR schedule all
restored), a short walltime costs restarts but never progress. `submit_sweep.sh`
is idempotent — re-run it on a timer and the campaign converges.

> At `N = 256` a step is slow enough that the default `save_freq=250` may not
> checkpoint before the first requeue, which would restart the run from zero
> forever. Use `SAVE_FREQ=25` for large `N`.

---

## 5. Where the logs and checkpoints are

Everything a run produces goes under `$TAILRL_MAZE_RUN_DIR` (default
`experiments/text_maze/runs/`, gitignored). Nothing is written into the source
tree.

```
$TAILRL_MAZE_RUN_DIR/
├── logs/
│   ├── <experiment_name>.log        # ← the training log. One per arm, full stdout.
│   ├── eval/
│   │   └── eval_<method>_ck<step>_G<N>_s<seed>.log   # ← the best-of-k eval logs
│   ├── sft_pass1/                   # the initialization-ladder eval logs
│   └── <jobname>-<jobid>.{out,err}  # SLURM only, from submit_sweep.sh
├── checkpoints/<WANDB_PROJECT>/<experiment_name>/
│   ├── latest_checkpointed_iteration.txt   # the resume pointer, and the "is it done?" check
│   └── global_step_<N>/                    # actor + optimizer, last 3 kept
├── wandb/                           # W&B run dirs (offline runs sync from here)
└── hydra/<experiment_name>/         # the fully resolved config for that run
```

---

## 6. Layout

```
experiments/text_maze/
├── verl/                     # the vendored fork — estimators in trainer/ppo/core_algos.py
├── src/                      # dataset builder + the reward functions
├── scripts/
│   ├── env.sh                # paths + defaults; sourced by everything
│   ├── train.sh              # single-run launcher
│   ├── sft_pretrain.py       # the pretraining stage that produced the ladder
│   ├── checkpoint_doctor.py  # mandatory checkpoint repair/verify preflight
│   ├── eval_logs_to_json.py  # console eval logs -> one JSON with mean/std
│   ├── reproduce/            # the four scripts of §4
│   ├── slurm/                # generic sbatch wrapper + idempotent sweep submitter
│   └── tests/                # doctor tests, DONE-as-EOS smoke, goal-reaching smoke
├── data/eval1000/            # the committed 1000-maze evaluation split
├── paper_results/            # measured Pass@k / Best-of-k results, with per-seed std
└── docs/RL_PIPELINE.md       # operations guide + the footgun list
```

---

## Acknowledgements

The RL framework is a fork of [verl](https://github.com/volcengine/verl); the
maze task and the SFT trainer descend from
[MaxRL](https://github.com/tajwarfahim/maxrl). Our thanks to both.
