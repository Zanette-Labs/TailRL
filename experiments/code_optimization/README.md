# Code Runtime Optimization

Given a working but slow C++ program, rewrite it to run faster without changing what
it prints. The reward is the *measured* speedup: every rollout is compiled, checked
against real test cases, and then simulated cycle-by-cycle on a modelled Skylake
core. A rollout that gets any answer wrong scores **0**, whatever its speed; one that
is correct scores `source_cycles / rollout_cycles`.

Three arms are trained and compared — **TailRL**, **GRPO**, **RLOO** — identical in
every respect except the advantage estimator, then evaluated on 878 held-out programs
with unbiased pass@k and best-of-k estimators.

This README is the reproduction guide: install, data, the gem5 timing setup, training,
evaluation. Follow it top to bottom.

---

## 1. Install

This experiment has two independent stacks, and they never talk to each other.

**The training stack** runs RL through a **vendored fork of
[verl](https://github.com/volcengine/verl)** at [`verl/`](verl/) (upstream commit
`17f283b1`, May 2025). The fork is where the science lives — the estimator dispatch
is in [`verl/trainer/ppo/ray_trainer.py`](verl/trainer/ppo/ray_trainer.py) and the
estimators themselves in [`code_opt/advantages.py`](code_opt/advantages.py) — so it
ships with the experiment rather than being a dependency you install.

```bash
git clone https://github.com/Zanette-Labs/TailRL.git
cd TailRL/experiments/code_optimization

conda create -n tailrl-codeopt python=3.10 && conda activate tailrl-codeopt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
```

There is nothing to `pip install -e`. Both `verl` and `code_opt` are imported off
`PYTHONPATH`, which [`scripts/env.sh`](scripts/env.sh) sets; every script here sources
it. Check the wiring:

```bash
source scripts/env.sh
python -c "from verl.trainer.ppo.ray_trainer import AdvantageEstimator as A; \
           print([e.value for e in A])"
# [..., 'rloo', 'tailrl', 'reinforce_baseline', 'binary_maxrl', 'pkpo_k4', ...]
```

**The measurement stack** is gem5 plus a pinned compiler toolchain, and it needs its
own Python 3.8 environment because gem5 v20.1 does not build on anything newer.
`scripts/setup_gem5.sh` creates it (§3). Nothing here ever does `import m5` — gem5 is
only ever run as a subprocess — so the two environments coexist without interacting.

You need **one 4-GPU node** for training and **many CPU cores**: the reward compiles
and simulates ~1000 programs per training step, and that is what the node's cores are
for. The runs this experiment was built around used a 288-core node with 256 reward
workers.

---

## 2. Data

Two downloads. Export the paths *first* or ~4 GB lands inside your checkout.

```bash
export PIE_TEST_CASE_DIR=/scratch/$USER/pie_test_cases     # ~3.4 GB
export PIE_PARQUET_ROOT=/scratch/$USER/pie_parquet         # ~21 MB

bash scripts/download_data.sh      # the PIE test-case corpus (92 MB download)
bash scripts/prepare_dataset.sh    # build the verl parquets from the Hub
```

`download_data.sh` fetches the PIE paper's own test-case release: ~100 cases per
problem across 3,907 problems. The correctness gate runs a rollout against *every*
usable case for its problem, so a thinner corpus silently redefines "correct".

`prepare_dataset.sh` converts
[`stablegradients/pie-gem5-bysrc`](https://huggingface.co/datasets/stablegradients/pie-gem5-bysrc)
(public, ungated) into the three-column parquets verl reads. This step is mandatory —
the Hub files carry `src_code` plus 51 provenance columns and no rendered prompt, so
verl's `RLHFDataset` cannot consume them directly. You get:

| file | rows | what |
| --- | ---: | --- |
| `pie_gem5_train.parquet` | 16,452 | training. 16,331 survive the 2,560-token prompt filter |
| `pie_gem5_val_200.parquet` | 200 | a seeded subsample, kept only so the config validates |
| `pie_gem5_test.parquet` | 878 | the held-out evaluation set |

Each row's `ground_truth` carries `usable_case_ids` — the cases every accepted human
submission for that problem passes — and `gem5_src_per_tc_ticks`, the source program's
reference cycle count on each of them. Those reference ticks are the denominator of
every reward, which is why §3's toolchain pins matter.

> The `usable_case_ids` were screened on the machine that built the dataset. Do not
> regenerate them on yours: re-running the correctness screen can flip verdicts on
> programs with signed-overflow or uninitialised-read behaviour, which changes the
> reward's denominator and quietly invalidates any comparison across runs.

---

## 3. The gem5 timing setup

```bash
export PIE_GEM5_HOME=/scratch/$USER/pie_gem5   # ~9 GB
bash scripts/setup_gem5.sh                     # 30-60 min, mostly the gem5 build
bash scripts/verify_setup.sh                   # must print VERIFY OK
```

This builds gem5 v20.1.0.2 with six source patches, an Ubuntu 20.04 x86_64 sysroot,
and the g++-9 that compiles candidate programs.
[`gem5/README.md`](gem5/README.md) explains every pin, and it is worth reading before
you decide to deviate from one: the reward is a ratio of *your* measured cycles to the
dataset's *stored* cycles, so a different gem5 version or a different compiler puts
numerator and denominator on different scales and every speedup comes out wrong.

Two things there are worth knowing even if you never touch the build:

**Why simulate at all.** Wall-clock timing on a shared node varies by more than the
effect being optimized, and the noise correlates with whatever else is running — a
policy can be rewarded for someone else's job finishing. Retired-instruction counts
are stable but nearly uncorrelated with real runtime here, because the costs that
matter are allocation, memory layout and I/O. gem5 gives a deterministic cycle count:
same program, same input, same number, on any host. That determinism is what makes
group-relative advantages meaningful — the spread within a group is signal, not
measurement noise.

**Why it is affordable.** Simulating one of these programs is dominated by *startup*
— process setup, dynamic linking, static initialisers, iostream construction — which
is identical every time. A patch stops the simulation at the program's first read
from stdin, after startup and before any input is consumed, and every test case is
then timed by forking from that point. One simulated startup is shared across all of
them.

Finally, and this is the check that catches a subtly wrong toolchain:

```bash
python3 code_opt/reward/gem5_reward.py --selftest --n 3
```

It grades real dataset rows. The source program, submitted as if it were a rollout,
must score ≈ 1.0 against its own stored reference ticks. If it does not, your
toolchain is not the one that produced the dataset and every reward will be off by a
constant factor that nothing downstream will warn you about.

---

## 4. Configure

Nothing here hardcodes a path. Three variables have no sensible default; the rest are
derived in [`scripts/env.sh`](scripts/env.sh).

```bash
export PIE_TEST_CASE_DIR=/scratch/$USER/pie_test_cases
export PIE_PARQUET_ROOT=/scratch/$USER/pie_parquet
export PIE_GEM5_HOME=/scratch/$USER/pie_gem5
export CODEOPT_RUN_DIR=/scratch/$USER/codeopt_runs        # default: ./runs
export WANDB_PROJECT=tailrl-code-optimization             # console logging works alone
```

| Variable | Required | Default |
| --- | --- | --- |
| `PIE_TEST_CASE_DIR` | **yes** | `<experiment>/data/merged_test_cases` |
| `PIE_PARQUET_ROOT` | **yes** | `<experiment>/data/parquet` |
| `PIE_GEM5_HOME` | **yes** | `<experiment>/gem5/build` |
| `CODEOPT_RUN_DIR` / `CODEOPT_EVAL_DIR` | no | `<experiment>/runs`, `.../runs/eval` |
| `BASE_MODEL` | no | `Qwen/Qwen3-1.7B` |
| `PIE_B` / `PIE_G` / `PIE_MICRO_BS` / `PIE_GPU_MEM_UTIL` | no | `64` / `16` / `2` / `0.5` |
| `PIE_GEM5_REWARD_WORKERS` | no | `nproc - 8` |
| `PIE_GEM5_BIN` / `_ROOTFS` / `_XCROSS_ROOT` / `_PYENV` | no | derived from `PIE_GEM5_HOME` |
| `WANDB_PROJECT` / `WANDB_ENTITY` | no | project default / your default entity |
| `TAILRL_SLURM_*` | no | generic 4-GPU defaults |

W&B credentials come from `~/.netrc` via `wandb login`. **No API key is read from, or
belongs in, this repository.**

---

## 5. Train

```bash
bash scripts/reproduce/train_all.sh --dry-run   # print the three commands
bash scripts/reproduce/train_all.sh             # sequentially, here
bash scripts/slurm/submit_all.sh                # or one job per arm, concurrently
```

Every arm is identical except `algorithm.adv_estimator`:

| `--method` | Estimator |
| --- | --- |
| `tailrl` | TailRL — tail-likelihood, computed as gap-over-survivors |
| `grpo` | Group-relative, z-scored baseline |
| `rloo` | Leave-one-out baseline |

The optimizer setup is deliberately plain, because anything else would muddy the
comparison: no KL penalty, no entropy bonus, no reference model, one PPO epoch with
`train_batch_size == ppo_mini_batch_size`. The importance ratio is therefore
identically 1 and the clip never engages, which makes this exactly single-update
on-policy REINFORCE with a group baseline. Whatever separates the arms is the
estimator.

| | |
| --- | --- |
| model | Qwen3-1.7B |
| batch | 64 prompts × 16 rollouts = **1024 rollouts/step** |
| optimizer | AdamW, lr `1e-6`, `grad_clip=2.0`, 1 epoch |
| reward | gem5 speedup ratio; 0 unless the rollout passes **every** usable case |
| timed cases | **1** per prompt per step, drawn at random, the **same** case for all 16 rollouts of a prompt |
| budget | a rollout more than **3×** slower than the source is stopped and scored 0 |
| schedule | 2570 steps ≈ 10 epochs |
| in-training eval | **none**; evaluation is post-hoc |
| cost | ~400–440 s/step on one 4-GPU node → **8–13 days per arm** |

Two of those deserve a word. Only **one** case is gem5-timed per step because
simulation is the bottleneck and per-case times within a program are nearly uniform,
so one case is a near-lossless estimate of the whole set at a fraction of the cost.
And it is the **same** case for all 16 rollouts of a prompt, so the group's rewards
are comparable and the advantage is not contaminated by case-to-case variation — with
independent cases the estimator would partly be ranking test cases rather than
programs.

Run a single arm by hand, or change its shape, with:

```bash
bash scripts/train.sh --method tailrl --seed 0
bash scripts/train.sh --help          # every knob, and what it costs
```

Runs resume: re-run the same command and each arm continues from its last checkpoint;
a finished arm exits immediately. Progress is one file:

```bash
cat $CODEOPT_RUN_DIR/*/latest_checkpointed_iteration.txt
```

---

## 6. Evaluate

```bash
bash scripts/slurm/submit_eval.sh               # 704 single-GPU array tasks
bash scripts/reproduce/best_at_k_eval.sh --aggregate
```

878 held-out programs × 4096 rollouts × 4 arms (the three trained arms plus the
untrained model as a reference point), every rollout compiled, correctness-checked
and simulated. `n = 4096` is not gratuitous: the unbiased best@k estimator needs
`n ≫ k` for low variance and the headline number is `k = 1024`. Budget a few hundred
GPU-hours.

Sharding is **contiguous in prompt order**, so all 4096 completions for a program land
in one shard and its best@k is exact rather than an average of partial maxima. Shards
are idempotent and self-claiming, so a partially failed array can simply be
resubmitted. Sampling is `temperature 0.6`, `top_p 0.95`, `top_k -1` — identical
across arms, which is the only thing that makes the comparison meaningful.

Both estimators are the exact unbiased ones
([`code_opt/eval/post_hoc_eval.py`](code_opt/eval/post_hoc_eval.py)), not
sample-k-and-take-the-max:

```
pass@k  = 1 - C(n-c, k) / C(n, k)
best@k  = Σ_i  s_(i) · C(n-i, k-1) / C(n, k)        (s sorted descending)
```

`best_at_k` evaluates that sum with an `O(n)` floating-point recurrence rather than
the closed form. At `n=4096, k=1024` the binomial `C(4095, 1023)` is a ~10^1200
integer and `float × C(...)` overflows float64; the recurrence never materialises it.

The run writes:

```
$CODEOPT_EVAL_DIR/
├── metrics/summary.json          # best@k and pass@k per arm
├── metrics/per_problem_<arm>.json  # all 878 programs individually
├── metrics/besteval_test.png     # the curves
└── generations/                  # raw completions for a fixed 64-program subset,
                                  # the same subset for every arm, so they can be
                                  # read side by side
```

Draw the curves from any results file with
`python3 analysis/plot_best_at_k.py --results <summary.json>`.

To evaluate a single checkpoint or a single shard:

```bash
bash scripts/eval_shard.sh --ckpt <run>/actor_ckpts/step_300/huggingface \
                           --arm tailrl_step300 --shard 0 --shards 176
```

---

## 7. Where things land

Everything a run produces goes under `$CODEOPT_RUN_DIR` (default
`experiments/code_optimization/runs/`, gitignored). Nothing is written into the source
tree.

```
$CODEOPT_RUN_DIR/
├── <model>_<method>_g16_bs64_s<seed>/
│   ├── latest_checkpointed_iteration.txt   # the resume pointer, and the "is it done?" check
│   ├── global_step_<N>/                    # rolling full-state checkpoint (1 kept, for resume)
│   └── actor_ckpts/step_<N>/huggingface/   # permanent weights-only snapshots, every 100 steps
├── eval/                                   # see §6
└── logs/
```

The `actor_ckpts/` snapshots are what the evaluation reads, and they exist because
verl keeps only one full-state checkpoint in its rolling buffer — enough to resume,
but it loses the trajectory. A save hook hard-links a weights-only copy out of every
hundredth save; hard links cost nothing and survive verl's rolling delete.

---

## 8. Layout

```
experiments/code_optimization/
├── verl/                     # the vendored fork; estimator dispatch in trainer/ppo/ray_trainer.py
├── code_opt/
│   ├── advantages.py         # TailRL + every baseline estimator, pure torch
│   ├── verl_register.py      # the [G]->[G] estimators, batched for verl
│   ├── train.py              # training entry point
│   ├── build_pie_gem5_parquet.py   # Hub dataset -> verl parquets
│   ├── reward/               # the gem5 speedup x correctness reward
│   ├── measurement/          # compile, run for correctness, time under gem5
│   ├── guards/               # the checkpoint carve-out hook
│   └── eval/                 # sharded pass@k / best_reward@k evaluation
├── gem5/
│   ├── README.md             # ← the timing stack, and why every version is pinned
│   ├── skylake_config/       # the SE-mode CPU model (darchr, BSD-3) + our two runners
│   └── gem5_v20.1.0.2_pie_timing.patch
├── scripts/
│   ├── env.sh                # paths + defaults; sourced by everything
│   ├── download_data.sh, prepare_dataset.sh, setup_gem5.sh, verify_setup.sh
│   ├── train.sh, eval_shard.sh, make_dump_prompts.py
│   ├── reproduce/            # train_all.sh, best_at_k_eval.sh
│   └── slurm/                # generic sbatch wrappers + idempotent submitters
├── analysis/plot_best_at_k.py
├── paper_results/            # the measured evaluation, for comparison against your own
└── tests/                    # CPU-only, no dataset or gem5 build required
```

## Tests

CPU-only, no GPU, no dataset, no gem5 build, no network:

```bash
source scripts/env.sh
pytest tests/ -q
```

They pin the estimator mathematics rather than merely exercising it — in particular
that TailRL reduces *exactly* to binary MaxRL on binary rewards, which is the identity
that fixes the leading `G` in its definition, and that `best_at_k` stays finite at
`n=4096, k=1024`. See [`tests/README.md`](tests/README.md).

---

## Acknowledgements

The RL framework is a fork of [verl](https://github.com/volcengine/verl). The task,
the dataset and the test-case corpus come from
[PIE / Learning Performance-Improving Code Edits](https://pie4perf.com)
(Shypula et al., ICLR 2024) and its
[pie-perf](https://github.com/madaan/pie-perf) release. The CPU model is
[darchr/gem5-skylake-config](https://github.com/darchr/gem5-skylake-config), and the
simulator is [gem5](https://www.gem5.org/). Our thanks to all of them.
