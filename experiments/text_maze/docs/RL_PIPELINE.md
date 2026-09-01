# Text-Maze RL pipeline — operations guide

How a run is put together, how a campaign is driven, how results come out — and,
most importantly, **the footguns in §6**. Each of those silently corrupted a
result at least once. Read §6 before you touch the telemetry or launch a sweep at
a group size or batch size you have not run before.

The [README](../README.md) is the reproduction guide; this is the reference.

---

## 1. What a run is

One **run** = one RL fine-tune of one SFT checkpoint, identified by its
experiment name:

```
textmaze_<suite>_<method>_<reward>_ckpt<STEP>_bs<BS>_G<N>_1gpu_0_seed<S>
```

- **method** — the advantage estimator, the only thing that varies between arms:
  `tailrl` | `grpo` | `rloo` | `pkpo` (which resolves to `pkpo_continuous` for
  continuous rewards). Set via `algorithm.adv_estimator`. `maxrl` and `reinforce`
  are also registered; `maxrl` is binary MaxRL, which TailRL reduces to exactly
  when the reward is binary.
- **ckpt STEP** — the SFT initialization. The ladder spans shortest-path success
  rates from `ckpt-2450` (0.0122 %) to `ckpt-3550` (0.8270 %). Downloaded into
  `$HF_CACHE_DIR/ckpt-<STEP>`.
- **N** (`rollout.n`) — rollouts sampled per prompt. Responses per step = `bs × N`.
- **reward** — `composite_v2`
  ([`src/maze_composite_v2_reward.py`](../src/maze_composite_v2_reward.py)),
  with `reward_transform=raw` (a no-op).

Training runs `total_training_steps` (5001), validating every `test_freq`, at
step 0 (`val_before_train`) and at the last step. Validation draws
`val_kwargs.n` responses per prompt over the 1000 held-out mazes and logs
`mean@N` plus `best@{2,4,…,N}` for three metric families: `is_shortest` (solved
via a shortest path — the headline), `goal_reached`, and `reward`.

Everything runs on **one GPU** via FSDP; the rollout backend is `hf`
(HFRollout), not vLLM.

The name is a contract, not a label: the checkpoint directory is
`$TAILRL_MAZE_RUN_DIR/checkpoints/$WANDB_PROJECT/<experiment_name>/`, and both
resume and evaluation resolve it from the configuration. Change a flag, get a
different run.

---

## 2. The launcher — `scripts/train.sh`

Single-run entry point. Resolves the experiment name and every Hydra override,
then calls `python -m verl.trainer.main_ppo`.

| flag | meaning |
|---|---|
| `--method` | advantage estimator → `algorithm.adv_estimator` |
| `--ckpt-step` / `--pass-rate-index` | which SFT checkpoint |
| `--seed`, `--n-rollouts` (N), `--batch-size`, `--total-steps` | the obvious knobs |
| `--reward`, `--reward-transform` | reward function + transform |
| `--pass-k` | `k_opt` for PKPO; guarded (see below) |
| `--val-n`, `--test-freq`, `--save-freq` | validation and checkpoint cadence |
| `--max-response-length`, `--extra-eos-token-ids` | generation length + stop tokens |
| `--val-only`, `--wandb-resume-id`, `--resume-global-steps` | re-eval mode (§5) |
| `--dry-run` | resolve and print the configuration; start nothing |

Fixed across all arms so the method comparison is clean: `lr=1e-4`,
`use_kl_loss=False`, `kl_coef=0.0`, `actor.dtype=float16`,
`ppo_mini_batch_size = batch_size`, `max_prompt_length=320`,
`gpu_memory_utilization=0.7`, validation sampling `do_sample=True,
temperature=1.0`, `max_actor_ckpt_to_keep=3`.

**PKPO guard.** `pass_k > n_rollouts` is invalid — `max@k` is undefined with
fewer than `k` samples per prompt; the continuous estimator asserts, and the
binary one used to emit all-NaN advantages, which under `float16` makes the grad
scaler skip every step forever. `pass_k == n_rollouts` is valid but *degenerate*:
`max(g_1..g_n)` is invariant to every non-maximal sample, so only the argmax
gets a nonzero advantage and `(n−1)/n` of the batch contributes no gradient. Both
flags default to 16, so `--method pkpo` with nothing else set lands exactly
there — hence the guard, and `PKPO_ALLOW_DEGENERATE=1` to run `k=n` deliberately.

`WANDB_PROJECT` is read from the environment and determines where checkpoints are
written. If it is not exported consistently, the write path and the watch path
diverge and every run looks unfinished forever.

---

## 3. Running a campaign

[`scripts/reproduce/`](../scripts/reproduce/) walks each matrix sequentially and
skips finished arms, so re-running a script resumes the campaign.
[`scripts/slurm/submit_sweep.sh`](../scripts/slurm/submit_sweep.sh) fans the same
matrices out, one job per arm, and is idempotent for the same reason — an arm
that is finished or already queued is not resubmitted. Run it on a timer and the
queue converges.

**Resume.** `resume_mode=auto` restores model, optimizer, RNG and LR scheduler
from the newest `global_step_<N>/actor/` in `default_local_dir`. A resumed job
prints `Loaded model from .../global_step_<N>/actor/...` — that line is the only
reliable proof it happened, and §6 explains why its absence is a hard failure
rather than a warning.

[`scripts/slurm/sbatch_train.sh`](../scripts/slurm/sbatch_train.sh) traps `USR1`
(sent ten minutes before the time limit) and calls `scontrol requeue`, so
preemption and walltime cost restarts rather than progress. It also holds a
duplicate-run lock keyed on the argument set: two live jobs with the same
configuration would interleave into one checkpoint directory and one W&B run, so
the second aborts. Training runs in the background with an explicit `wait` —
`exec`ing it would block the shell and the `USR1` trap would never fire.

---

## 4. Checkpoint hygiene — `checkpoint_doctor.py`

Repairs SFT checkpoints saved by a newer `transformers` than the runtime's 4.5x:

- Flattens 5.x `rope_parameters` → top-level `rope_theta`. Missing this gives a
  100×-too-small RoPE base, scrambled positional encoding, and **0 %**
  goal-reaching from a model that still emits well-formed paths.
- Fixes `tokenizer_class` (`TokenizersBackend` → `PreTrainedTokenizerFast`).
- **DONE-as-EOS.** The maze `DONE` token (id 7) must be in
  `generation_config.eos_token_id`. The reference checkpoints shipped `[2,7]`;
  ours shipped `2`, so generation ran to the 180-token ceiling and post-goal
  meandering tokens got reinforced. The doctor sets `[2,7]`; runs also pass
  `--extra-eos-token-ids 7`, wired into `hf_rollout.py`.

Every launcher runs it as a preflight and aborts if the checkpoint would still
load broken. Tests: `scripts/tests/test_checkpoint_doctor.py`,
`scripts/tests/test_eos_done_smoke.py`.

---

## 5. Getting results out

**W&B's final-step metrics are not reliable** (footgun #1). The authoritative
numbers come from re-evaluating each finished checkpoint offline:

1. [`scripts/reproduce/eval_ladder.sh`](../scripts/reproduce/eval_ladder.sh) —
   for each finished run, a `--val-only` job: load the final checkpoint, run one
   validation at a large sampling budget, exit. It hard-fails unless the log
   proves the resume happened.
2. [`scripts/eval_logs_to_json.py`](../scripts/eval_logs_to_json.py) parses the
   console metrics into one JSON, adding an `_aggregate` block with mean, sample
   std (ddof=1) and per-seed values for each configuration.

`scripts/reeval_and_log.py` additionally pushes the parsed metrics back into an
existing W&B run id — `train.sh --val-only --wandb-resume-id <id>` calls it.

If you write your own log parser, take the two rules from footguns #2 and #3
with you: the **global** step is the one in the console `step:<N>` lines (W&B's
`_step` is per-incarnation), and the text must be de-wrapped before matching.

---

## 6. Footguns

**#1 — W&B drops the final validation.** W&B commits a history row only once a
*later* step is logged or `run.finish()` is called, and verl's only `finish()`
lived in a GC finalizer that CPython does not guarantee at exit and that never
runs on SIGKILL. The end-of-training validation is the one log with no later step
to force it, so it was lost for about half of one campaign. **Fixed:**
`Tracking.finish()` is public and idempotent, and `fit()` calls it before both
return paths. When backfilling, use `Api().run().summary.update()` — a
server-side patch. `wandb.log(step=5000)` is *silently dropped* if step 5000
already exists in history. Verifying "the key exists" is not enough; verify the
value is fresh against a known reference.

**#2 — verl opens a new W&B run per resume, with a local step counter.** One
experiment becomes many W&B runs, and their `_step` is not the global training
step. Never trust `_step` for a resumed run. The global step is in the console
`step:<N>` lines; that is what the log parsers anchor on.

**#3 — verl pretty-prints metrics wrapped across lines**, with a Ray
`(TaskRunner pid=N)` prefix landing *between* a key and its value. A
line-anchored regex silently parses ~153 of ~182 metrics and drops
`is_shortest` — the headline. **Always de-wrap** (strip ANSI, strip pid
prefixes, collapse newlines) before matching, and treat a missing required
metric as a hard failure rather than a silent success. See `parse_metrics()` in
`scripts/reeval_and_log.py`.

**#4 — an evaluation that cannot find its checkpoint measures the base model and
exits 0.** `--val-only` resumes via `resume_mode=auto` from
`checkpoints/$WANDB_PROJECT/<experiment>/`. If `WANDB_PROJECT` at eval time
differs from training, that directory does not exist, verl falls back to the base
SFT checkpoint, and you get plausible, wrong numbers with a zero exit code. This
invalidated a whole evaluation batch. The tell was arithmetic: `best@1024`
(0.037–0.613) came out *below* `best@64` (0.906–0.971), which is impossible.
`eval_ladder.sh` now requires `Loaded model from .../global_step_N` in the log.

**#5 — `save_freq` must fit inside a preemption window, so it scales with N.**
No checkpoint is written until `save_freq` steps complete. At a flat
`save_freq=250` and ~96 s/step, an `N=256` run needs 6.7 h *uninterrupted* to
write anything — so on a queue that preempts hourly, every such run restarted
from step 0 forever while looking busy. Save every ~5–25 steps at large `N`. The
cost is negligible: `max_actor_ckpt_to_keep=3` bounds the payload to ~138 MB per
run for this tiny model.

**#6 — the generation chunk must divide the generation batch.** `hf_rollout`
computes `num_chunks = bs // max(rollout.micro_batch_size // N, 1)` and
`DataProto.chunk` asserts an *equal* split. With `micro=4000` and `N=256`:
`4000//256 = 15`, `256//15 = 17` chunks, `256 % 17 ≠ 0` → hard crash
mid-generation. `micro=8000` gives 8 chunks and is fine. Separately,
`ppo_micro_batch_size_per_gpu` must divide `batch_size × N` — 4096 asserts at
`N=4` (256·4 = 1024). `scripts/reproduce/_common.sh` computes `min(256·N, 4096)`.

**#7 — thundering-herd startup.** Submitting a whole sweep at once lands many
cold starts on the same few nodes; simultaneous container mounts, checkpoint
reads and Ray init serialize into a multi-minute I/O stall with 0 % GPU and
frozen logs. This looks exactly like a deadlock and is not — the tell is that all
co-located jobs' logs jump forward *together* once it clears. **Diagnose a stall
by whether logs advance over time, not by an instantaneous GPU-utilization
snapshot.** verl's progress output goes to stdout, not stderr.

---

## 7. Quick reference

```bash
# one run
bash scripts/train.sh --method tailrl --ckpt-step 2450 --seed 0 \
  --gpu-ids 0 --n-rollouts 16 --batch-size 256 --total-steps 5001 \
  --extra-eos-token-ids 7 --val-n 64 --test-freq 1000 --save-freq 250

# a campaign, locally or on SLURM (both idempotent, both resume)
bash scripts/reproduce/checkpoint_sweep.sh
bash scripts/slurm/submit_sweep.sh checkpoint_sweep --dry-run

# evaluate finished runs and compile one JSON
bash scripts/reproduce/eval_ladder.sh

# tests (no GPU)
python scripts/tests/test_checkpoint_doctor.py
python scripts/tests/test_reeval_and_log.py
```

Credentials come from `~/.netrc` via `wandb login`. No API key is read from, or
belongs in, this repository.
