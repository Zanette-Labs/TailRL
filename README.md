<div align="center">

# Tail-Likelihood Reinforcement Learning

<p>
Shrinivas Ramasubramanian<sup>1</sup> &nbsp;&nbsp; Daman Arora<sup>1</sup> &nbsp;&nbsp; Fahim Tajwar<sup>1</sup> &nbsp;&nbsp; Guanning Zeng<sup>1</sup><br>
Qingyang Wu<sup>4</sup> &nbsp;&nbsp; Zhongzhu Zhou<sup>4</sup> &nbsp;&nbsp; Chenfeng Xu<sup>4</sup><br>
Haiwen Feng<sup>2,3</sup> &nbsp;&nbsp; Yuda Song<sup>1</sup> &nbsp;&nbsp; Aarti Singh<sup>1</sup> &nbsp;&nbsp; Ruslan Salakhutdinov<sup>1</sup><br>
J. Andrew Bagnell<sup>5,1</sup> &nbsp;&nbsp; Jeff Schneider<sup>1,†</sup> &nbsp;&nbsp; Andrea Zanette<sup>1,†</sup>
</p>

<p>
<sup>1</sup>Carnegie Mellon University &nbsp;&nbsp; <sup>2</sup>UC Berkeley &nbsp;&nbsp; <sup>3</sup>Impossible, Inc. &nbsp;&nbsp; <sup>4</sup>Together AI &nbsp;&nbsp; <sup>5</sup>Aurora Innovation<br>
<sup>†</sup>Joint advising
</p>

<a href="https://zanette-labs.github.io/TailRL-website/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge"></a>
<a href="https://github.com/Zanette-Labs/TailRL">
    <img src="https://img.shields.io/badge/Code-%2300B4D8?style=for-the-badge"></a>

</div>

This is the official implementation of our paper "<strong>Tail-Likelihood Reinforcement Learning</strong>".

## The estimator

**TailRL** is an advantage estimator for policy-gradient RL with *continuous* rewards.
Where GRPO/RLOO/REINFORCE reduce a group of rollouts to a mean-centred baseline, TailRL
integrates over the reward's upper tail: it sorts the group, takes the gap between
consecutive order statistics, and divides each gap by the number of rollouts at or above it.

For a group of `N` rollouts with rewards `r`, sorted ascending as `r₍₁₎ ≤ … ≤ r₍ₙ₎`:

```
gapᵢ        = r₍ᵢ₎ − r₍ᵢ₋₁₎                    (r₍₀₎ ≜ 0)
survivorsᵢ  = N − i + 1                        (the i-th rollout counts itself)
Aᵢ          = N · Σ_{j ≤ i} gapⱼ / survivorsⱼ   (then mean-centred)
```

## Installation

Each experiment is self-contained and pins its own stack, because they genuinely differ —
three of them ship their own [verl](https://github.com/volcengine/verl) fork, and one also
builds a cycle-accurate CPU simulator. **Install the experiment you want to run, into its
own environment.** Install torch first, matched to your CUDA runtime.

```bash
git clone https://github.com/Zanette-Labs/TailRL.git
cd TailRL

conda create -n tailrl python=3.10 && conda activate tailrl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # your CUDA
pip install -r requirements.txt                                    # imagenet_localization
```

| Experiment | Extra requirements | GPUs |
| --- | --- | --- |
| `imagenet_localization` | none beyond the root `requirements.txt` | 1 |
| `text_maze` | [`experiments/text_maze/requirements.txt`](experiments/text_maze/requirements.txt) — adds ray, hydra, transformers | 1 |
| `gui_grounding` | [`experiments/gui_grounding/requirements.txt`](experiments/gui_grounding/requirements.txt) — adds vLLM, flash-attn | multi |
| `code_optimization` | [`experiments/code_optimization/requirements.txt`](experiments/code_optimization/requirements.txt) — adds vLLM, flash-attn, **and a gem5 build** | multi + many CPU cores |

Package versions drift; if the setup above breaks, use your judgement or open an issue.

## Reproducing our experiments

Every experiment has its own README with the full reproduction guide. The commands below
are the short path through each. All of them take `--dry-run` (or a `--dry-run` equivalent)
to print what would run without starting anything, and all of them resume: re-run the same
command and finished work is skipped.

### ImageNet object localization

Bounding-box localization as an RL task, on a policy that is categorical over a finite box
set — so the *population* gradient can be computed exactly and compared against the
finite-`N` estimator.

```bash
cd experiments/imagenet_localization
export IMAGENET_DIR=/path/to/imagenet          # required; the only value with no default

pytest tests/ -q                                # no GPU or dataset needed, ~35s
python -m experiments.imagenet_localization.pilot --steps 1000   # prints PILOT PASSED

bash scripts/reproduce/supervised_baselines.sh  # 21 runs
bash scripts/reproduce/rl_methods.sh            # 48 runs
bash scripts/make_figures.sh                    # once the runs exist
```

Full guide: [`experiments/imagenet_localization/README.md`](experiments/imagenet_localization/README.md)

### Text-maze navigation

17×17 gridworld mazes written as text, with a 3.9M-parameter decoder pretrained from
scratch. The point is the **initialization sweep**: seven starting policies whose
shortest-path success spans `0.83%` down to `0.012%`, then RL from each — which isolates
what an estimator does when high-reward rollouts are attainable but rare.

```bash
cd experiments/text_maze
export HF_CACHE_DIR=$HOME/.cache/tailrl_maze_sft_ckpts
export TAILRL_MAZE_DATA_DIR=/scratch/$USER/tailrl_maze_data   # export FIRST, or ~5 GB lands in the checkout

bash scripts/setup_checkpoints.sh 2450,3000,3250,3350,3400,3450,3550
bash scripts/download_dataset.sh && bash scripts/prepare_dataset.sh
bash scripts/verify_setup.sh 3550               # must print VERIFY OK

bash scripts/reproduce/checkpoint_sweep.sh      # 84 runs — the headline comparison
bash scripts/reproduce/eval_ladder.sh           # Pass@k / Best-of-k, once runs exist
```

Full guide: [`experiments/text_maze/README.md`](experiments/text_maze/README.md)

### GUI grounding

Qwen2.5-VL 3B and 7B click UI targets in screenshots, scored by distance to the target,
and measured on ScreenSpot-Pro.

```bash
cd experiments/gui_grounding
export GUI_DATA_DIR=/scratch/$USER/gui_data     # export FIRST, ~54 GB of data otherwise
export GUI_IMAGE_DIR=/scratch/$USER/gui_images

bash scripts/prepare_data.sh                    # downloads GTA1 + ScreenSpot-Pro, converts to parquet
bash scripts/train.sh --method tailrl --model 3b --seed 1        # or grpo, rloo
bash scripts/eval.sh --ckpt <run>/actor_only/global_step_26448/actor --arm tailrl --shards 4
python3 analysis/dump_bestk_json.py --set 3B:26448:$EVAL_OUT --base $EVAL_OUT --out results_3b.json
```

Full guide: [`experiments/gui_grounding/README.md`](experiments/gui_grounding/README.md)

### Code runtime optimization

Rewriting slow C++ programs to run faster, rewarded by the *measured* speedup: every
rollout is compiled, checked against real test cases, and simulated cycle-by-cycle under
gem5. This one needs a simulator built before anything will run.

```bash
cd experiments/code_optimization
export PIE_TEST_CASE_DIR=/scratch/$USER/pie_test_cases   # ~3.4 GB
export PIE_PARQUET_ROOT=/scratch/$USER/pie_parquet
export PIE_GEM5_HOME=/scratch/$USER/pie_gem5             # ~9 GB

bash scripts/download_data.sh && bash scripts/prepare_dataset.sh
bash scripts/setup_gem5.sh                      # 30-60 min, mostly the gem5 build
bash scripts/verify_setup.sh                    # must print VERIFY OK

bash scripts/reproduce/train_all.sh             # tailrl, grpo, rloo
bash scripts/slurm/submit_eval.sh               # unbiased pass@k / best-of-k at n=4096
```

Full guide: [`experiments/code_optimization/README.md`](experiments/code_optimization/README.md)

## Configuration

Nothing in this repository hardcodes a filesystem path, and no experiment writes into the
checkout by default. Each one documents its own variables; the ones with no sensible
default are `IMAGENET_DIR`, `HF_CACHE_DIR` (text-maze), and `PIE_TEST_CASE_DIR` /
`PIE_PARQUET_ROOT` / `PIE_GEM5_HOME` (code optimization). A `.env` at the repo root works
too — the shell scripts under `experiments/*/scripts/` source it. See
[`.env.example`](.env.example).

W&B credentials come from `~/.netrc` via `wandb login`. **No API key is read from, or
should ever be written into, this repository.**

## Tests

Every suite is CPU-only and needs no dataset. The three experiments that vendor a verl
fork must have their own directory on `PYTHONPATH`, so that `import verl` resolves to
*that* fork and not to another one — run them from the experiment directory, or source its
`scripts/env.sh`, rather than from the repo root:

```bash
pytest experiments/imagenet_localization/tests/ -q          # no fork; runs from anywhere

cd experiments/gui_grounding      && PYTHONPATH=$PWD pytest tests/ -q
cd experiments/code_optimization  && source scripts/env.sh && pytest tests/ -q
python experiments/text_maze/scripts/tests/test_checkpoint_doctor.py
```

## Acknowledgements

The RL framework is a fork of [verl](https://github.com/volcengine/verl) — for
`gui_grounding` via [EasyR1](https://github.com/hiyouga/EasyR1) — and we thank its authors
for an easy codebase to work with. The maze task and its SFT trainer descend from
[MaxRL](https://github.com/tajwarfahim/maxrl). GUI grounding follows the SE-GUI reward and
protocol, trains on [GTA1](https://huggingface.co/datasets/Salesforce/grounding_dataset),
and evaluates on [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro).
Code optimization uses the task, dataset and test cases from
[PIE](https://pie4perf.com) (Shypula et al., ICLR 2024) and its
[pie-perf](https://github.com/madaan/pie-perf) release, the
[darchr](https://github.com/darchr/gem5-skylake-config) Skylake CPU model, and the
[gem5](https://www.gem5.org/) simulator. Our thanks to all of them.

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@misc{ramasubramanian2026tailrl,
  title  = {Tail-Likelihood Reinforcement Learning},
  author = {Ramasubramanian, Shrinivas and Arora, Daman and Tajwar, Fahim
            and Zeng, Guanning and Wu, Qingyang and Zhou, Zhongzhu and Xu, Chenfeng
            and Feng, Haiwen and Song, Yuda and Singh, Aarti and Salakhutdinov, Ruslan
            and Bagnell, J. Andrew and Schneider, Jeff and Zanette, Andrea},
  year   = {2026},
  note   = {https://zanette-labs.github.io/TailRL-website/},
}
```

## Correspondence

[shrinivr@andrew.cmu.edu](mailto:shrinivr@andrew.cmu.edu)
