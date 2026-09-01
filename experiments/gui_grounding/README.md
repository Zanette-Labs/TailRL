# GUI Grounding

Clicking the right control in a screenshot: the model sees a full-resolution UI image and an
instruction ("open the layer blending options") and emits a single pixel coordinate, scored by a
*continuous* reward that pays for landing in the target box and decays with distance outside it.
This is the experiment's contrast with the others here — the reward is dense and the action is a
point, so a rollout is either a hit or a graded near-miss rather than a long generation.
Qwen2.5-VL 3B and 7B are trained on GTA1 with **TailRL**, **GRPO**, and **RLOO** — identical in every
respect except `algorithm.adv_estimator` — and evaluated on ScreenSpot-Pro with unbiased pass@k and
best_reward@k.

> **Normalization note.** `compute_tailrl_outcome_advantage` here omits the leading `N` that the
> reference implementations apply, so its advantages are exactly `canonical_TailRL / N`. `N` is
> fixed at 8 for every group in this experiment, making it one global constant on the
> policy-gradient term. The file is kept as it was *run*, because it is what produced
> `paper_results/`. See the docstring in
> [`verl/trainer/core_algos.py`](verl/trainer/core_algos.py).

---

## 1. Install

RL runs through a **vendored fork of [verl](https://github.com/volcengine/verl)** at [`verl/`](verl/)
(via [EasyR1](https://github.com/hiyouga/EasyR1)); the estimators live in
[`verl/trainer/core_algos.py`](verl/trainer/core_algos.py). It ships with the experiment rather than
being installed. You need GPUs — the released runs used one node of 4×GH200 (96 GB) for both the 3B and the
7B line, the 7B differing by a smaller micro-batch.

```bash
git clone https://github.com/Zanette-Labs/TailRL.git
cd TailRL/experiments/gui_grounding

conda create -n tailrl-gui python=3.10 && conda activate tailrl-gui

# torch FIRST, matched to your CUDA runtime (12.4 shown) -- vllm and flash-attn build against it
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Nothing to `pip install -e`. The fork is imported as top-level `verl` off `PYTHONPATH`, which
[`scripts/env.sh`](scripts/env.sh) sets; every script here sources it. Check the wiring:

```bash
source scripts/env.sh
python -c "import verl.trainer.core_algos as c; print(sorted(c.ADV_ESTIMATOR_MAP))"
# ['gae', 'grpo', 'grpo_passk', 'reinforce_plus_plus', 'remax', 'rloo', 'tailrl']

pytest tests -q      # 446 pass, CPU-only; 10 skip without the Qwen processor cached
```

To run inside a container instead, set `TAILRL_GUI_SIF` to an Apptainer image (and
`TAILRL_GUI_BIND` for your scratch mounts) — `env.sh` will wrap every command in it.

---

## 2. Data

Both datasets are public and ungated; no HF token is needed.

| | Dataset | Size |
| --- | --- | --- |
| train | [`Salesforce/grounding_dataset`](https://huggingface.co/datasets/Salesforce/grounding_dataset) (GTA1) | ~34.7 GB raw → 70,528 rows + 16 GB JPEGs |
| eval | [`likaixin/ScreenSpot-Pro`](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) | ~3.2 GB, 1,581 items |

```bash
# export these FIRST or ~54 GB lands in the checkout
export GUI_RAW_DIR=/scratch/$USER/gui_raw       # deletable after conversion (except ScreenSpot-Pro)
export GUI_IMAGE_DIR=/scratch/$USER/gui_images
export GUI_DATA_DIR=/scratch/$USER/gui_data

bash scripts/prepare_data.sh      # downloads both, converts to parquet
```

Expect `kept 70528` and a benign warning that it differs from the nominal 70,688 — 160 rows have
unusable boxes and are dropped. That number is load-bearing: 70,528 / 8 = 8,816 steps per epoch,
and 3 epochs is the 26,448 steps the released runs trained for. ScreenSpot-Pro images are
**symlinked**, not copied, so keep `$GUI_RAW_DIR/ss_pro_raw`.

---

## 3. Run

```bash
source scripts/env.sh

# one arm; the three differ in --method ONLY
bash scripts/train.sh --method tailrl --model 3b --seed 1     # or grpo, rloo
bash scripts/train.sh --method tailrl --dry-run               # print the command, run nothing

# the full comparison, on a cluster
for m in tailrl grpo rloo; do sbatch scripts/slurm/train.sbatch --method $m --model 3b; done
```

Runs resume from their last checkpoint, so re-running the same command continues an interrupted
run. Checkpoints land under `$CKPT_ROOT/<experiment_name>/`.

**Evaluate** — pass@k and best_reward@k on ScreenSpot-Pro, plus the untrained base model as the
control that says whether RL expanded coverage or narrowed it:

```bash
bash scripts/eval.sh --ckpt $CKPT_ROOT/<run>/actor_only/global_step_26448/actor --arm tailrl --shards 4
bash scripts/eval.sh --ckpt Qwen/Qwen2.5-VL-3B-Instruct --arm base --shards 4

python3 analysis/dump_bestk_json.py --set 3B:26448:$EVAL_OUT --base $EVAL_OUT --out results_3b.json
```

`--n` sets samples per item (k must be ≤ n; default 512 supports k ≤ 128). Sampling is
T=0.6 / top_p=0.95 / top_k=−1 and is recorded into every results JSON — every arm in a comparison
must use identical values. Cost scales with **items**, not with n: the prompt is a ~3,100-token
screenshot and the response is a ~13-token coordinate, so prefill dominates.

Measured results are in [`paper_results/`](paper_results/) — both estimators at every k, overall and
split by ScreenSpot-Pro `category` and `ui_type`, with bootstrap CIs over items:
`results_{3b,7b}_final.json` (k ≤ 1024, n = 4096) and `results_ladder_n512.{json,csv}` (all 21
checkpoint rungs × 3 arms × 2 model sizes, plus the base model as step 0).

---

## Acknowledgements

The RL framework is a fork of [verl](https://github.com/volcengine/verl) via
[EasyR1](https://github.com/hiyouga/EasyR1). The reward and training protocol follow
SE-GUI (Yuan et al., NeurIPS 2025); the training corpus is
[GTA1](https://huggingface.co/datasets/Salesforce/grounding_dataset) and the benchmark is
[ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro). Our thanks to all of them.
