# ImageNet Object Localization

Given an image, predict a bounding box around the object. The policy is a
ResNet-50 backbone with four categorical heads — one each for the box centre
`x_c`, `y_c`, its width `w` and its height `h` — over `K = 50` uniformly spaced
bins. A rollout samples one bin per head to form a box, and the reward is that
box's IoU against the ground-truth box: a genuinely continuous scalar in `[0, 1]`
rather than a 0/1 verifier signal.

That makes it a direct testbed for what an advantage estimator should do with a
continuous reward. It is also small enough to run end-to-end on one GPU, and
because the policy is categorical over a finite set of `K⁴ = 6.25M` boxes, the
population-level TailRL objective can be computed *exactly* — so finite-`N`
TailRL can be compared against the very objective it is estimating, not just
against a benchmark number.

You need a machine with an NVIDIA GPU and ~200 GB of disk. Everything below runs
from the repository root.

---

## 1. Install

```bash
git clone https://github.com/Zanette-Labs/TailRL.git
cd TailRL

conda create -n tailrl python=3.10 -y
conda activate tailrl
```

Install PyTorch first, matched to the CUDA runtime on your node — `nvidia-smi`
prints it in the top-right. For CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Then the rest:

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` puts `experiments.imagenet_localization` on the import path;
`export PYTHONPATH=$PWD` from the repo root does the same if you prefer not to
install.

---

## 2. Get the dataset

The **ImageNet Object Localization Challenge** release — ILSVRC CLS-LOC images
plus two solution CSVs, ~166 GB. Accept the competition rules once in a browser,
then:

```bash
pip install kaggle
# kaggle.json goes in ~/.kaggle/ — see https://www.kaggle.com/docs/api
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip -d /path/to/imagenet
```

The layout must be exactly:

```
/path/to/imagenet/
├── LOC_train_solution.csv        # ImageId,PredictionString
├── LOC_val_solution.csv
└── ILSVRC/
    └── Data/
        └── CLS-LOC/
            ├── train/<wnid>/<wnid>_<index>.JPEG
            └── val/<image_id>.JPEG
```

The plain ImageNet classification release will **not** work — it has the class
folders but none of the box annotations, and training fails on the missing
`LOC_*_solution.csv`.

The first training run also pulls pretrained ResNet-50 weights into
`~/.cache/torch/`. Pre-populate that cache if your node is offline.

---

## 3. Set your paths

Export these in the shell you will train from. Only `IMAGENET_DIR` is required;
the rest default to directories inside the repo.

```bash
export IMAGENET_DIR=/path/to/imagenet                  # required

export TAILRL_RESULTS_DIR=/scratch/you/tailrl/results  # default: <repo>/results/imagenet_localization
export TAILRL_FIGURES_DIR=/scratch/you/tailrl/figures  # default: <repo>/figures/imagenet_localization
```

Point `TAILRL_RESULTS_DIR` at a disk with room — every run writes two ~290 MB
checkpoints plus per-epoch milestones, so a full sweep is tens of GB. Put the
exports in your shell rc or job script so they survive a new session.

Now check the install, the dataset and the GPU together:

```bash
pytest experiments/imagenet_localization/tests/ -q      # no GPU or dataset needed, ~35s
python -m experiments.imagenet_localization.pilot --steps 1000   # prints PILOT PASSED
```

---

## 4. Reproduce the paper

Two scripts, one per comparison. Each trains its whole matrix sequentially on
one GPU and **skips arms that already finished**, so an interrupted sweep resumes
by re-running the same command. Nothing assumes a scheduler — submit them however
your cluster likes.

The hyperparameters they train with are in [`config.yaml`](config.yaml), which
`run.py` reads directly. It is commented, and the committed values are the
paper's.

```bash
cd experiments/imagenet_localization

bash scripts/reproduce/supervised_baselines.sh   # 21 runs
bash scripts/reproduce/rl_methods.sh             # 48 runs

bash scripts/make_figures.sh                     # once the runs exist
```

**`supervised_baselines.sh`** — population-level TailRL against anchors that see
the ground-truth box directly.

| `--method` | Arm |
| --- | --- |
| `tailrl_population` | The exact population-level TailRL objective, over all `K⁴` boxes |
| `mse` | MSE on the ground-truth coordinates |
| `l1_iou_match` | L1 |
| `giou` | GIoU |
| `l1_giou` | L1 + GIoU at the DETR weights (`l1_weight: 5`, `giou_weight: 2`) |
| `ordinal_ce` | Ordinal cross-entropy over the `K` bins |
| `cross_entropy` | Plain cross-entropy over the `K` bins |

**`rl_methods.sh`** — every RL method at every rollout budget
`N ∈ {16, 64, 256, 1024}`, so the estimators are compared at matched budgets
throughout.

| `--method` | Arm |
| --- | --- |
| `tailrl` | TailRL |
| `grpo` | Group-relative z-scored baseline |
| `rloo` | Leave-one-out baseline |
| `reinforce` | Mean-centred baseline |

Cost scales with `N` — a step scores `N` rollouts per image — so `N = 1024` is far
slower than `N = 16`. For those, use several GPUs; `batch_size` is per GPU, so
divide it to hold the effective batch at 128:

```bash
torchrun --nproc_per_node=4 -m experiments.imagenet_localization.run \
    --method tailrl --N 1024 --batch_size 32 \
    --output_dir "$TAILRL_RESULTS_DIR/tailrl_K50_N1024_seed42"
```

---

## 5. Where things land

One directory per run, under `$TAILRL_RESULTS_DIR`:

```
$TAILRL_RESULTS_DIR/<run_name>/
├── train.log             # full training stdout (the reproduce scripts tee it here)
├── metrics.json          # per-epoch history, rewritten atomically each epoch
├── best.pt               # checkpoint at best val/iou_greedy
├── last.pt               # most recent epoch
└── checkpoints/
    └── epoch_{1,10,25,50}.pt   # for offline gradient analysis, plus final.pt
```
