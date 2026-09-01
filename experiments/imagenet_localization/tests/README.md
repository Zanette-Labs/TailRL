# Tests — ImageNet localization

384 tests, pure CPU. No GPU and no dataset required: the tests that need real
ImageNet skip automatically when `IMAGENET_DIR` is unset, via the `requires_data`
marker in `conftest.py`.

Run from the **repository root**:

```bash
pytest experiments/imagenet_localization/tests/ -q          # all
pytest experiments/imagenet_localization/tests/ -v          # per-test names
pytest experiments/imagenet_localization/tests/test_advantages.py -v   # one file
```

The whole suite takes about 35 seconds on 8 CPU threads. The one slow test is
`test_training_step.py::test_tailrl_overfits_single_image` (~15s), which runs 800
real training steps. `-k "not overfits"` drops it — note that also deselects the
two `test_supervised_epoch_overfits_a_fixed_batch` cases.

To exercise the data-dependent tests, export `IMAGENET_DIR` first:

```bash
IMAGENET_DIR=/path/to/imagenet pytest experiments/imagenet_localization/tests/ -q
```

## What each file covers

| File | Covers |
| --- | --- |
| `test_advantages.py` | Every estimator in `ADVANTAGE_FNS`: rank properties, invariances, hand-computed values. Includes the exact TailRL ≡ binary MaxRL reduction on binary rewards |
| `test_tailrl_mask_fix.py` | The `TAILRL_SURVIVAL_EPS` survival-floor mask and its boundary |
| `test_reward_invariance.py` | Which estimators are invariant to affine / monotone reward transforms |
| `test_binary_reward.py` | The `binary_0.5` / `binary_0.75` reward transforms |
| `test_pkpo.py` | The PKPO estimator and its `k` fallback |
| `test_losses.py` | Supervised losses: ordinal CE, cross-entropy, `tailrl_population` |
| `test_giou.py` | The GIoU / L1+GIoU family and its GT-matching rules |
| `test_iou.py`, `test_iou_vectorized.py` | IoU maths and the batched implementation |
| `test_data.py` | `ImageNetLocDataset`, `PredictionString` parsing, box normalization, collate |
| `test_model.py` | `LocalizationPolicy` / `LocalizationRegressor`, head init, freeze/unfreeze |
| `test_evaluate.py` | Validation metrics, greedy and sampled |
| `test_training_step.py` | One training step for every method; gradient flow on all four heads |
| `test_metrics_online.py` | `train.py`'s streaming running-metrics accumulator |
| `test_run.py` | `run.py`'s CLI surface, config precedence, checkpoint round-trip |
| `test_sweep.py` | SLURM script generation, stage grouping, job-name hashing |
| `test_pilot.py` | The single-image overfit smoke test's CLI |
| `test_gradient_analysis.py` | Offline gradient / cosine analysis |
| `test_bestk.py` | The unbiased best@k estimator |

## Conventions

- **No network, no dataset, no GPU** in the default path. Model tests construct
  `LocalizationPolicy(pretrained=False)` so nothing is downloaded.
- Tests that assert on a flag default read it through `parse_args()` rather than
  hardcoding the number, so editing `config.yaml` does not silently break them —
  except where a test is deliberately pinning the shipped default.
- `conftest.py` owns the `requires_data` marker and the `IMAGENET_DIR` lookup;
  add data-dependent tests behind that marker rather than skipping by hand.
