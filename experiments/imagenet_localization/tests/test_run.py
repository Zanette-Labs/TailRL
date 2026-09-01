"""Tests for experiments.imagenet_localization.run.

Covers CLI parsing and checkpoint round-trip. No ImageNet dataset required.
Full integration tests (train loop) are skipped — use requires_data marker for
those once the dataset is available.
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch

from experiments.imagenet_localization.models.model import LocalizationPolicy
from experiments.imagenet_localization.run import (
    build_scheduler,
    parse_args,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_namespace(**overrides) -> argparse.Namespace:
    """Return the default parse_args() namespace after applying overrides.

    Uses parse_args() with a minimal argv to keep the test decoupled from
    the actual sys.argv. We monkeypatch sys.argv inside each test that calls
    parse_args directly.
    """
    defaults = dict(
        method="tailrl",
        K=50,
        N=64,
        seed=42,
        epochs=30,
        batch_size=128,
        lr=5e-4,
        warmup_epochs=1,
        grad_clip=10.0,
        eval_every=1,
        N_eval_samples=1024,
        data_dir="/tmp",
        output_dir="/tmp",
        num_workers=8,
        train_subsample=None,
        wandb=False,
        wandb_project="tailrl-imagenet-localization",
        no_pretrained=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# 1. Default values
# ---------------------------------------------------------------------------


def test_parse_args_defaults(monkeypatch):
    """parse_args() must set all defaults correctly from spec §13.

    We supply only the three required arguments (--method, --data_dir,
    --output_dir) and verify that every optional flag is at its spec default.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--method", "tailrl", "--data_dir", "/tmp", "--output_dir", "/tmp"],
    )
    args = parse_args()

    assert args.method == "tailrl"
    assert args.K == 50
    assert args.N == 64
    assert args.seed == 42
    assert args.epochs == 30
    assert args.batch_size == 128
    assert abs(args.lr - 5e-4) < 1e-10
    assert args.warmup_epochs == 1
    assert args.grad_clip == 10.0
    assert args.eval_every == 1
    assert args.N_eval_samples == 1024
    assert args.num_workers == 8
    assert args.train_subsample is None
    assert args.wandb is False
    assert args.wandb_project == "tailrl-imagenet-localization"
    assert not hasattr(args, "include_mse"), (
        "--include_mse belongs to sweep.py, not run.py: on a single run it did "
        "nothing at all, so it was removed rather than left as a silent no-op."
    )
    assert args.no_pretrained is False
    assert args.data_dir == "/tmp"
    assert args.output_dir == "/tmp"


# ---------------------------------------------------------------------------
# 2. Required arguments
# ---------------------------------------------------------------------------


def test_parse_args_required_method_missing(monkeypatch):
    """Omitting --method must cause a SystemExit (argparse error).

    parse_args() must declare --method as required=True.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--data_dir", "/tmp", "--output_dir", "/tmp"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_required_data_dir_missing(monkeypatch):
    """Omitting --data_dir must cause a SystemExit."""
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--method", "tailrl", "--output_dir", "/tmp"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_output_dir_defaults_to_env(monkeypatch, tmp_path):
    """Omitting --output_dir must fall back to $TAILRL_RESULTS_DIR.

    Unlike --data_dir there *is* a sensible repo-relative default here, so the
    flag is optional: exporting TAILRL_RESULTS_DIR is enough to run.
    """
    monkeypatch.setenv("TAILRL_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--method", "tailrl", "--data_dir", "/tmp"],
    )
    args = parse_args()
    assert args.output_dir == str(tmp_path)


def test_parse_args_output_dir_flag_beats_env(monkeypatch, tmp_path):
    """An explicit --output_dir must win over $TAILRL_RESULTS_DIR."""
    monkeypatch.setenv("TAILRL_RESULTS_DIR", str(tmp_path / "from_env"))
    explicit = str(tmp_path / "from_flag")
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--method", "tailrl", "--data_dir", "/tmp",
         "--output_dir", explicit],
    )
    assert parse_args().output_dir == explicit


# ---------------------------------------------------------------------------
# 3. Method choices validation
# ---------------------------------------------------------------------------


def test_method_choices_validated(monkeypatch):
    """An invalid --method value must be rejected with SystemExit.

    Argparse's ``choices`` constraint must be enforced at parse time so that
    downstream code never receives an unknown method string.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--method", "invalid_name",
            "--data_dir", "/tmp",
            "--output_dir", "/tmp",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_all_valid_methods_accepted(monkeypatch):
    """Every documented method choice must be accepted without error."""
    valid_methods = [
        "tailrl", "binary_maxrl", "grpo", "rloo", "reinforce",
        "ordinal_ce", "cross_entropy", "mse", "giou",
    ]
    for method in valid_methods:
        monkeypatch.setattr(
            "sys.argv",
            [
                "run.py",
                "--method", method,
                "--data_dir", "/tmp",
                "--output_dir", "/tmp",
            ],
        )
        args = parse_args()
        assert args.method == method, (
            f"Method '{method}' was not accepted by parse_args()"
        )


# ---------------------------------------------------------------------------
# 4. Checkpoint save / load round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    """save_checkpoint must persist and restore model + optimizer state exactly.

    Uses a tiny LocalizationPolicy(K=10, pretrained=False) so the test runs
    quickly on CPU without downloading pretrained weights.
    """
    K = 10
    model = LocalizationPolicy(K=K, pretrained=False, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Simulate a forward+backward to give the optimizer non-trivial state.
    x = torch.rand(2, 3, 224, 224)
    logits = model(x)
    loss = sum(v.mean() for v in logits.values())
    loss.backward()
    optimizer.step()

    metrics = {"val/iou_greedy": 0.42, "loss_mean": 0.1}
    epoch = 5
    output_dir = str(tmp_path)
    best_val_iou_tracker = [float("-inf")]

    save_checkpoint(
        model, optimizer, epoch, metrics, output_dir,
        best_val_iou_tracker=best_val_iou_tracker,
    )

    # last.pt must exist; best.pt must also exist because iou improved from -inf
    assert os.path.isfile(os.path.join(output_dir, "last.pt")), "last.pt not created"
    assert os.path.isfile(os.path.join(output_dir, "best.pt")), "best.pt not created on first improvement"
    assert best_val_iou_tracker[0] == pytest.approx(0.42)

    # Load and verify state_dict round-trip
    ckpt = torch.load(os.path.join(output_dir, "last.pt"), map_location="cpu")
    assert ckpt["epoch"] == epoch
    assert ckpt["metrics"] == metrics

    model2 = LocalizationPolicy(K=K, pretrained=False, seed=99)
    model2.load_state_dict(ckpt["model_state_dict"])

    # Parameters must be bit-exact after loading.
    for (n1, p1), (n2, p2) in zip(
        sorted(model.state_dict().items()),
        sorted(model2.state_dict().items()),
    ):
        assert torch.equal(p1, p2), (
            f"Parameter '{n1}' differs after checkpoint round-trip"
        )


def test_checkpoint_best_only_updates_on_improvement(tmp_path):
    """best.pt must only be overwritten when val/iou_greedy strictly improves.

    Scenario:
      epoch 1: iou=0.30 -> saves best.pt (first improvement from -inf)
      epoch 2: iou=0.20 -> does NOT overwrite best.pt
      epoch 3: iou=0.40 -> overwrites best.pt (new best)
    """
    K = 10
    model = LocalizationPolicy(K=K, pretrained=False, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    output_dir = str(tmp_path)
    best_val_iou_tracker = [float("-inf")]

    # Epoch 1
    save_checkpoint(
        model, optimizer, 1, {"val/iou_greedy": 0.30}, output_dir,
        best_val_iou_tracker=best_val_iou_tracker,
    )
    best_mtime_after_epoch1 = os.path.getmtime(os.path.join(output_dir, "best.pt"))
    assert best_val_iou_tracker[0] == pytest.approx(0.30)

    # Epoch 2 — worse, best.pt must NOT be updated
    save_checkpoint(
        model, optimizer, 2, {"val/iou_greedy": 0.20}, output_dir,
        best_val_iou_tracker=best_val_iou_tracker,
    )
    assert os.path.getmtime(os.path.join(output_dir, "best.pt")) == best_mtime_after_epoch1, (
        "best.pt was overwritten even though val/iou_greedy did not improve"
    )
    assert best_val_iou_tracker[0] == pytest.approx(0.30)

    # Epoch 3 — better, best.pt must be updated
    save_checkpoint(
        model, optimizer, 3, {"val/iou_greedy": 0.40}, output_dir,
        best_val_iou_tracker=best_val_iou_tracker,
    )
    assert os.path.getmtime(os.path.join(output_dir, "best.pt")) > best_mtime_after_epoch1, (
        "best.pt was NOT updated when val/iou_greedy improved from 0.30 to 0.40"
    )
    assert best_val_iou_tracker[0] == pytest.approx(0.40)

    # Verify the loaded best checkpoint corresponds to epoch 3
    ckpt = torch.load(os.path.join(output_dir, "best.pt"), map_location="cpu")
    assert ckpt["epoch"] == 3, (
        f"best.pt epoch field should be 3, got {ckpt['epoch']}"
    )


# ---------------------------------------------------------------------------
# 5. Checkpoint: no val/iou_greedy means no best.pt
# ---------------------------------------------------------------------------


def test_checkpoint_no_best_when_no_val_iou(tmp_path):
    """save_checkpoint must not create best.pt if val/iou_greedy is absent.

    This happens during the first few epochs when eval is skipped
    (eval_every > 1) or for the MSE regressor whose metrics dict doesn't
    include val/iou_greedy until evaluate_mse_regressor is called.
    """
    K = 10
    model = LocalizationPolicy(K=K, pretrained=False, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    output_dir = str(tmp_path)
    best_val_iou_tracker = [float("-inf")]

    # Metrics without val/iou_greedy
    save_checkpoint(
        model, optimizer, 1, {"loss_mean": 0.5}, output_dir,
        best_val_iou_tracker=best_val_iou_tracker,
    )

    assert os.path.isfile(os.path.join(output_dir, "last.pt")), "last.pt must always be saved"
    assert not os.path.isfile(os.path.join(output_dir, "best.pt")), (
        "best.pt must NOT be created when val/iou_greedy is not in metrics"
    )
    # Tracker must remain at its initial sentinel
    assert best_val_iou_tracker[0] == float("-inf")


# ---------------------------------------------------------------------------
# 5. LR schedule construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epochs,warmup_epochs",
    [(1, 1), (1, 5), (2, 2), (3, 10), (30, 1), (30, 0), (5, 2)],
)
def test_scheduler_never_divides_by_zero(epochs, warmup_epochs):
    """build_scheduler must survive a run no longer than its warmup.

    With warmup_epochs >= epochs the cosine phase would get T_max <= 0, and
    CosineAnnealingLR divides by it — a ZeroDivisionError on the first
    scheduler.step(). The shipped config pairs warmup_epochs=1 with epochs=30,
    but `--epochs 1` is the natural smoke test, so short runs must work too.
    """
    steps_per_epoch = 15
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    scheduler, desc = build_scheduler(
        optimizer, warmup_epochs, epochs, steps_per_epoch,
    )
    assert desc, "build_scheduler must describe the schedule it built"

    for _ in range(epochs * steps_per_epoch):
        optimizer.step()
        scheduler.step()          # must not raise
        lr = optimizer.param_groups[0]["lr"]
        assert lr == lr and lr >= 0.0, f"LR became {lr}"


def test_scheduler_warmup_then_cosine_on_a_normal_run():
    """On the shipped recipe the LR ramps up over warmup, then decays to ~0."""
    steps_per_epoch = 10
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler, _ = build_scheduler(optimizer, 1, 10, steps_per_epoch)

    lrs = []
    for _ in range(10 * steps_per_epoch):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    peak = max(lrs)
    assert abs(peak - 5e-4) < 1e-6, f"peak LR {peak} should reach --lr"
    assert lrs.index(peak) < 2 * steps_per_epoch, "peak should land at end of warmup"
    assert lrs[-1] < peak / 100, f"LR should decay to ~0, ended at {lrs[-1]}"
