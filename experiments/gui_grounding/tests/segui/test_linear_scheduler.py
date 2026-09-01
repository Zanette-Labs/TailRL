"""The `linear` lr schedule must be HuggingFace's, step for step.

SE-GUI's fork never sets lr_scheduler_type, so it inherits the HF/TRL default ("linear"):
warmup to the peak, then a straight line to 0 at num_training_steps. Reproducing their optimizer
protocol means reproducing that exact curve, so this test pins our implementation against
transformers' own -- not against a re-derivation of it.
"""
import pytest
import torch

from verl.utils.torch_functional import get_linear_schedule_with_warmup


def _params():
    return [torch.nn.Parameter(torch.zeros(1))]


def _trace(sched_factory, steps, lr=1e-6):
    opt = torch.optim.SGD(_params(), lr=lr)
    sched = sched_factory(opt)
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


@pytest.mark.parametrize("warmup,total", [(0, 100), (0, 26508), (10, 100), (500, 26508)])
def test_matches_transformers_exactly(warmup, total):
    hf = pytest.importorskip("transformers.optimization")
    ours = _trace(lambda o: get_linear_schedule_with_warmup(o, warmup, total), min(total, 2000))
    theirs = _trace(lambda o: hf.get_linear_schedule_with_warmup(o, warmup, total), min(total, 2000))
    assert ours == theirs, "our linear schedule diverges from HuggingFace's"


def test_decays_to_zero_at_the_horizon():
    total = 1000
    lrs = _trace(lambda o: get_linear_schedule_with_warmup(o, 0, total), total + 5)
    assert lrs[0] == pytest.approx(1e-6), "no warmup -> starts at the peak"
    assert lrs[total // 2] == pytest.approx(0.5e-6, rel=1e-6), "halfway -> half the peak"
    assert lrs[total] == pytest.approx(0.0, abs=1e-15), "reaches 0 at the horizon"
    assert all(x >= 0.0 for x in lrs), "must never go negative past the horizon"
    assert lrs[-1] == pytest.approx(0.0, abs=1e-15)


def test_zero_warmup_never_divides_by_zero():
    lrs = _trace(lambda o: get_linear_schedule_with_warmup(o, 0, 10), 3)
    assert lrs[0] == pytest.approx(1e-6)


def test_registered_in_the_worker_branch():
    """The scheduler is useless if fsdp_workers cannot select it."""
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[2] / "verl/workers/fsdp_workers.py"
    ).read_text()
    assert 'lr_scheduler_type == "linear"' in src, "no branch selects the linear schedule"
    assert "get_linear_schedule_with_warmup" in src, "linear schedule not imported in fsdp_workers"
