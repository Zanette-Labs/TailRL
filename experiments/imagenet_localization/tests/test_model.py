"""Tests for experiments.imagenet_localization.models.model.

All tests use pretrained=False to avoid downloading weights and keep runtime
under 60 seconds on CPU. Batch sizes are kept small (B=2 or B=3) for speed.

The autouse seed fixture in conftest seeds torch before each test, but the
model also accepts an explicit seed for head init — we use seed=42 throughout.
"""

from __future__ import annotations

import pytest
import torch

from experiments.imagenet_localization.models.model import (
    LocalizationPolicy,
    LocalizationRegressor,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HEAD_NAMES = ("x_c", "y_c", "w", "h")
# After the spec_patch_architecture.md change: 64 channels × 7×7 spatial = 3136.
FEATURE_DIM = 64 * 7 * 7
_SEED = 42
_IMG_SHAPE = (3, 224, 224)


def _make_model(K: int = 50, seed: int = _SEED) -> LocalizationPolicy:
    """Construct a CPU model with no pretrained weights."""
    return LocalizationPolicy(K=K, pretrained=False, seed=seed)


def _random_batch(B: int = 2) -> torch.Tensor:
    """Return a (B, 3, 224, 224) float tensor of uniform noise in [-3, 3]."""
    return torch.rand(B, *_IMG_SHAPE) * 6.0 - 3.0


# ---------------------------------------------------------------------------
# 1. Output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [10, 25, 50, 100])
def test_forward_produces_4_heads_of_correct_size(K):
    """forward() must return a dict with exactly 4 heads each of shape (B, K).

    Failure: if any head is missing or has wrong shape the downstream REINFORCE
    loss will broadcast incorrectly or crash.
    """
    model = _make_model(K=K)
    model.eval()
    x = _random_batch(B=2)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, dict), "forward() must return a dict"
    assert set(out.keys()) == set(HEAD_NAMES), (
        f"Expected keys {HEAD_NAMES}, got {set(out.keys())}"
    )
    for h in HEAD_NAMES:
        assert out[h].shape == (2, K), (
            f"Head '{h}': expected shape (2, {K}), got {out[h].shape}"
        )


def test_batch_size_preserved():
    """The first dimension of every head output must equal the input batch size.

    Failure: if the backbone collapses the batch dimension, all images in a
    minibatch would share identical predictions.
    """
    K = 50
    model = _make_model(K=K)
    model.eval()
    x = _random_batch(B=3)
    with torch.no_grad():
        out = model(x)

    for h in HEAD_NAMES:
        assert out[h].shape[0] == 3, (
            f"Head '{h}': expected batch dim 3, got {out[h].shape[0]}"
        )


# ---------------------------------------------------------------------------
# 2. Independence of heads
# ---------------------------------------------------------------------------


def test_heads_produce_different_outputs():
    """The 4 heads must produce different logit tensors on the same input.

    If heads share weights or are aliased, the RL policy collapses: all
    coordinates would be predicted identically, making the agent degenerate.
    We check every pairwise combination.
    """
    model = _make_model(K=50, seed=_SEED)
    model.eval()
    x = _random_batch(B=2)
    with torch.no_grad():
        out = model(x)

    head_list = list(HEAD_NAMES)
    for i in range(len(head_list)):
        for j in range(i + 1, len(head_list)):
            h1, h2 = head_list[i], head_list[j]
            assert not torch.equal(out[h1], out[h2]), (
                f"Heads '{h1}' and '{h2}' produced identical outputs — "
                "they may be aliased or share weights."
            )


def test_head_parameters_are_separate_tensors():
    """All 4 heads must have distinct weight and bias tensors in memory.

    Aliased parameters would cause double-counting gradients and silent bugs
    during backward.  We check that all 8 parameter tensors (4 weights + 4
    biases) have unique Python object ids.
    """
    model = _make_model(K=50)

    weight_ids = {id(model.heads[h].weight) for h in HEAD_NAMES}
    bias_ids = {id(model.heads[h].bias) for h in HEAD_NAMES}

    assert len(weight_ids) == 4, (
        f"Expected 4 distinct weight tensors, got {len(weight_ids)}"
    )
    assert len(bias_ids) == 4, (
        f"Expected 4 distinct bias tensors, got {len(bias_ids)}"
    )

    # All 8 param tensors (weights + biases) must be pairwise distinct.
    all_ids = weight_ids | bias_ids
    assert len(all_ids) == 8, (
        f"Expected 8 distinct parameter tensors total, got {len(all_ids)}"
    )


# ---------------------------------------------------------------------------
# 3. Numerical stability
# ---------------------------------------------------------------------------


def test_no_nan_in_forward():
    """forward() on random noise inputs must produce no NaN logits.

    NaN in logits propagates through softmax/log-softmax to the loss and
    poisons the entire training run.
    """
    model = _make_model(K=50)
    model.eval()

    for trial in range(3):
        # Different noise each trial (global seed is set per-test but we vary
        # the input by using different manual seeds for the data tensor).
        torch.manual_seed(trial + 1)
        x = torch.rand(2, *_IMG_SHAPE) * 6.0 - 3.0
        with torch.no_grad():
            out = model(x)
        for h in HEAD_NAMES:
            assert not torch.isnan(out[h]).any(), (
                f"NaN detected in head '{h}' on trial {trial}"
            )


def test_no_nan_in_backward():
    """Backward pass on the sum of head means must not produce NaN gradients.

    NaN gradients indicate numerical instability (e.g., exploding activations
    or a broken computation graph) that would make training fail silently.
    """
    model = _make_model(K=50)
    model.train()
    x = _random_batch(B=2)

    out = model(x)
    # Scalar loss: sum of per-head mean logits.
    loss = sum(out[h].mean() for h in HEAD_NAMES)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), (
                f"NaN gradient detected in parameter '{name}'"
            )


# ---------------------------------------------------------------------------
# 4. Gradient coverage
# ---------------------------------------------------------------------------


def test_all_params_get_gradient():
    """After a backward pass, > 90% of trainable params must have nonzero grad.

    A low percentage would indicate that most of the network is disconnected
    from the loss — e.g., if features are detached accidentally.

    Note: BatchNorm running stats (running_mean / running_var) are not
    trainable parameters and do not appear in named_parameters(requires_grad).
    Some BN weight/bias pairs may legitimately receive zero gradient for
    a single batch (e.g., when the layer happens to be in eval mode or the
    input variance is tiny) — hence the 90% threshold, not 100%.
    """
    model = _make_model(K=50)
    model.train()
    x = _random_batch(B=2)

    out = model(x)
    loss = sum(out[h].mean() for h in HEAD_NAMES)
    loss.backward()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    nonzero_grad_count = sum(
        1
        for _, p in trainable
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    total = len(trainable)
    fraction = nonzero_grad_count / total if total > 0 else 0.0

    assert fraction >= 0.90, (
        f"Only {nonzero_grad_count}/{total} ({100*fraction:.1f}%) trainable "
        "params have nonzero gradient — expected >= 90%."
    )


# ---------------------------------------------------------------------------
# 5. freeze / unfreeze
# ---------------------------------------------------------------------------


def test_freeze_stops_backbone_gradient():
    """After freeze_backbone(), backbone params must receive no gradient.

    During the head-only warmup phase we freeze the backbone to avoid
    destroying pretrained features before the heads stabilise.  If freeze
    doesn't work, the backbone gets noisy gradients from untrained heads.
    """
    model = _make_model(K=50)
    model.train()
    model.freeze_backbone()

    x = _random_batch(B=2)
    out = model(x)
    loss = sum(out[h].mean() for h in HEAD_NAMES)
    loss.backward()

    for name, param in model.features.named_parameters():
        # After freeze, requires_grad=False so grad should be None.
        assert param.grad is None, (
            f"Backbone parameter '{name}' has a gradient after freeze_backbone()"
        )


def test_unfreeze_restores_backbone_gradient():
    """After freeze then unfreeze, at least one backbone param must get a grad.

    Verifies that unfreeze_backbone() is a true inverse of freeze_backbone().
    """
    model = _make_model(K=50)
    model.train()
    model.freeze_backbone()
    model.unfreeze_backbone()

    x = _random_batch(B=2)
    out = model(x)
    loss = sum(out[h].mean() for h in HEAD_NAMES)
    loss.backward()

    has_nonzero_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.features.parameters()
    )
    assert has_nonzero_grad, (
        "No backbone parameter received a nonzero gradient after unfreeze_backbone()"
    )


def test_wrong_input_resolution_raises():
    """The Linear head dim is hardcoded to 64 × 7 × 7 = 3136 (i.e., 224x224 input).
    A 256x256 input produces an 8x8 spatial map → flatten = 64*8*8 = 4096, which
    must raise RuntimeError on the matmul. This test pins the input-resolution
    contract so a silent shape mismatch can't sneak in mid-training.
    """
    import pytest as _pytest
    model = _make_model(K=10)
    x = torch.randn(1, 3, 256, 256)
    with _pytest.raises(RuntimeError):
        model(x)


def test_freeze_backbone_does_not_freeze_spatial_reduce():
    """freeze_backbone() must only freeze self.features. The 1x1 conv bottleneck
    in self.spatial_reduce — which is the entire point of the architecture
    patch — must remain trainable, otherwise warmup-phase localization signal
    is lost again.
    """
    model = _make_model(K=10)
    model.freeze_backbone()
    for p in model.spatial_reduce.parameters():
        assert p.requires_grad is True, (
            "freeze_backbone() unexpectedly froze a spatial_reduce parameter"
        )


def test_spatial_reduce_gets_gradient_when_backbone_frozen():
    """spatial_reduce must train during the warmup phase even when the backbone
    is frozen. This is critical: without it, heads-only training has no
    spatially-informed signal and plateaus at the class-mean-box ceiling.

    After spec_patch_architecture.md, the 1x1 conv bottleneck sits between
    self.features (frozen during warmup) and self.heads, and must remain
    trainable so heads-only warmup actually produces localization signal.
    """
    model = _make_model(K=10)
    model.freeze_backbone()
    model.train()

    x = _random_batch(B=2)
    out = model(x)
    loss = sum(out[h].sum() for h in HEAD_NAMES)
    loss.backward()

    conv = model.spatial_reduce[0]   # nn.Conv2d
    assert conv.weight.grad is not None, (
        "spatial_reduce conv has no gradient — it appears to have been frozen "
        "along with the backbone, defeating the architecture patch."
    )
    assert conv.weight.grad.abs().sum().item() > 0.0, (
        "spatial_reduce conv received zero gradient under frozen backbone — "
        "warmup-phase training would be useless."
    )

    # Heads should also still receive gradient.
    for h in HEAD_NAMES:
        head_grad = model.heads[h].weight.grad
        assert head_grad is not None and head_grad.abs().sum().item() > 0.0, (
            f"Head '{h}' received zero gradient under frozen backbone."
        )


# ---------------------------------------------------------------------------
# 6. LocalizationRegressor
# ---------------------------------------------------------------------------


def test_regressor_forward_in_unit_range():
    """LocalizationRegressor output must lie in [0, 1]^4 (sigmoid guarantee).

    If the sigmoid is absent or applied incorrectly, the MSE baseline would
    produce unbounded predictions that cannot be interpreted as image
    coordinates in [0, 1].
    """
    model = LocalizationRegressor(pretrained=False, seed=_SEED)
    model.eval()
    x = _random_batch(B=4)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 4), f"Expected shape (4, 4), got {out.shape}"
    assert out.min().item() >= 0.0, (
        f"Regressor output has value below 0: {out.min().item()}"
    )
    assert out.max().item() <= 1.0, (
        f"Regressor output has value above 1: {out.max().item()}"
    )
