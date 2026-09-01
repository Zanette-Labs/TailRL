"""Pin VF.entropy_from_logits (the true token entropy logged during training, never in the loss)."""
import numpy as np
import pytest
import torch

from verl.utils.torch_functional import entropy_from_logits


@pytest.mark.parametrize("shape", [(7, 13), (3, 5, 257), (1300, 97)])
def test_matches_categorical_entropy(shape):
    torch.manual_seed(0)
    logits = torch.randn(*shape, dtype=torch.float32)
    ref = torch.distributions.Categorical(logits=logits).entropy()
    out = entropy_from_logits(logits)
    assert out.shape == logits.shape[:-1]
    assert torch.allclose(out, ref, atol=1e-5)


def test_chunking_is_invariant():
    torch.manual_seed(1)
    logits = torch.randn(1300, 97)
    a = entropy_from_logits(logits, chunk_size=512)
    b = entropy_from_logits(logits, chunk_size=64)
    c = entropy_from_logits(logits, chunk_size=100000)
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)


def test_uniform_is_log_vocab():
    V = 50
    out = entropy_from_logits(torch.zeros(4, V))           # uniform -> H = log V
    assert torch.allclose(out, torch.full((4,), float(np.log(V))), atol=1e-5)


def test_peaked_is_near_zero():
    logits = torch.full((3, 20), -1e4)
    logits[:, 0] = 1e4                                       # near one-hot -> H ~ 0
    out = entropy_from_logits(logits)
    assert torch.all(out >= 0) and torch.all(out < 1e-3)


def test_fp32_output_from_bf16_input():
    out = entropy_from_logits(torch.randn(8, 33, dtype=torch.bfloat16))
    assert out.dtype == torch.float32 and out.shape == (8,)
