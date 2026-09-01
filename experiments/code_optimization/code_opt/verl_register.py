"""Adapter between the per-group estimators in :mod:`code_opt.advantages` and verl.

The fork's ``AdvantageEstimator`` enum, its ``use_critic`` check and its
``compute_advantage`` dispatch are patched directly in
``verl/trainer/ppo/ray_trainer.py``; that dispatch imports :data:`CUSTOM_ESTIMATORS`
from here at call time. Keeping the maths in ``advantages.py`` and the batching here
means the estimators stay unit-testable without Ray, vLLM or a GPU.
"""

from __future__ import annotations

import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch

from code_opt.advantages import (
    binary_maxrl_advantage as _binary_maxrl_1d,
    empirical_cdf_transform as _empirical_cdf,
    pkpo_advantage as _pkpo_1d,
    reinforce_baseline_advantage as _reinforce_baseline_1d,
    rloo_advantage as _rloo_1d,
    tailrl_advantage as _tailrl_1d,
)


def _make_verl_advantage_fn(per_group_fn):
    """Wrap a ``[G] -> [G]`` estimator into the signature verl calls.

    Batch structure: verl flattens ``B`` prompts x ``N`` rollouts into one batch
    dimension, so with ``B=64`` and ``N=16`` every tensor below has 1024 rows.

    ``token_level_rewards``  ``[B*N, seq_len]``  outcome reward, on the EOS position
    ``response_mask``        ``[B*N, seq_len]``  1 on response tokens, 0 on padding
    ``index``                ``[B*N]``           prompt group id per row

    The wrapper reduces each sequence to its scalar reward, groups the scalars by
    prompt, applies ``per_group_fn`` to each group, scatters the advantages back, and
    broadcasts them across the response tokens.
    """

    def verl_advantage_fn(
        token_level_rewards: torch.Tensor,   # [B*N, seq_len]
        response_mask: torch.Tensor,         # [B*N, seq_len]
        index: np.ndarray,                   # [B*N] prompt group ids
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Ablation, off by default so the advantage path is bit-identical when unset:
        # replace each group's rewards by their within-group empirical CDF before the
        # estimator's baseline sees them. Read per call (once per step) so it is
        # monkeypatchable in tests, and env-only rather than threaded through Hydra.
        apply_cdf = os.environ.get("PIE_REWARD_CDF_TRANSFORM", "0") == "1"

        with torch.no_grad():
            scores = (token_level_rewards * response_mask).sum(dim=-1)   # [B*N]

            id2indices: dict = defaultdict(list)
            for i in range(scores.shape[0]):
                id2indices[index[i]].append(i)

            seq_advantages = torch.zeros_like(scores)                    # [B*N]
            for global_indices in id2indices.values():
                idx_t = torch.tensor(global_indices, dtype=torch.long, device=scores.device)
                group_scores = scores[idx_t]                             # [N]
                if apply_cdf:
                    group_scores = _empirical_cdf(group_scores)
                seq_advantages[idx_t] = per_group_fn(group_scores)

            advantages = seq_advantages.unsqueeze(-1) * response_mask    # [B*N, seq_len]
            returns = advantages          # outcome reward: returns == advantages

        return advantages, returns

    return verl_advantage_fn


#: ``algorithm.adv_estimator=<key>`` selects one of these. ``grpo`` and the other
#: stock estimators verl already implements are not re-registered here; ``rloo`` is,
#: because this fork's version drops verl's extra normalization.
CUSTOM_ESTIMATORS = {
    "tailrl": _make_verl_advantage_fn(_tailrl_1d),
    "rloo": _make_verl_advantage_fn(_rloo_1d),
    "reinforce_baseline": _make_verl_advantage_fn(_reinforce_baseline_1d),
    "binary_maxrl": _make_verl_advantage_fn(_binary_maxrl_1d),
    "pkpo_k4": _make_verl_advantage_fn(partial(_pkpo_1d, k_opt=4, variant="loo_minus_one")),
    "pkpo_k8": _make_verl_advantage_fn(partial(_pkpo_1d, k_opt=8, variant="loo_minus_one")),
    "pkpo_k16": _make_verl_advantage_fn(partial(_pkpo_1d, k_opt=16, variant="loo_minus_one")),
}
