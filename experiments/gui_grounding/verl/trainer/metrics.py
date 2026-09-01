# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def aggregate_by_group(group_keys: list, values: list) -> dict[str, tuple[float, int]]:
    """Per-group (mean, n) of a per-sample metric split by an aligned per-sample group key (e.g.
    ScreenSpot-Pro `category` or `ui_type`). Group keys are stringified; inputs must be equal-length."""
    acc: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(group_keys, values):
        acc[str(key)].append(float(value))
    return {key: (float(np.mean(vals)), len(vals)) for key, vals in acc.items()}


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    return {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        "critic/score/std": torch.std(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        "critic/rewards/std": torch.std(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        "critic/advantages/std": torch.std(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **compute_length_metrics(batch),
    }


def compute_group_advantage_metrics(
    batch: DataProto,
    mid_lo: float = 0.2,
    mid_hi: float = 0.8,
    degenerate_std: float = 1e-3,
    bimodal_std_min: float = 0.30,
    bimodal_mid_max: float = 0.25,
    reward_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Within-group (per-prompt) reward/advantage spread + degeneracy diagnostics.

    The estimators act PER GROUP (per prompt), so batch-level std is not the faithful quantity.
    Money plot: group/reward_std_mean vs group/advantage_std_mean -- GRPO pins the advantage std
    ~1 regardless of reward spread; TailRL lets it track the spread. degenerate/bimodal fracs confirm
    the filtered set stays graded during training and does not drift into bimodality.

    `reward_range` is the (lo, hi) span of `overall` for the reward in use. The mid-band and
    bimodality thresholds are SHAPE tests, so they are applied to the rescaled reward
    (r - lo) / (hi - lo) and mean the same thing whatever the reward's units: without this, the
    SE-GUI reward (overall in [0, 2.5]) would report mid_frac ~ 0 purely because 0.2-0.8 sits below
    almost every value it can produce. The default (0, 1) leaves the Gaussian reward untouched.
    degenerate_frac deliberately stays on the RAW scale: "this group is flat" is an absolute claim.
    """
    scores = batch.batch["token_level_rewards"].sum(-1)          # (bs,) scalar reward per rollout
    advantages = batch.batch["advantages"]                       # (bs, resp_len)
    response_mask = batch.batch["response_mask"].bool()          # (bs, resp_len)
    index = batch.non_tensor_batch["uid"]                        # (bs,) group ids
    tok = response_mask.float().sum(-1).clamp(min=1.0)
    adv_scalar = (advantages * response_mask).sum(-1) / tok      # per-rollout advantage (constant per token for outcome estimators)
    lo, hi = float(reward_range[0]), float(reward_range[1])
    span = (hi - lo) if hi > lo else 1.0
    id2r, id2a = defaultdict(list), defaultdict(list)
    for i in range(scores.shape[0]):
        id2r[index[i]].append(scores[i])
        id2a[index[i]].append(adv_scalar[i])
    r_stds, a_stds, mids, degen, bimod, surv = [], [], [], [], [], []
    for idx in id2r:
        r = torch.stack(id2r[idx]).float()
        a = torch.stack(id2a[idx]).float()
        if r.numel() < 2:
            continue
        rs = r.std(unbiased=False).item()
        rn = (r - lo) / span                                     # reward on a comparable 0-1 scale
        rns = rn.std(unbiased=False).item()
        r_stds.append(rs)
        a_stds.append(a.std(unbiased=False).item())
        mid = ((rn >= mid_lo) & (rn <= mid_hi)).float().mean().item()
        mids.append(mid)
        degen.append(1.0 if rs < degenerate_std else 0.0)
        # bimodal == HIGH spread with empty middle (mass at BOTH ends); NOT a tight low/high cluster
        bimod.append(1.0 if (rns >= bimodal_std_min and mid < bimodal_mid_max) else 0.0)
        # survivor right-tail spread: std of the graded winners (rescaled reward > 0.2). On a
        # bimodal-dominant corpus this tracks whether TailRL has continuous signal to grade among the
        # survivors it keeps. Reported in RAW units so it is comparable with reward_std_mean.
        surv_r = r[rn > 0.2]
        surv.append(surv_r.std(unbiased=False).item() if surv_r.numel() >= 2 else 0.0)
    mean = lambda x: float(sum(x) / len(x)) if x else 0.0  # noqa: E731
    # Decile histogram of the rescaled reward over EVERY rollout in the batch. The scalar
    # graded-ness summaries above can all look healthy while the mass is actually piled at the two
    # ends, so this is the shape itself: bin_0 is the dead rollouts, bin_9 the saturated ones, and
    # a reward that grades usefully has mass in between. Values are fractions summing to 1.
    flat = ((scores.detach().float().cpu() - lo) / span).clamp(0.0, 1.0)
    counts = torch.histc(flat, bins=10, min=0.0, max=1.0)
    hist = {f"reward_hist/bin_{i}": (counts[i] / max(1, flat.numel())).item() for i in range(10)}
    return {
        **hist,
        "group/reward_std_mean": mean(r_stds),        # within-group reward spread, avg over groups
        "group/advantage_std_mean": mean(a_stds),     # within-group Std(A), avg over groups
        "group/mid_frac_mean": mean(mids),            # avg mid-band mass (graded-ness during training)
        "group/degenerate_frac": mean(degen),         # fraction of groups with ~zero reward spread
        "group/bimodal_frac": mean(bimod),            # fraction of ~bimodal 0/1 groups (drift check)
        "group/survivor_std_mean": mean(surv),        # avg std of survivors (r>0.2): graded right-tail
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
