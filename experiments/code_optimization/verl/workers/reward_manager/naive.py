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

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_reward_workers = int(os.environ.get("VERL_REWARD_WORKERS", 128))

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # ── Pre-extract all inputs (sequential, fast) ──
        reward_inputs = []
        valid_response_lengths = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # skip_special_tokens=False to preserve <think></think> markers
            # for reasoning models (Qwen3, DeepSeek-R1); extract_code strips them
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            reward_inputs.append({
                "data_source": data_source,
                "solution_str": response_str,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "prompt_str": prompt_str,
            })
            valid_response_lengths.append(int(valid_response_length))

        # ── Compute rewards in parallel ──
        # Try test-case-level parallelism first (much better CPU utilization).
        # Falls back to completion-level parallelism if batch function unavailable.
        compute_scores_batch = getattr(self, '_compute_scores_batch', None)
        if compute_scores_batch is not None:
            scores = compute_scores_batch(reward_inputs, num_workers=self.num_reward_workers)
        else:
            def _score(kwargs):
                return self.compute_score(
                    data_source=kwargs["data_source"],
                    solution_str=kwargs["solution_str"],
                    ground_truth=kwargs["ground_truth"],
                    extra_info=kwargs["extra_info"],
                )

            with ThreadPoolExecutor(max_workers=self.num_reward_workers) as pool:
                scores = list(pool.map(_score, reward_inputs))

        # ── Collect results ──
        for i, score in enumerate(scores):
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_lengths[i] - 1] = reward

            data_source = reward_inputs[i]["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", reward_inputs[i]["prompt_str"])
                print("[response]", reward_inputs[i]["solution_str"])
                print("[ground_truth]", reward_inputs[i]["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
