# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import multiprocessing
import os
from functools import partial

import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class _CustomRewardCallable:
    """Top-level (picklable) replacement for the old get_custom_reward_fn closure.

    PrimeRewardManager ships the reward fn to a spawn multiprocessing.Pool by
    pickling it inside each task tuple; a local closure (and a function pulled from
    an importlib-loaded 'custom_module') is NOT picklable, so the pool.map raised
    and the manager fell back to SEQUENTIAL scoring every step. This object stores
    only the file path / function name / kwargs (all picklable) and lazily (re)loads
    the reward module in whatever process calls it, so parallel scoring works.
    """

    def __init__(self, file_path, function_name, reward_kwargs):
        self.file_path = file_path
        self.function_name = function_name
        self.reward_kwargs = dict(reward_kwargs or {})
        self._raw_fn = None

    def _ensure_loaded(self):
        if self._raw_fn is None:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("custom_module", self.file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["custom_module"] = module
            spec.loader.exec_module(module)
            self._raw_fn = getattr(module, self.function_name)

    def __call__(self, *args, **kwargs):
        self._ensure_loaded()
        return self._raw_fn(*args, **kwargs, **self.reward_kwargs)

    def __getstate__(self):
        # never pickle the loaded fn/module handle; each process reloads by path
        return {
            "file_path": self.file_path,
            "function_name": self.function_name,
            "reward_kwargs": self.reward_kwargs,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._raw_fn = None


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    # Return a picklable callable (not a closure) so PrimeRewardManager's spawn Pool
    # can serialize it and score in parallel instead of falling back to sequential.
    _ = raw_fn  # loaded above only to validate the fn exists; workers reload by path
    return _CustomRewardCallable(file_path, function_name, reward_kwargs)


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """
    from verl.workers.reward_manager import get_reward_manager_cls

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        for additional_key in [
            "accuracy_interval_fractions_per_prompt_global",
            "accuracy_interval_fractions_per_prompt_per_datasource",
            "majority_vote_accuracy_global",
            "majority_vote_accuracy_per_datasource"
        ]:
            if (
                additional_key in reward_result 
                and reward_result[additional_key] is not None
            ):
                reward_extra_infos_dict[
                    additional_key
                ] = reward_result[
                    additional_key
                ]

    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)