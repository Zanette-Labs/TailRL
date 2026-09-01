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
"""
Reward config
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from ...utils.py_functional import get_abs_path


@dataclass
class RewardConfig:
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    reward_range: Tuple[float, float] = (0.0, 1.0)
    """(lo, hi) span of the reward function's `overall`. DIAGNOSTICS ONLY -- nothing rescales the
    reward itself, and the estimators never see this. It puts the shape tests in
    compute_group_advantage_metrics (mid-band mass, bimodality) on a comparable 0-1 scale so they
    mean the same thing across rewards with different units: SE-GUI's additive point reward spans
    [0, 2.5] (or [0.5, 2.5] once the format term is earned), where the default 0.2-0.8 mid-band
    would otherwise report ~0 graded mass no matter how graded the group actually is."""
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            self.reward_function = get_abs_path(self.reward_function, prompt="Reward function")
