"""Training entry point: verl's PPO trainer, with this experiment's hooks installed.

The estimator is chosen with ``algorithm.adv_estimator=tailrl|grpo|rloo`` and the
reward with ``custom_reward_function.path=code_opt/reward/gem5_reward.py``.

One hook is installed before verl is handed control: the **actor carve-out**. verl
keeps only ``max_actor_ckpt_to_keep`` checkpoints in its rolling buffer (1 here, to
bound disk), but the post-hoc evaluation needs the whole trajectory, so the hook
hard-links each save into ``$PIE_ACTOR_CARVEOUT_DIR/step_<N>/``. Hard links cost
nothing and survive verl's rolling delete. It is a no-op if the variable is unset,
and it must be installed before ``verl.trainer.main_ppo`` resolves the symbol it
patches -- hence the import order below.

The task-specific per-step metrics (``gate_pass_rate``, ``mean_speedup``,
``max_speedup``, ``speedup_rate``, ``fraction_dead_groups``) are computed inside the
fork itself, in ``verl/trainer/ppo/metric_utils.py``, rather than by a driver-side
monkeypatch -- a patch installed here would not reach the Ray workers that actually
call it.

Usage -- normally through ``scripts/train.sh``:

    python -m code_opt.train \\
      algorithm.adv_estimator=tailrl \\
      data.train_files=$PIE_PARQUET_ROOT/pie_gem5_train.parquet \\
      custom_reward_function.path=code_opt/reward/gem5_reward.py ...
"""

import os

import hydra

from code_opt.guards.actor_carveout import install_actor_carveout_hook

install_actor_carveout_hook()

from verl.trainer.main_ppo import run_ppo  # noqa: E402 -- must follow the hook

# Hydra needs an absolute path to the fork's config directory.
_VERL_CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verl", "trainer", "config")
)


@hydra.main(config_path=_VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
