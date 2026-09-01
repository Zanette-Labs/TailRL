"""Code-optimization experiment: TailRL vs GRPO vs RLOO on the PIE C++ benchmark,
with a live gem5 speedup x correctness reward.

Layout:
  advantages.py     the estimators, pure torch, no verl dependency
  verl_register.py  the adapter the vendored verl fork dispatches into
  train.py          training entry point
  reward/           the gem5 speedup reward verl calls per rollout
  measurement/      compile, run-for-correctness, and time under gem5
  guards/           the checkpoint carve-out hook
  eval/             sharded pass@k / best_reward@k test evaluation
"""
