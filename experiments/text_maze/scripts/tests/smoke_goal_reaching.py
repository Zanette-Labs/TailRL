#!/usr/bin/env python3
"""Generation smoke test: assert an SFT checkpoint ACTUALLY reaches goals.

This is the guard that would have caught the rope_theta bug immediately: a
broken checkpoint produces valid-format paths but goal_rate == 0. We sample a
handful of eval mazes, score goal_reached with the canonical maze judge, and
FAIL (exit 1) if not a single rollout reaches the goal.

Run inside the verl container (has transformers + torch), on a GPU or CPU:
  python scripts/tests/smoke_goal_reaching.py [ckpt_dir] [N] [K]
Defaults: ckpt-3550 from $HF_CACHE_DIR, N=50 prompts, K=32 samples each.

NOTE: use a HIGH-pass-rate checkpoint (ckpt-3550, the default) for the gate. The
low SFT checkpoints have <0.05% goal rate, so N*K=1600 samples can legitimately
see zero goals and this test would false-fail on them; verify their rate with
N=200 K=128 (~25k samples) instead. The bug this guards against zeros ALL
checkpoints, so 3550 is the correct, robust canary.
"""
import ast
import importlib.util
import os
import sys

import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
CACHE = os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/.cache/maxrl_sft_ckpts"))

CK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CACHE, "ckpt-3550")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
K = int(sys.argv[3]) if len(sys.argv) > 3 else 32

spec = importlib.util.spec_from_file_location("mz", os.path.join(REPO, "verl/utils/reward_score/maze.py"))
mz = importlib.util.module_from_spec(spec); spec.loader.exec_module(mz)
PARQUET = os.path.join(REPO, "data/main_parquet_eval1000/test.parquet")


def field(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


import transformers
print(f"[smoke] transformers {transformers.__version__} | ckpt {os.path.basename(CK)} | "
      f"config rope_theta {AutoConfig.from_pretrained(CK).rope_theta} | N={N} K={K}")

tok = AutoTokenizer.from_pretrained(CK); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = "<pad>"
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(CK, torch_dtype=(torch.bfloat16 if dev == "cuda" else torch.float32)).to(dev).eval()

rows = pq.read_table(PARQUET).to_pylist()[:N]
prompts = [(field(r["prompt"])[0]["content"] if isinstance(field(r["prompt"]), list) else field(r["prompt"])) for r in rows]
gts = [field(r["reward_model"])["ground_truth"] for r in rows]
enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
pad_id = tok.convert_tokens_to_ids(tok.pad_token)

total = goal = solved = 0
with torch.no_grad():
    for i in range(N):
        ids = enc.input_ids[i:i+1].repeat(K, 1); am = enc.attention_mask[i:i+1].repeat(K, 1)
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=180,
                             do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=pad_id)
        any_goal = False
        for g in out[:, ids.shape[1]:]:
            gr = mz.judge_maze(solution_str=tok.decode(g, skip_special_tokens=False), ground_truth=gts[i]).get("goal_reached", 0.0)
            total += 1; goal += (1 if gr > 0 else 0); any_goal = any_goal or gr > 0
        solved += 1 if any_goal else 0

rate = goal / total if total else 0.0
print(f"[smoke] goal_reached: {goal}/{total} = {rate*100:.4f}%  | mazes solved >=1: {solved}/{N}")
if goal == 0:
    print("[smoke] FAIL: zero goal-reaching rollouts. The checkpoint is BROKEN for this "
          "environment (run checkpoint_doctor.py; check transformers version / rope_theta).")
    sys.exit(1)
print("[smoke] PASS: checkpoint reaches goals — setup is sound.")
sys.exit(0)
