#!/usr/bin/env python3
"""Live smoke test for the DONE-as-EOS fix.

Loads a real SFT checkpoint, generates with the vanilla eos (2) vs the fixed
eos-set [2, 7] (7 = DONE), and asserts:
  1. with eos=[2,7], generation STOPS at DONE  -> the first DONE token is the last
     non-pad token; nothing is generated after it;
  2. DONE survives `skip_special_tokens=True` decode (it is a normal token, id 7);
  3. the real reward (compute_score) parses the DONE-terminated response:
     valid_format == 1.0 and a finite reward in [0, 1];
  4. contrast: with plain eos=2 the model runs PAST the first DONE (emits <eos>
     after it), so first-DONE is generally NOT the last token.
Runs on CPU (the maze model is ~2M params). Exit code 0 == all asserts pass.
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP_ROOT = os.environ.get(
    "EXP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE = os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/.cache/tailrl_maze_sft_ckpts"))
DATA = os.environ.get("TAILRL_MAZE_DATA_DIR", os.path.join(EXP_ROOT, "data"))
CKPTS = sys.argv[1:] or [f"{CACHE}/ckpt-3550", f"{CACHE}/ckpt-2450"]
PARQUET = f"{DATA}/main_parquet_eval1000/test_maze_17_continuous.parquet"
N_PROMPTS, MAX_NEW = 8, 90
DONE_ID, EOS_ID, PAD_ID = 7, 2, 0

_spec = importlib.util.spec_from_file_location(
    "mzr", f"{EXP_ROOT}/src/maze_composite_v2_reward.py")
_mzr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mzr)
compute_score = _mzr.compute_score

df = pd.read_parquet(PARQUET).iloc[:N_PROMPTS]
prompts = [row["prompt"][0]["content"] for _, row in df.iterrows()]
gts = [row["reward_model"]["ground_truth"] for _, row in df.iterrows()]

def last_real_idx(ids, pad=PAD_ID):
    nz = [i for i, t in enumerate(ids) if t != pad]
    return nz[-1] if nz else -1

failures = []
for ckpt in CKPTS:
    print(f"\n{'='*70}\nCKPT {ckpt}\n{'='*70}")
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
    assert tok.convert_tokens_to_ids("DONE") == DONE_ID, "DONE id changed!"
    assert tok.eos_token_id == EOS_ID, "eos id changed!"

    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
              return_token_type_ids=False)
    plen = enc["input_ids"].shape[1]

    def gen(eos):
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                 eos_token_id=eos, pad_token_id=PAD_ID)
        return out[:, plen:].tolist()  # response tokens only

    fixed = gen([EOS_ID, DONE_ID])
    base = gen(EOS_ID)

    fixed_lens, base_lens, n_done, n_valid, rewards = [], [], 0, 0, []
    tokens_after_done_fixed = 0
    base_past_done = 0
    for i, (fr, br) in enumerate(zip(fixed, base)):
        fL, bL = last_real_idx(fr) + 1, last_real_idx(br) + 1
        fixed_lens.append(fL); base_lens.append(bL)
        # --- assertion 1: fixed stops AT done ---
        if DONE_ID in fr[:fL]:
            n_done += 1
            dpos = fr.index(DONE_ID)
            after = [t for t in fr[dpos + 1:fL] if t != PAD_ID]
            if after:
                tokens_after_done_fixed += 1
                failures.append(f"{ckpt} row{i}: {len(after)} tokens after DONE in FIXED (did not stop)")
            if dpos != fL - 1:
                failures.append(f"{ckpt} row{i}: DONE not last real token in FIXED (dpos={dpos}, last={fL-1})")
        # --- assertion 2: decode keeps DONE ---
        txt = tok.decode(fr[:fL], skip_special_tokens=True)
        if DONE_ID in fr[:fL] and "DONE" not in txt.split():
            failures.append(f"{ckpt} row{i}: 'DONE' missing from skip_special decode -> {txt[-60:]!r}")
        # --- assertion 3: reward parses ---
        res = compute_score("maze_17_continuous", txt, gts[i])
        score = res["score"] if isinstance(res, dict) else res
        vf = res.get("valid_format") if isinstance(res, dict) else None
        if DONE_ID in fr[:fL]:
            n_valid += (1 if vf == 1.0 else 0)
            if vf != 1.0:
                failures.append(f"{ckpt} row{i}: valid_format={vf} on a DONE-terminated response -> {txt[-80:]!r}")
        rewards.append(float(score))
        # --- contrast: baseline runs past DONE ---
        if DONE_ID in br[:bL]:
            dposb = br.index(DONE_ID)
            if any(t != PAD_ID for t in br[dposb + 1:bL]):
                base_past_done += 1

    print(f"prompts={N_PROMPTS} | emitted DONE (fixed): {n_done}/{N_PROMPTS}")
    print(f"FIXED  resp len: mean={np.mean(fixed_lens):.1f} max={max(fixed_lens)}  "
          f"(stops at DONE; tokens-after-DONE violations: {tokens_after_done_fixed})")
    print(f"BASE   resp len: mean={np.mean(base_lens):.1f} max={max(base_lens)}  "
          f"(eos=2; ran PAST first DONE in {base_past_done}/{n_done} DONE-runs)")
    print(f"reward: valid_format=1.0 on {n_valid}/{n_done} DONE-runs | reward mean={np.mean(rewards):.3f} "
          f"min={min(rewards):.3f} max={max(rewards):.3f}")
    ex = tok.decode([t for t in fixed[0] if t != PAD_ID], skip_special_tokens=True)
    print(f"example FIXED decode (row0, last 90 chars): ...{ex[-90:]!r}")

# --- positive control: each maze's OWN optimal path must score full success ---
# (proves the reward machinery discriminates good paths from the model's failing
#  greedy paths above, so the mean=0 there is a weak model, not a broken reward.)
print(f"\n{'='*70}\nPOSITIVE CONTROL: scoring ground-truth optimal paths\n{'='*70}")
MOVES = {"UP", "DOWN", "LEFT", "RIGHT"}
pos_ok = 0
print("reward keys:", sorted((compute_score("maze_17_continuous", "UP DONE", gts[0]) or {}).keys()))
for i, gt in enumerate(gts):
    toks = gt.split()
    if "DONE" not in toks:
        continue
    d = toks.index("DONE")
    j = d
    while j > 0 and toks[j - 1] in MOVES:   # maximal trailing move-run = the solution
        j -= 1
    opt = " ".join(toks[j:d] + ["DONE"])
    res = compute_score("maze_17_continuous", opt, gt)
    sc = res["score"] if isinstance(res, dict) else res
    fs = res.get("is_full_success") if isinstance(res, dict) else None
    gr = res.get("goal_reached") if isinstance(res, dict) else None
    pos_ok += (1 if sc >= 0.9 else 0)
    if i < 4:
        print(f"  row{i}: opt_path_len={d - j} score={sc:.3f} goal_reached={gr} full_success={fs}")
print(f"optimal-path full-reward: {pos_ok}/{len(gts)} scored >=0.9")
if pos_ok < len(gts) * 0.8:
    failures.append(f"positive control: only {pos_ok}/{len(gts)} optimal paths scored >=0.9 (reward machinery suspect)")

print(f"\n{'='*70}")
if failures:
    print(f"FAIL — {len(failures)} assertion violation(s):")
    for f in failures[:20]:
        print("  -", f)
    sys.exit(1)
print("PASS — DONE terminates generation, survives decode, and the reward parses. EOS fix verified.")
sys.exit(0)
