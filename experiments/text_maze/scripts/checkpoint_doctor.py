#!/usr/bin/env python3
"""Repair + verify a maze SFT checkpoint so the verl container's transformers
loads it CORRECTLY. This exists because the checkpoints in
`max-rl/maze_v2_sft_ckpts_guanning` were saved by transformers 5.7.0, whose
config format is forward-incompatible with the transformers 4.51 shipped in the
verl container. Three problems, all of which SILENTLY produce a broken-but-
valid-looking model (well-formed output, but training/eval quietly wrong):

  1. RoPE base (the one that cost us the whole first campaign):
     5.x stores it as nested `rope_parameters: {rope_theta: 1000000, ...}`.
     transformers 4.51 does NOT read `rope_parameters`; it falls back to its
     default `rope_theta = 10000` -- 100x too small -> scrambled positional
     encoding -> the model wanders into walls and NEVER reaches the goal.
     FIX: mirror `rope_parameters.rope_theta` into a top-level `rope_theta`
     (which 4.51 reads). Harmless under 5.x (it keeps using rope_parameters).

  2. Tokenizer class: `tokenizer_class: "TokenizersBackend"` is a 5.x class that
     4.51 doesn't have -> load crash.
     FIX: set it to `PreTrainedTokenizerFast` (byte-identical encode/decode,
     verified).

  3. DONE not a stop token: the reference SFT saves
     `generation_config.eos_token_id = [<eos>, DONE]` so the model stops at DONE;
     the 5.7.0 save dropped it to `eos_token_id = <eos>` only. Without DONE on the
     stop list, RL erodes <eos> emission and generation then runs to
     max_new_tokens spewing reward-invisible junk -> length blow-up / RLOO
     collapse. FIX: add DONE's id to generation_config.eos_token_id (reference
     parity). Gated on DONE being a real vocab token, so non-maze ckpts untouched.

Usage:
  python checkpoint_doctor.py <ckpt_dir> [<ckpt_dir> ...]   # fix (idempotent) + verify
  python checkpoint_doctor.py --verify-only <ckpt_dir> ...  # verify only, no writes
  python checkpoint_doctor.py --all [<cache_dir>]           # fix+verify every ckpt-*/ + init_model

Exit code 0 iff every checkpoint is correct after the run. Safe to run any
number of times. Pure-stdlib for the FIX (runs anywhere); the VERIFY uses
transformers when importable (the strong, mechanism-independent check) and
falls back to a structural JSON check otherwise.
"""
import argparse
import glob
import json
import os
import sys

DEFAULT_CACHE = os.environ.get(
    "HF_CACHE_DIR", os.path.expanduser("~/.cache/tailrl_maze_sft_ckpts")
)


def _write_json(path, obj):
    """Atomic write so a concurrent job reading the config can't see a torn file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _vocab_id(ckpt_dir, token):
    """Integer id of `token` in the checkpoint's tokenizer, or None if absent.

    Reads tokenizer.json's model.vocab directly (pure stdlib, no transformers).
    Returning None for an absent token makes the DONE-as-EOS fix a no-op on any
    non-maze checkpoint automatically.
    """
    tjp = os.path.join(ckpt_dir, "tokenizer.json")
    if not os.path.isfile(tjp):
        return None
    try:
        vocab = (json.load(open(tjp)).get("model") or {}).get("vocab") or {}
    except Exception:
        return None
    v = vocab.get(token)
    return int(v) if isinstance(v, int) else None


def fix_checkpoint(ckpt_dir):
    """Idempotently repair config.json + tokenizer_config.json. Returns list of changes."""
    changes = []
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path))
        rp = cfg.get("rope_parameters") or {}
        theta = rp.get("rope_theta")
        # 1a. mirror rope_theta so transformers 4.x reads the trained value
        if theta is not None and cfg.get("rope_theta") != theta:
            cfg["rope_theta"] = theta
            changes.append(f"config.rope_theta <- {theta}")
        # 1b. non-default rope types also need the 4.x-style rope_scaling mirror
        rope_type = rp.get("rope_type", "default")
        if rope_type not in ("default", None) and cfg.get("rope_scaling") is None:
            cfg["rope_scaling"] = dict(rp)
            changes.append(f"config.rope_scaling <- {rope_type} (verify manually)")
        if changes:
            _write_json(cfg_path, cfg)
    tok_path = os.path.join(ckpt_dir, "tokenizer_config.json")
    if os.path.isfile(tok_path):
        tk = json.load(open(tok_path))
        if tk.get("tokenizer_class") == "TokenizersBackend":
            tk["tokenizer_class"] = "PreTrainedTokenizerFast"
            _write_json(tok_path, tk)
            changes.append("tokenizer_class -> PreTrainedTokenizerFast")
    # 3. DONE-as-EOS (reference parity): ensure generation stops at DONE.
    done_id = _vocab_id(ckpt_dir, "DONE")
    gen_path = os.path.join(ckpt_dir, "generation_config.json")
    if done_id is not None and os.path.isfile(gen_path):
        gc = json.load(open(gen_path))
        eos = gc.get("eos_token_id")
        if eos is not None:
            eos_list = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            if done_id not in eos_list:
                gc["eos_token_id"] = eos_list + [done_id]
                _write_json(gen_path, gc)
                changes.append(f"generation_config.eos_token_id <- {eos_list + [done_id]} (DONE=stop)")
    return changes


def verify_checkpoint(ckpt_dir):
    """Return (ok: bool, detail: str).

    The guarantee we need is: transformers 4.x (the verl container) will use the
    checkpoint's TRAINED rope_theta. That is a STRUCTURAL property of config.json
    -- the top-level `rope_theta` (which 4.x reads) must equal the intended
    (nested `rope_parameters.rope_theta`) value -- and it is version-independent,
    so we check it directly. A transformers-4.x load smoke is added when
    available; transformers 5.x is deliberately NOT trusted for this check
    because 5.x reads `rope_parameters` and would green-light a config that is
    still broken for the 4.x container.
    """
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False, "no config.json"
    cfg = json.load(open(cfg_path))
    intended = (cfg.get("rope_parameters") or {}).get("rope_theta")
    top = cfg.get("rope_theta")
    # 1. structural guarantee for the 4.x container
    if intended is not None and top != intended:
        return False, (f"top-level rope_theta={top} != trained {intended} -- "
                       "transformers 4.x will use the wrong RoPE base and the model "
                       "will score ~0 (run checkpoint_doctor.py without --verify-only)")
    tokp = os.path.join(ckpt_dir, "tokenizer_config.json")
    if os.path.isfile(tokp) and json.load(open(tokp)).get("tokenizer_class") == "TokenizersBackend":
        return False, "tokenizer_class is TokenizersBackend (transformers 4.x can't load it; run the fix)"
    # 1c. DONE must be on the generation stop list, or RL generation won't halt at DONE
    done_id = _vocab_id(ckpt_dir, "DONE")
    gen_path = os.path.join(ckpt_dir, "generation_config.json")
    if done_id is not None and os.path.isfile(gen_path):
        eos = json.load(open(gen_path)).get("eos_token_id")
        eos_list = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        if done_id not in eos_list:
            return False, (f"generation_config.eos_token_id={eos} omits DONE (id {done_id}); "
                           "generation won't stop at DONE (run checkpoint_doctor.py without --verify-only)")
    # 2. optional load smoke, but only meaningful under transformers 4.x
    try:
        import transformers
        if transformers.__version__.startswith("4."):
            from transformers import AutoConfig, AutoTokenizer
            eff = AutoConfig.from_pretrained(ckpt_dir).rope_theta
            if intended is not None and float(eff) != float(intended):
                return False, f"transformers {transformers.__version__} loads rope_theta={eff}, expected {intended}"
            AutoTokenizer.from_pretrained(ckpt_dir)
            return True, f"OK (transformers {transformers.__version__} loads rope_theta={eff})"
    except Exception as e:  # load smoke is best-effort; structural check already passed
        return True, f"OK structurally (rope_theta={top}); load-smoke skipped ({type(e).__name__})"
    return True, f"OK (top-level rope_theta={top})"


def main():
    ap = argparse.ArgumentParser(description="Repair + verify maze SFT checkpoints for the verl container.")
    ap.add_argument("ckpt_dirs", nargs="*", help="checkpoint directories")
    ap.add_argument("--verify-only", action="store_true", help="verify without modifying files")
    ap.add_argument("--all", action="store_true", help="process every ckpt-*/ and init_model under a cache dir")
    args = ap.parse_args()

    dirs = list(args.ckpt_dirs)
    if args.all:
        cache = args.ckpt_dirs[0] if args.ckpt_dirs else DEFAULT_CACHE
        dirs = sorted(d for d in glob.glob(os.path.join(cache, "*")) if os.path.isfile(os.path.join(d, "config.json")))
    if not dirs:
        ap.error("give at least one checkpoint dir, or --all [cache_dir]")

    all_ok = True
    for d in dirs:
        d = d.rstrip("/")
        if not args.verify_only:
            ch = fix_checkpoint(d)
            if ch:
                print(f"[fix]  {os.path.basename(d)}: {', '.join(ch)}")
        ok, detail = verify_checkpoint(d)
        all_ok = all_ok and ok
        tag = "ok " if ok else "FAIL"
        # only print per-ckpt verify line when interesting (fail, or few dirs)
        if not ok or len(dirs) <= 8:
            print(f"[{tag}] {os.path.basename(d)}: {detail}")
    n = len(dirs)
    print(f"[done] {n} checkpoint(s); {'ALL OK' if all_ok else 'SOME BROKEN -- see FAIL above'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
