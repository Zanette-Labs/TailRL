#!/usr/bin/env python3
"""Fast, dependency-free tests for checkpoint_doctor.py (no model / no GPU).

Builds synthetic checkpoint configs that mimic the transformers-5.x save format
and asserts the doctor detects the breakage, repairs it, and is idempotent.

Run:  python scripts/tests/test_checkpoint_doctor.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOCTOR = os.path.normpath(os.path.join(HERE, "..", "checkpoint_doctor.py"))


def make_ckpt(d, rope_parameters=None, rope_theta="__omit__", tok_class="PreTrainedTokenizerFast",
              vocab=None, eos_token_id="__omit__"):
    os.makedirs(d, exist_ok=True)
    cfg = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 4}
    if rope_parameters is not None:
        cfg["rope_parameters"] = rope_parameters
    if rope_theta != "__omit__":
        cfg["rope_theta"] = rope_theta
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    json.dump({"tokenizer_class": tok_class}, open(os.path.join(d, "tokenizer_config.json"), "w"))
    if vocab is not None:  # maze-style tokenizer so _vocab_id("DONE") resolves
        json.dump({"model": {"vocab": vocab}}, open(os.path.join(d, "tokenizer.json"), "w"))
    if eos_token_id != "__omit__":
        json.dump({"eos_token_id": eos_token_id}, open(os.path.join(d, "generation_config.json"), "w"))


def doctor(*args):
    r = subprocess.run([sys.executable, DOCTOR, *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def cfg_of(d):
    return json.load(open(os.path.join(d, "config.json")))


def tok_of(d):
    return json.load(open(os.path.join(d, "tokenizer_config.json")))


def gen_of(d):
    return json.load(open(os.path.join(d, "generation_config.json")))


MAZE_VOCAB = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "DONE": 7, "UP": 13, "DOWN": 14}


def main():
    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1; print(f"  PASS  {name}")
        else:
            failed += 1; print(f"  FAIL  {name}")

    with tempfile.TemporaryDirectory() as tmp:
        # 1. 5.x-style broken: nested rope_parameters, NO top-level rope_theta, TokenizersBackend
        d = os.path.join(tmp, "broken")
        make_ckpt(d, rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
                  rope_theta="__omit__", tok_class="TokenizersBackend")
        rc, out = doctor("--verify-only", d)
        check("verify-only flags a fresh 5.x checkpoint as broken (exit 1)", rc == 1)

        rc, out = doctor(d)  # fix + verify
        check("fix repairs it (exit 0)", rc == 0)
        check("fix added top-level rope_theta=1000000", cfg_of(d).get("rope_theta") == 1000000.0)
        check("fix set tokenizer_class=PreTrainedTokenizerFast", tok_of(d).get("tokenizer_class") == "PreTrainedTokenizerFast")

        before = open(os.path.join(d, "config.json")).read()
        rc, out = doctor(d)  # run again
        check("idempotent: re-run exits 0", rc == 0)
        check("idempotent: config unchanged on re-run", open(os.path.join(d, "config.json")).read() == before)

        # 2. wrong top-level value (e.g. someone hand-set 10000) must be caught
        d2 = os.path.join(tmp, "wrongval")
        make_ckpt(d2, rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"}, rope_theta=10000.0)
        rc, out = doctor("--verify-only", d2)
        check("verify flags wrong top-level rope_theta=10000 (exit 1)", rc == 1)
        rc, out = doctor(d2)
        check("fix corrects wrong rope_theta -> 1000000", cfg_of(d2).get("rope_theta") == 1000000.0 and rc == 0)

        # 3. already-correct checkpoint: no changes, exit 0
        d3 = os.path.join(tmp, "good")
        make_ckpt(d3, rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"}, rope_theta=1000000.0)
        good_before = open(os.path.join(d3, "config.json")).read()
        rc, out = doctor(d3)
        check("already-correct: exit 0, no rewrite", rc == 0 and open(os.path.join(d3, "config.json")).read() == good_before)

        # 4. maze checkpoint with DONE missing from the eos stop list
        good_rope = dict(rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"}, rope_theta=1000000.0)
        d4 = os.path.join(tmp, "maze_no_done")
        make_ckpt(d4, **good_rope, vocab=MAZE_VOCAB, eos_token_id=2)
        rc, out = doctor("--verify-only", d4)
        check("verify flags maze ckpt with eos=2 missing DONE (exit 1)", rc == 1)
        rc, out = doctor(d4)
        check("fix sets generation_config.eos_token_id=[2,7]", rc == 0 and gen_of(d4).get("eos_token_id") == [2, 7])
        g_before = open(os.path.join(d4, "generation_config.json")).read()
        rc, out = doctor(d4)
        check("DONE fix idempotent: re-run no rewrite", rc == 0 and open(os.path.join(d4, "generation_config.json")).read() == g_before)

        # 5. maze checkpoint already [2,7]: no change
        d5 = os.path.join(tmp, "maze_ok")
        make_ckpt(d5, **good_rope, vocab=MAZE_VOCAB, eos_token_id=[2, 7])
        gb5 = open(os.path.join(d5, "generation_config.json")).read()
        rc, out = doctor(d5)
        check("maze ckpt already [2,7]: exit 0, no rewrite", rc == 0 and open(os.path.join(d5, "generation_config.json")).read() == gb5)

        # 6. non-maze checkpoint (no DONE token): DONE fix is a no-op
        d6 = os.path.join(tmp, "nonmaze")
        make_ckpt(d6, **good_rope, vocab={"<eos>": 2, "hello": 5}, eos_token_id=2)
        rc, out = doctor(d6)
        check("non-maze ckpt (no DONE): exit 0, eos untouched", rc == 0 and gen_of(d6).get("eos_token_id") == 2)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
