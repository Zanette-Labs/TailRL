"""One shard of the best_reward@k / pass@k test evaluation for a single snapshot.

Reuses the proven generate+gem5-score path and the verified-unbiased best_at_k
from post_hoc_eval.py. Shards the test prompts CONTIGUOUSLY (prompt i -> shard
i*N//T) so every prompt is scored exactly once; the aggregator recombines the
per-prompt raw scores into the full-n best@k. Idempotent: a finished shard
writes shard_<i>of<T>.done.json and is skipped on re-run.

The gem5 reward config (case_select / K / workers / native timeout) is read from
the environment by gem5_reward.py — the cell script sets it. We dump the RAW
per-completion scores (the speedup-ratio reward) + accs so best@k at ANY k can be
recomputed offline without re-running gem5.

Usage (normally driven by scripts/eval_shard.sh):
  python -m code_opt.eval.besteval_shard \
    --hf_dir "$RUN_DIR/actor_ckpts/step_300/huggingface" \
    --val_parquet "$PIE_PARQUET_ROOT/pie_gem5_test.parquet" \
    --shard 3 --n_shards 176 --n_completions 4096 --temperature 0.6 --top_p 0.95 \
    --out_dir "$EVAL_ROOT/tailrl_step300"
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from code_opt.eval.post_hoc_eval import (   # noqa: E402
    generate_completions, score_completions, best_at_k, pass_at_k,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True, help="snapshot's huggingface/ dir")
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, required=True)
    ap.add_argument("--n_completions", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_case_dir", default=os.environ.get("PIE_TEST_CASE_DIR"),
                    help="Merged PIE test-case corpus; defaults to $PIE_TEST_CASE_DIR.")
    ap.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"shard_{args.shard}of{args.n_shards}.done.json"
    raw = out_dir / f"shard_{args.shard}of{args.n_shards}.jsonl"
    if done.exists():
        print(f"[shard {args.shard}/{args.n_shards}] done.json present, skip"); return 0

    df = pd.read_parquet(args.val_parquet).reset_index(drop=True)
    n = len(df)
    # contiguous slice: prompt i belongs to shard floor(i*T/n)
    idx = [i for i in range(n) if (i * args.n_shards) // n == args.shard]
    if not idx:
        print(f"[shard {args.shard}] empty slice"); done.write_text(json.dumps({"empty": True})); return 0
    sub = df.iloc[idx].reset_index(drop=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    def render(c): return tok.apply_chat_template(
        [{"role": "user", "content": c}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)

    prompts, gts, meta = [], [], []
    for gi, (_, row) in zip(idx, sub.iterrows()):   # gi = GLOBAL prompt index (0..n-1)
        uc = next((m["content"] for m in row["prompt"] if m.get("role") == "user"), None)
        if uc is None:
            continue
        gt = json.loads(row["reward_model"]["ground_truth"])
        prompts.append(render(uc)); gts.append(row["reward_model"]["ground_truth"])
        meta.append({"global_idx": gi, "problem_id": gt["problem_id"], "src_id": gt["src_id"],
                     "n_usable": len(gt["usable_case_ids"])})

    print(f"[shard {args.shard}/{args.n_shards}] {len(prompts)} prompts x n={args.n_completions} "
          f"(case_select={os.environ.get('PIE_GEM5_CASE_SELECT')} K={os.environ.get('PIE_GEM5_REWARD_K')} "
          f"temp={args.temperature}) on {args.hf_dir}", flush=True)
    t0 = time.monotonic()
    comps = generate_completions(args.hf_dir, prompts, args.n_completions,
                                 max_tokens=args.max_tokens, temperature=args.temperature,
                                 top_p=args.top_p, tensor_parallel_size=args.tensor_parallel_size)
    print(f"[shard {args.shard}] gen {time.monotonic()-t0:.0f}s", flush=True)
    t0 = time.monotonic()
    per = score_completions(comps, gts, args.test_case_dir)
    print(f"[shard {args.shard}] gem5 score {time.monotonic()-t0:.0f}s", flush=True)

    # Rollout dump. Two modes (mutually exclusive; prompt-set mode wins):
    #  (A) prompt-set random dump [BESTEVAL_DUMP_PROMPTS_JSON]: for prompts whose
    #      GLOBAL index is in the chosen set, write BESTEVAL_DUMP_N uniformly-random
    #      completions (seeded by global_idx -> reproducible & arm-consistent).
    #  (B) legacy top-K-by-speedup [BESTEVAL_DUMP_GENS=K]: top-K per prompt.
    # Both write a sidecar gens_*.jsonl; scores for ALL n completions always land
    # in the raw .jsonl so best@k/pass@k use the full n regardless of the dump.
    dump_prompts_json = os.environ.get("BESTEVAL_DUMP_PROMPTS_JSON", "")
    dump_set, dump_n = None, int(os.environ.get("BESTEVAL_DUMP_N", "64"))
    if dump_prompts_json:
        if not os.path.isfile(dump_prompts_json):
            # Do not lose a finished shard's gem5 scoring over a missing dump file.
            print(f"[shard {args.shard}] BESTEVAL_DUMP_PROMPTS_JSON={dump_prompts_json} "
                  f"does not exist; skipping the rollout dump (scores are unaffected). "
                  f"Generate it with scripts/make_dump_prompts.py.", flush=True)
            dump_prompts_json = ""
        else:
            dump_set = set(json.load(open(dump_prompts_json))["chosen_global_idx"])
    dump_k = int(os.environ.get("BESTEVAL_DUMP_GENS", "0"))
    want_gens = (dump_set is not None) or (dump_k > 0)
    gens_path = out_dir / f"shard_{args.shard}of{args.n_shards}.gens.jsonl" if want_gens else None
    gf = open(gens_path, "w") if gens_path else None
    with open(raw, "w") as f:
        for m, results, texts, pr in zip(meta, per, comps, prompts):
            scores = [float(r.get("score", 0.0)) for r in results]
            accs = [float(r.get("acc", 0.0)) for r in results]
            speeds = [float(r.get("speedup", 0.0)) for r in results]
            rec = {**m, "n": len(scores), "n_pass": int(sum(a == 1.0 for a in accs)),
                   "scores": scores}
            # Metrics are a convenience — `scores` is the source of truth (the aggregator
            # recomputes from it). Guard so any estimator issue NEVER 0-bytes the file and
            # loses this cell's ~30-40 min of gem5 scoring.
            try:
                rec["best_at_k"] = {str(k): best_at_k(scores, k) for k in args.k_values}
                rec["pass_at_k"] = {str(k): pass_at_k(len(accs), rec["n_pass"], k) for k in args.k_values}
            except Exception as e:  # noqa: BLE001
                rec["best_at_k"] = None; rec["pass_at_k"] = None; rec["metric_error"] = repr(e)
                print(f"[shard {args.shard}] metric calc failed for {m.get('problem_id')}: {e!r} "
                      f"(scores saved, recomputable offline)", flush=True)
            f.write(json.dumps(rec) + "\n")
            if gf is None:
                continue
            if dump_set is not None:
                if m["global_idx"] not in dump_set:
                    continue
                import random as _rnd
                take = min(dump_n, len(scores))
                sel = sorted(_rnd.Random(m["global_idx"]).sample(range(len(scores)), take))
                gf.write(json.dumps({**m, "prompt": pr, "dump_mode": "uniform_random",
                                     "dump_seed": m["global_idx"], "gens": [
                    {"gen_index": j, "score": scores[j], "acc": accs[j],
                     "speedup": speeds[j], "text": texts[j]} for j in sel]}) + "\n")
            else:
                top = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:dump_k]
                gf.write(json.dumps({**m, "prompt": pr, "dump_mode": "top_by_speedup", "top": [
                    {"rank": r, "score": scores[j], "acc": accs[j], "speedup": speeds[j],
                     "text": texts[j]} for r, j in enumerate(top)]}) + "\n")
    if gf is not None:
        gf.close()
    done.write_text(json.dumps({"shard": args.shard, "n_shards": args.n_shards,
                                "n_prompts": len(meta), "raw": str(raw)}))
    print(f"[shard {args.shard}] wrote {raw} ({len(meta)} prompts)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
