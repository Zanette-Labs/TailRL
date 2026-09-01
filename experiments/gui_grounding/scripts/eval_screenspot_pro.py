#!/usr/bin/env python3
"""ScreenSpot-Pro GUI-grounding eval for the TailRL vs GRPO vs RLOO comparison.

Mirror of scripts/eval_refcoco.py (same DP sharding / --merge / CLI surface), adapted for
point grounding in normalized [0,1000] coordinates:

Phase A (point metrics): greedy decode over the full split -> in-box hit-rate + mean soft
                         reward, reported OVERALL and per `category` (industry) and `ui_type`.
Phase B (tail metric):   bestk_n samples @ temp 1.0 -> unbiased best@k on the 0/1 in-box hit
                         (== pass@k) AND unbiased maxreward@k on the continuous soft reward,
                         overall + per category.

DATA-PARALLEL: run one process per GPU with --num_shards N --shard_id i (strided 1/N slice),
then aggregate with --merge. Reuses examples/reward_function/gui_grounding.compute_score so the
eval `accuracy` (in-box hit) and `overall` (soft) match the training reward exactly.

Run INSIDE the Apptainer container (vLLM + Qwen2.5-VL). vLLM/transformers are imported inside
main() so the unbiased estimators can be imported on CPU (e.g. by the test suite).
"""
import argparse
import bisect
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from jinja2 import Template


def load_reward_module(repo_root, module="segui_point.py"):
    """Load the reward used for SCORING. Selectable because best_reward@k must be measured with the
    reward the model was actually trained on: a SE-GUI run scored by the Gaussian reward reports a
    quantity nothing optimised. `hit` is unaffected -- the in-box criterion is identical in both
    (pinned by tests/segui/test_segui_point.py::test_the_two_rewards_agree_on_the_hit_criterion)."""
    path = os.path.join(repo_root, "examples/reward_function", module)
    spec = importlib.util.spec_from_file_location("gui_grounding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bestk_unbiased(n, c, k):
    """Unbiased best@k == pass@k (Chen et al. 2021): P(>=1 of k random samples is a hit)."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _logC(n, k):
    """log C(n, k), vectorised over n (k scalar); -inf where invalid."""
    from scipy.special import gammaln
    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, -np.inf)
    ok = (k >= 0) & (n >= k)
    out[ok] = gammaln(n[ok] + 1) - gammaln(k + 1) - gammaln(n[ok] - k + 1)
    return out


def max_reward_at_k(rewards, k):
    """Unbiased E[max reward over a random size-k subset] from N per-prompt samples.
    Continuous generalisation of pass@k; reduces to bestk_unbiased for binary rewards."""
    r = np.sort(np.asarray(rewards, dtype=float))
    N = r.size
    k = min(k, N)
    i = np.arange(1, N + 1)
    w = np.exp(_logC(i - 1, k - 1) - _logC(np.array(N), k))
    return float(np.dot(w, r))


_SIZE_LABELS = ["tiny", "small", "medium", "large"]   # len == len(default --size_edges) + 1

# Defaults for the Phase-B sampled pass. These are PROTOCOL, so they are named here, exposed as
# flags, and recorded in every results JSON -- never left implicit in a SamplingParams call.
#
# 0.95 is the pass@k convention the estimator was established under (Chen et al. 2021, HumanEval).
# T=0.6 / top_p=0.95 / top_k=-1 is also what MaxRL uses to validate its LLM benchmarks (smollm,
# qwen3), which matters because MaxRL is the closest methodological neighbour to this work -- TailRL
# reduces to centred MaxRL for binary rewards (core_algos.py, Prop 6.1).
#
# NOTE the tension, since it will come up again: for their COVERAGE measurement (maze, n=2048)
# MaxRL instead samples at T=1.0 with NO nucleus truncation. Lower temperature concentrates the
# distribution, so it depresses pass@k at large k and compresses exactly the distributional-breadth
# difference these arms are being compared on. Whichever is chosen, every arm must use the same one.
BESTK_TEMPERATURE = 0.6
BESTK_TOP_P = 0.95
BESTK_TOP_K = -1          # -1 disables top-k


def _row_size_bin(row, edges):
    """GT box area as a fraction of the [0,1000]^2 canvas -> size-bin label. Reads row["gt"]
    (the JSON GT string/dict that load_split stores), NOT row["answer"]."""
    v = row["gt"]
    v = json.loads(v) if isinstance(v, str) else v
    b = v.get("bbox", v.get("bbox_2d")) if isinstance(v, dict) else v
    x1, y1, x2, y2 = [float(t) for t in b][:4]
    area_frac = max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1.0e6
    return _SIZE_LABELS[bisect.bisect_right(edges, area_frac)]


def load_split(parquet_path):
    df = pd.read_parquet(parquet_path)
    has_cat = "category" in df.columns
    has_ui = "ui_type" in df.columns
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        imgs = r["images"]
        img = imgs[0] if isinstance(imgs, (list, np.ndarray)) else imgs
        rows.append({
            # STABLE identity = the source parquet row index. Sharding is a strided slice and the
            # best@k path subsets before sharding, so a row's position in `rows` is not stable
            # across shards or subset sizes -- per-sample records would be unjoinable without this.
            "idx": i,
            "image": img, "problem": r["problem"], "gt": r["answer"],
            "category": (r["category"] if has_cat else "all"),
            "ui_type": (r["ui_type"] if has_ui else "all"),
        })
    return rows


def shard(rows, num_shards, shard_id):
    """Strided slice so each shard gets a balanced, deterministic 1/num_shards of the rows."""
    return rows[shard_id::num_shards] if num_shards > 1 else rows


def _content_list(prompt_str):
    """Interleave image/text by splitting on '<image>' -- mirrors verl.utils.dataset._build_messages
    so the image token lands at the SAME position the model saw in training. A template with text
    BEFORE {{content}} (e.g. gui_g1.jinja's 'Grounding instruction is: ') keeps the image mid-prompt
    instead of forcing it first; gui.jinja (content first) keeps the image first. Either way this
    matches training exactly."""
    out = []
    for i, chunk in enumerate(prompt_str.split("<image>")):
        if i != 0:
            out.append({"type": "image"})
        if chunk:
            out.append({"type": "text", "text": chunk})
    return out


def _area_resize(img, min_pixels, max_pixels):
    """Replicate verl.utils.dataset.process_image's area-resize (down to <=max_pixels then up to
    >=min_pixels, int() truncation, then RGB) WITHOUT importing verl, so eval/filter stay verl-free
    (runnable under NO_OVERLAY) while the image frame still == the training frame. Bit-exact."""
    import math
    w, h = img.width, img.height
    if max_pixels and w * h > max_pixels:
        f = math.sqrt(max_pixels / (w * h)); w, h = int(w * f), int(h * f); img = img.resize((w, h))
    if min_pixels and w * h < min_pixels:
        f = math.sqrt(min_pixels / (w * h)); w, h = int(w * f), int(h * f); img = img.resize((w, h))
    return img.convert("RGB") if img.mode != "RGB" else img


def build_inputs(rows, image_dir, processor, template, min_pixels=None, max_pixels=None):
    """vLLM inputs matching training EXACTLY: (1) render the template over the FULL '<image>'-prefixed
    problem with per-row rw/rh (the model-frame dims, for resolution-declaring templates like
    gta1_point.jinja; unused vars are harmless), (2) honor an optional leading <system>...</system>
    block in the rendered prompt (-> a system message, mirroring verl dataset._build_messages),
    (3) split on '<image>' for image placement (see _content_list); (4) area-resize the image like
    the training dataset does before the Qwen processor's smart_resize (see _area_resize), so the
    reward's resized_frame() assumption holds for every image."""
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    except Exception:  # pragma: no cover
        from qwen_vl_utils import smart_resize
    inputs = []
    for r in rows:
        image = _area_resize(Image.open(os.path.join(image_dir, r["image"])), min_pixels, max_pixels)
        # smart_resize on the area-resized dims == gui_grounding.resized_frame(orig) (two-stage parity)
        rh, rw = smart_resize(image.height, image.width, factor=28,
                              min_pixels=min_pixels, max_pixels=max_pixels)
        rendered = template.render(content=r["problem"], rw=rw, rh=rh)
        system_str = None
        if rendered.startswith("<system>") and "</system>" in rendered:
            system_str, rendered = rendered[len("<system>"):].split("</system>", 1)
        messages = [{"role": "user", "content": _content_list(rendered)}]
        if system_str:
            messages.insert(0, {"role": "system", "content": system_str.strip()})
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
    return inputs


def score_samples(rows, outputs, gui, min_pixels, max_pixels, sigma_min=0.0, sigma_scale=0.5, format_weight=0.0):
    """rows[i] vs outputs[i].outputs (list of samples) -> per-example
    {"hits":[0/1], "softs":[float], "fmts":[0/1]}.

    Reuses gui_grounding.compute_score: accuracy == 0/1 in-box hit, overall == continuous soft.
    min/max_pixels MUST equal the processor's (the eval generation frame) so the model's
    resized-pixel click is mapped into [0,1000] with the dims the model actually saw.

    `fmts` is kept because `overall` is AFFINE in the format flag under both rewards
    (segui_point: point + w*fmt; gui_grounding: (1-w)*soft + w*fmt), so persisting it makes the
    composite recomputable at ANY format_weight post hoc. Without it, scoring at the wrong weight
    is only repairable by re-generating every sample.
    """
    out = []
    for r, o in zip(rows, outputs):
        hits, softs, fmts = [], [], []
        for comp in o.outputs:
            s = gui.compute_score({"response": comp.text, "ground_truth": r["gt"]},
                                  sigma_min=sigma_min, sigma_scale=sigma_scale, format_weight=format_weight,
                                  min_pixels=min_pixels, max_pixels=max_pixels)
            hits.append(float(s["accuracy"]))
            softs.append(float(s["overall"]))
            fmts.append(float(s["format"]))
        out.append({"hits": hits, "softs": softs, "fmts": fmts})
    return out


def dump_samples(rows, scored, path):
    """Persist EVERY sample's (reward, hit) to parquet, keyed by the source row index.

    Without this the whole sampled eval is unrecoverable: score_samples reduces each completion to
    two floats and the list dies with the process, so pass@k cannot be recomputed at a different k,
    bootstrap CIs cannot be taken over items, and no failure can be inspected. 256 samples x 1,581
    items is ~405k rows -- a few MB of parquet against several GPU-hours of generation.
    """
    recs = {"idx": [], "category": [], "ui_type": [], "sample_idx": [], "reward": [], "hit": [],
            "format": []}
    for r, s in zip(rows, scored):
        fmts = s.get("fmts") or [float("nan")] * len(s["softs"])
        for j, (soft, hit, fmt) in enumerate(zip(s["softs"], s["hits"], fmts)):
            recs["idx"].append(r["idx"])
            recs["category"].append(r["category"])
            recs["ui_type"].append(r["ui_type"])
            recs["sample_idx"].append(j)
            recs["reward"].append(soft)
            recs["hit"].append(hit)
            recs["format"].append(fmt)
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"wrote {len(df):,} per-sample records -> {path}")
    return path


def merge_shards(output_dir, num_shards, allow_partial=False):
    """Weighted-by-count merge of per-shard results -> eval_results.json (exact: every metric
    is a per-example mean, so weighting each section/key's metrics by `n` is exact).

    REFUSES to merge an incomplete shard set. `num_shards` used to be accepted and then ignored,
    so a run that lost a shard -- which happens: vLLM dies on a stale NFS handle roughly 1 shard in
    50 -- produced a merged number over a SUBSET, written to the same filename, with `num_shards`
    recording the count found rather than the count expected. Nothing downstream could tell. Worse,
    comparing arms that each lost a different shard compares them on different item sets, which is
    the one failure that invalidates the whole comparison rather than just adding noise.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "eval_results.shard*.json")))
    if not files:
        raise FileNotFoundError(f"no shard files in {output_dir}")
    found = set()
    for f in files:
        m = re.search(r"eval_results\.shard(\d+)\.json$", os.path.basename(f))
        if m:
            found.add(int(m.group(1)))
    missing = sorted(set(range(num_shards)) - found) if num_shards and num_shards > 1 else []
    if missing:
        msg = (f"INCOMPLETE shard set in {output_dir}: {len(found)}/{num_shards} present, "
               f"missing shard ids {missing}")
        if not allow_partial:
            raise SystemExit(f"ABORT: {msg}\n       re-run them, or pass --allow_partial to merge "
                             f"a subset (the result is then NOT comparable to a full run).")
        print(f"WARNING: {msg} -- merging anyway (--allow_partial)")

    shards = [json.load(open(f)) for f in files]
    # A shard scored with a different reward/format_weight/prompt is not poolable with the others:
    # the metrics are means of different quantities. Catch it here rather than in a plot.
    scorings = {json.dumps(s.get("scoring"), sort_keys=True) for s in shards}
    if len(scorings) > 1:
        raise SystemExit(f"ABORT: shards in {output_dir} were scored with DIFFERENT settings; "
                         f"they cannot be pooled:\n       " + "\n       ".join(sorted(scorings)))
    merged = {"checkpoint": shards[0]["checkpoint"], "num_shards": len(files),
              "expected_shards": num_shards, "shard_ids": sorted(found),
              "complete": not missing, "scoring": shards[0].get("scoring"),
              "point": {}, "bestk": {}}
    for section in ("point", "bestk"):
        keys = set().union(*[set(s.get(section, {}).keys()) for s in shards])
        for key in keys:
            parts = [s[section][key] for s in shards if key in s.get(section, {}) and s[section][key]]
            n = sum(p.get("n", 0) for p in parts)
            if n == 0:
                continue
            agg = {"n": n}
            for mk in parts[0]:
                if mk == "n":
                    continue
                agg[mk] = sum(p[mk] * p["n"] for p in parts) / n
            merged[section][key] = agg
    out = os.path.join(output_dir, "eval_results.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print("merged", len(files), "shards ->", out)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)          # merged HF dir OR base model id
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-VL-3B-Instruct")
    # Root that `--format_prompt` and `--reward_module` are resolved against. Defaults to the
    # experiment directory (this file's parent's parent), so the harness works from a clean clone
    # with no configuration; TAILRL_GUI_ROOT overrides it if you run from somewhere else.
    ap.add_argument("--repo_root", default=os.environ.get(
        "TAILRL_GUI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--data_dir", required=True)                # dir with <split>.parquet
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="screenspot_pro")        # parquet basename
    ap.add_argument("--num_shards", type=int, default=1)        # data-parallel: # of single-GPU replicas
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--merge", action="store_true")             # merge existing shard files, then exit
    ap.add_argument("--allow_partial", action="store_true",
                    help="merge even if shards are missing; the result is NOT comparable to a full run")
    # Persist every (item, sample) reward/hit so pass@k can be recomputed at any k and bootstrapped
    # over items without re-generating. Cheap in bytes, irreplaceable once the process exits.
    ap.add_argument("--dump_samples", action="store_true")
    ap.add_argument("--no_bestk", action="store_true")          # greedy point metrics only (v2 sanity)
    ap.add_argument("--bestk_subset", type=int, default=0)      # 0 = all examples
    ap.add_argument("--bestk_n", type=int, default=1024)
    ap.add_argument("--bestk_ks", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--bestk_chunk_items", type=int, default=64,
                    help="items generated per llm.generate call. Caps peak host/GPU memory "
                         "independently of --num_shards; see the chunking comment in Phase B. "
                         "64 x n=512 = 33k in-flight requests, inside the envelope that 16/32/64-"
                         "shard runs proved. Raise only with a memory measurement in hand.")
    ap.add_argument("--bestk_temperature", type=float, default=BESTK_TEMPERATURE)
    ap.add_argument("--bestk_top_p", type=float, default=BESTK_TOP_P)
    ap.add_argument("--bestk_top_k", type=int, default=BESTK_TOP_K,
                    help="-1 disables top-k (vLLM convention)")
    ap.add_argument("--max_tokens", type=int, default=64)       # direct point -> short response
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--max_pixels", type=int, default=12845056)  # eval NATIVE (errata §R2): 16,384 image-token cap
    # reward shape -- MUST match data.reward.reward_function_kwargs in training (else the soft reward
    # is on a different scale than the wandb val score). Errata O1: defaults 0.0/0.5 == compute_score's
    # own defaults == the training config; T1 pins this parity.
    ap.add_argument("--sigma_min", type=float, default=0.0)
    ap.add_argument("--sigma_scale", type=float, default=0.5)
    # 0.5 is the weight the released runs TRAINED with (segui_point: overall = point + w*format,
    # so the reward lives on [0, 2.5]). It used to default to 0.0, which silently scored
    # best_reward@k on [0, 2] -- a different quantity that still looks entirely plausible.
    ap.add_argument("--format_weight", type=float, default=0.5)
    ap.add_argument("--max_model_len", type=int, default=17408)  # native SS-Pro val: 16,384 image + text (errata §R2)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    # size-binned pass@k (sub-hypothesis: TailRL gains concentrate on larger elements). Edges are GT-box
    # area fractions of the [0,1000]^2 canvas; len(edges)+1 == len(_SIZE_LABELS).
    ap.add_argument("--size_edges", type=float, nargs="+", default=[0.0005, 0.002, 0.01])
    # prompt template relative to repo_root; 7B/GTA1 passes examples/format_prompt/gui_g1.jinja, the
    # 3B/Click-100k line keeps the default gui.jinja (train/eval parity per-experiment).
    ap.add_argument("--format_prompt", default="examples/format_prompt/gui_point_v2.jinja")
    ap.add_argument("--reward_module", default="segui_point.py",
                    help="reward file under examples/reward_function/ used for scoring; must match "
                         "the run's worker.reward.reward_function or best_reward@k is meaningless")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.merge:
        merge_shards(args.output_dir, args.num_shards, allow_partial=args.allow_partial)
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    gui = load_reward_module(args.repo_root, args.reward_module)
    template = Template(open(os.path.join(args.repo_root, args.format_prompt)).read().strip())
    processor = AutoProcessor.from_pretrained(
        args.tokenizer, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    llm = LLM(model=args.checkpoint_dir, tokenizer=args.tokenizer, trust_remote_code=True,
              dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 1}, max_model_len=args.max_model_len, seed=args.seed,
              # pin vLLM's image processing to the SAME frame the reward normalizes by (else the
              # model sees the model-default ~12.8 MP -> wrong coords AND prompts can exceed max_model_len)
              mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels})

    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)
    # Provenance for the reward SCALE and the sampling distribution. Recorded because a run scored
    # with the wrong format_weight (or sampled at the wrong top_p) produces numbers that look
    # entirely sane -- the 3B best@k pass was scored at format_weight 0.0 against a run trained at
    # 0.5, which put every reward@k on [0,2] instead of [0,2.5] with nothing in the artifact saying
    # so. merge_shards now refuses to merge shards whose `scoring` blocks disagree.
    results = {"checkpoint": args.checkpoint_dir, "shard_id": args.shard_id,
               "num_shards": args.num_shards,
               "scoring": {"reward_module": args.reward_module,
                           "format_weight": args.format_weight,
                           "sigma_min": args.sigma_min, "sigma_scale": args.sigma_scale,
                           "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
                           "format_prompt": args.format_prompt,
                           "bestk_n": args.bestk_n,
                           "bestk_temperature": args.bestk_temperature,
                           "bestk_top_p": args.bestk_top_p,
                           "bestk_top_k": args.bestk_top_k},
               "point": {}, "bestk": {}}
    split_pq = os.path.join(args.data_dir, f"{args.split}.parquet")

    # ---- Phase A: point metrics (greedy), overall + per category + per ui_type ----
    rows = shard(load_split(split_pq), args.num_shards, args.shard_id)
    if rows:
        outs = llm.generate(build_inputs(rows, args.image_dir, processor, template, args.min_pixels, args.max_pixels), greedy)
        scored = score_samples(rows, outs, gui, args.min_pixels, args.max_pixels, args.sigma_min, args.sigma_scale, args.format_weight)

        def point_agg(keep):
            idxs = [i for i in range(len(rows)) if keep(rows[i])]
            if not idxs:
                return None
            hits = [scored[i]["hits"][0] for i in idxs]
            softs = [scored[i]["softs"][0] for i in idxs]
            return {"hit": float(np.mean(hits)), "soft": float(np.mean(softs)), "n": len(idxs)}

        results["point"]["overall"] = point_agg(lambda r: True)
        for cat in sorted(set(r["category"] for r in rows)):
            results["point"][f"cat:{cat}"] = point_agg(lambda r, c=cat: r["category"] == c)
        for ui in sorted(set(r["ui_type"] for r in rows)):
            results["point"][f"ui:{ui}"] = point_agg(lambda r, u=ui: r["ui_type"] == u)
        print(f"[point shard {args.shard_id}/{args.num_shards}] overall: {results['point']['overall']}")

    # ---- Phase B: best@k (binary in-box) + maxreward@k (soft), overall + per category ----
    if not args.no_bestk:
        rows = load_split(split_pq)
        rng = np.random.default_rng(args.seed)                  # SAME subset across shards
        if args.bestk_subset and args.bestk_subset < len(rows):
            rows = [rows[i] for i in sorted(rng.choice(len(rows), args.bestk_subset, replace=False))]
        rows = shard(rows, args.num_shards, args.shard_id)
        if rows:
            samp = SamplingParams(n=args.bestk_n, temperature=args.bestk_temperature,
                                  top_p=args.bestk_top_p, top_k=args.bestk_top_k,
                                  max_tokens=args.max_tokens)
            # Generate in ITEM CHUNKS. build_inputs decodes every image into host memory at once and
            # llm.generate holds a RequestOutput per (item, sample), so peak memory used to scale
            # with SHARD SIZE -- making --num_shards a memory knob disguised as a parallelism knob.
            # Every previously-run config happened to stay under ~99 items/shard and so never hit
            # it; a 4-shard sweep (395 items x 512 samples) OOM-killed the 7B jobs at --mem=112G and
            # tripped `CUDA error: an illegal memory access` inside vLLM's multimodal transfer path
            # on the 3B ones. Chunking pins peak memory to --bestk_chunk_items no matter how the
            # work is split, so shard count can be chosen for scheduling alone.
            #
            # Exactness is unaffected: score_samples returns one entry per row in row order, so the
            # concatenation over chunks is identical to one call over all rows. Chunks are scored
            # and discarded, so the images do not accumulate.
            gscored, scored = [], []
            step = max(1, args.bestk_chunk_items)
            for c0 in range(0, len(rows), step):
                chunk = rows[c0:c0 + step]
                inputs = build_inputs(chunk, args.image_dir, processor, template, args.min_pixels, args.max_pixels)
                # greedy on the SAME subset -> apples-to-apples best@1-vs-greedy (Phase A greedy is
                # the full split, so it is NOT comparable to best@1 which lives on this subset).
                gscored += score_samples(chunk, llm.generate(inputs, greedy), gui, args.min_pixels, args.max_pixels, args.sigma_min, args.sigma_scale, args.format_weight)
                scored += score_samples(chunk, llm.generate(inputs, samp), gui, args.min_pixels, args.max_pixels, args.sigma_min, args.sigma_scale, args.format_weight)
                del inputs

            def bestk_agg(keep):
                idxs = [i for i in range(len(rows)) if keep(rows[i])]
                if not idxs:
                    return None
                d = {}
                for k in args.bestk_ks:
                    hb = [bestk_unbiased(args.bestk_n, int(round(sum(scored[i]["hits"]))), k) for i in idxs]
                    sm = [max_reward_at_k(scored[i]["softs"], k) for i in idxs]
                    d[f"hitbest@{k}"] = float(np.mean(hb))
                    d[f"softmax@{k}"] = float(np.mean(sm))
                # greedy on this SAME subset -> directly comparable to softmax@1/hitbest@1
                d["greedy_hit"]  = float(np.mean([gscored[i]["hits"][0]  for i in idxs]))
                d["greedy_soft"] = float(np.mean([gscored[i]["softs"][0] for i in idxs]))
                d["n"] = len(idxs)
                return d

            results["bestk"]["overall"] = bestk_agg(lambda r: True)
            for cat in sorted(set(r["category"] for r in rows)):
                results["bestk"][f"cat:{cat}"] = bestk_agg(lambda r, c=cat: r["category"] == c)
            # size-binned pass@k / best@k (merge_shards already weights every bestk key by n)
            for lab in _SIZE_LABELS:
                results["bestk"][f"size:{lab}"] = bestk_agg(lambda r, L=lab: _row_size_bin(r, args.size_edges) == L)
            print(f"[best@k shard {args.shard_id}/{args.num_shards}] overall: "
                  f"{json.dumps(results['bestk'].get('overall'), indent=2)}")

            if args.dump_samples:
                dump_samples(rows, scored, os.path.join(
                    args.output_dir, f"samples.shard{args.shard_id}of{args.num_shards}.parquet"))

    fname = "eval_results.json" if args.num_shards == 1 else f"eval_results.shard{args.shard_id}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", os.path.join(args.output_dir, fname))


if __name__ == "__main__":
    main()
