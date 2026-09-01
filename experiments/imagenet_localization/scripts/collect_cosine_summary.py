#!/usr/bin/env python3
"""Consolidate the TailRL-vs-J_ord gradient cosine results into one tracked JSON.

The per-run outputs written by ``gradient_analysis.py cosine`` live under
``$TAILRL_RESULTS_DIR``, which is gitignored, so the numbers behind the "TailRL
converges to J_ord as G grows" claim are not reproducible from the repo alone.
This script collects every ``cosine_vs_G_*_ref_tailrl_population.json`` it can find
and emits a single self-describing summary into the experiment's ``docs/``.

The reference objective is ``tailrl_population`` (J_ord), which
``losses.localization_tailrl_population_loss`` derives as the exact N -> infinity
limit of the TailRL estimator for the max-over-GT IoU reward. The prediction under
test is therefore: cos(grad_TailRL(G), grad_J_ord) -> 1 as G -> infinity, while
estimators of a different objective (grpo / rloo / reinforce) plateau.

Usage:
    python -m experiments.imagenet_localization.scripts.collect_cosine_summary
    python experiments/imagenet_localization/scripts/collect_cosine_summary.py \
        --results_dir <dir> --output <path>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import sys
from pathlib import Path

# Also runnable as a plain file path, which needs the repo root on sys.path
# before the package import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.imagenet_localization import paths  # noqa: E402

DEFAULT_RESULTS = paths.results_dir()
DEFAULT_OUTPUT = os.path.join(paths.experiment_dir(), "docs", "cosine_tailrl_vs_population.json")

RL_METHODS = ("tailrl", "binary_maxrl", "grpo", "rloo", "reinforce")

CAVEATS = [
    "All source runs are seed 43 only; no across-seed error bars.",
    "The high-G curves (cosine_check_population, cosine_check_population_xl) use only "
    "50-100 val images and 2-3 trials, so the ~0.96 plateau at G >= 65536 "
    "rests on a small sample.",
    "TailRL cosine saturates near 0.96 rather than reaching 1.0 between G=65536 "
    "and G=262144. This is most likely the finite-sample floor of the "
    "estimator plus the K=50 coordinate discretisation, not a failure of the "
    "N -> infinity identity, but it is an empirical asymptote below 1.",
    "binary_maxrl reports cosine exactly 0.000 at every G and every epoch. A "
    "true cosine is essentially never identically zero across 27 cells; this "
    "looks like a degenerate (zeroed) advantage/gradient rather than a "
    "meaningful orthogonality result. Verify before using it in a figure.",
    "Gradients are taken over the 4 classification heads only (model.heads); "
    "the shared backbone is excluded, per gradient_analysis.py.",
]


def _run_name(path: str, results_dir: str) -> str:
    rel = os.path.relpath(path, results_dir)
    return rel.split(os.sep)[0]


def _source_path(path: str, results_dir: str) -> str:
    """Path to record for a source file: repo-relative when the results tree
    lives inside the repo, otherwise relative to the results tree itself.
    """
    rel = os.path.relpath(path, paths.repo_root())
    if rel.startswith(os.pardir + os.sep):
        rel = os.path.relpath(path, results_dir)
    return rel


def collect(results_dir: str) -> dict:
    pattern = os.path.join(results_dir, "**", "cosine_vs_G*ref_tailrl_population.json")
    files = sorted(glob.glob(pattern, recursive=True))

    sources: list[dict] = []
    by_epoch: dict[str, dict] = {}
    tailrl_points: dict[int, dict] = {}

    for path in files:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skip] {path}: {exc}")
            continue

        res = d.get("results", {})
        epoch = d.get("epoch")
        run = _run_name(path, results_dir)
        sources.append({
            "path": _source_path(path, results_dir),
            "run": run,
            "epoch": epoch,
            "n_images": d.get("n_images"),
            "n_trials": d.get("n_trials"),
            "ref_method": d.get("ref_method"),
            "methods": sorted(res.keys()),
        })

        # Per-epoch, all-methods table (from the multi-method runs).
        if len(res) > 1:
            key = f"epoch_{epoch}"
            slot = by_epoch.setdefault(key, {
                "epoch": epoch,
                "n_images": d.get("n_images"),
                "n_trials": d.get("n_trials"),
                "source_run": run,
                "methods": {},
            })
            for m in RL_METHODS:
                if m not in res:
                    continue
                slot["methods"][m] = {
                    str(G): {
                        "cos_mean": res[m][G].get("mean"),
                        "cos_std": res[m][G].get("std"),
                        "cos_values": res[m][G].get("values"),
                        "mag_ratio_mean": res[m][G].get("mag_ratio_mean"),
                        "mag_ratio_std": res[m][G].get("mag_ratio_std"),
                    }
                    for G in sorted(res[m], key=lambda x: int(x))
                }

        # Combined TailRL curve across every source (highest n_images wins ties).
        for G, r in res.get("tailrl", {}).items():
            g = int(G)
            cand = {
                "G": g,
                "cos_mean": r.get("mean"),
                "cos_std": r.get("std"),
                "cos_values": r.get("values"),
                "mag_ratio_mean": r.get("mag_ratio_mean"),
                "epoch": epoch,
                "n_images": d.get("n_images"),
                "n_trials": d.get("n_trials"),
                "source_run": run,
            }
            prev = tailrl_points.get(g)
            if prev is None or (cand["n_images"] or 0) > (prev["n_images"] or 0):
                tailrl_points[g] = cand

    return {
        "description": (
            "Gradient cosine similarity between RL policy-gradient estimators at "
            "G rollouts and the supervised J_ord (tailrl_population) gradient, on "
            "ImageNet single-box localization."
        ),
        "claim_under_test": (
            "tailrl_population is the exact N -> infinity limit of TailRL for the "
            "max-over-GT IoU reward, so cos(grad_TailRL(G), grad_J_ord) -> 1 as G "
            "grows, while estimators of a different objective plateau."
        ),
        "reference_objective": "tailrl_population",
        "gradient_scope": "model.heads only (backbone excluded)",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "experiments/imagenet_localization/scripts/collect_cosine_summary.py",
        "caveats": CAVEATS,
        "sources": sources,
        "tailrl_vs_population": {
            "note": "Combined across source runs; where a G appears in more "
                    "than one run the entry with more images is kept.",
            "points": [tailrl_points[g] for g in sorted(tailrl_points)],
        },
        "all_methods_by_epoch": by_epoch,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS,
                    help="Tree of run directories to search (default: $TAILRL_RESULTS_DIR)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Summary JSON to write (default: <experiment>/docs/cosine_tailrl_vs_population.json)")
    args = ap.parse_args()

    summary = collect(args.results_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    pts = summary["tailrl_vs_population"]["points"]
    print(f"wrote {args.output}")
    print(f"  sources           : {len(summary['sources'])}")
    print(f"  TailRL curve points  : {len(pts)}  "
          f"(G={pts[0]['G']}..{pts[-1]['G']})" if pts else "  TailRL curve: empty")
    print(f"  all-method epochs : {sorted(summary['all_methods_by_epoch'])}")


if __name__ == "__main__":
    main()
