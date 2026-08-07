"""
run_hierarchical.py — CLI: final Layer 2 node, recovers Lambda
=====================================================================

Runs after ALL day_XXXX inference nodes finish (enforced by the DAG's
PARENT/CHILD line). Globs their posterior_samples.npy files and runs the
hierarchical sampler over Lambda.

Usage:
    python run_hierarchical.py \\
        --day_posteriors_root /path/to/layer1_per_day \\
        --lambda_true_path /path/to/injection/lambda_true.npy \\
        --output_root /path/to/layer2_hierarchical \\
        --n_live 1000 --n_pool 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hierarchical_sampler import run_hierarchical_nessai
from population_model import LAMBDA_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day_posteriors_root", required=True,
                     help="Dir containing day_XXXX/posterior_samples.npy for each day")
    ap.add_argument("--lambda_true_path", default=None,
                     help="Optional — path to lambda_true.npy for a coverage check "
                          "(synthetic validation only; omit for real data)")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_pool", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(args.day_posteriors_root)
    day_dirs = sorted(root.glob("day_*"))
    if not day_dirs:
        raise FileNotFoundError(f"No day_* subfolders found under {root}")

    day_posteriors = []
    for d in day_dirs:
        p = d / "posterior_samples.npy"
        if not p.exists():
            print(f"[warn] missing {p}, skipping this day")
            continue
        day_posteriors.append(np.load(p))
    print(f"Loaded {len(day_posteriors)} day posteriors from {root}")

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_chain = run_hierarchical_nessai(
        day_posteriors, output_dir=out_dir / "nessai_run",
        seed=0, n_live=args.n_live, n_pool=args.n_pool, resume=args.resume,
    )
    np.save(out_dir / "lambda_posterior_samples.npy", lambda_chain)
    print(f"Saved Lambda posterior to {out_dir}/lambda_posterior_samples.npy")

    if args.lambda_true_path:
        lam_true = np.load(args.lambda_true_path)
        results = {}
        for j, name in enumerate(LAMBDA_NAMES):
            lo, hi = np.quantile(lambda_chain[:, j], [0.05, 0.95])
            median = np.median(lambda_chain[:, j])
            covered = bool(lo <= lam_true[j] <= hi)
            results[name] = dict(true=float(lam_true[j]), median=float(median),
                                   ci_lo=float(lo), ci_hi=float(hi), covered=covered)
        with open(out_dir / "lambda_coverage.json", "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
    else:
        print("No lambda_true_path given — skipping coverage check "
              "(expected for real data, not synthetic validation).")


if __name__ == "__main__":
    main()