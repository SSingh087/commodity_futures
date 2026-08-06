"""
run_stage3_layer1.py — CLI-driven single (model, injection) runner
======================================================================

Designed to be called once per (checkpoint_dir, seed) pair from a DAG
node, or directly for a one-off test. Output goes into a self-naming
subfolder so many parallel runs never collide:

    <output_root>/<model_tag>/inj_<seed:04d>/
        posterior_samples.npy
        coverage_results.json     <- includes insertion percentiles, the
                                      raw ingredient for the PP plot
        posterior_kde.png

Usage:
    python run_stage3_layer1.py \\
        --checkpoint_dir /path/to/checkpoints/SchwartzSmithFWD/<tag> \\
        --seed 3 \\
        --output_root /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L1 \\
        --n_live 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from surrogate import SurrogateWrapper, PARAM_NAMES
from gen_pseudo_data import make_random_injection
from sampler import run_nessai
from plot_utils import plot_posterior_kde, credible_interval_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True,
                     help="Path to one trained Stage 1 model's checkpoint dir "
                          "(must contain best_model.pt and norm_stats.npz).")
    ap.add_argument("--seed", type=int, required=True,
                     help="Injection seed — determines both theta_true and "
                          "the noise realisation. One seed = one PP-plot point.")
    ap.add_argument("--output_root", required=True,
                     help="Root output dir. A subfolder named after the "
                          "checkpoint dir's own name is created automatically, "
                          "and within that, one subfolder per seed.")
    ap.add_argument("--sigma_obs", type=float, default=0.02)
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_pool", type=int, default=4)
    ap.add_argument("--ci", type=float, default=0.90)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    model_tag = checkpoint_dir.name  # e.g. bs2048_a0.10_b0.10_..._319402-0

    run_dir = Path(args.output_root) / model_tag / f"inj_{args.seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_tag}")
    print(f"Injection seed: {args.seed}")
    print(f"Output dir: {run_dir}")

    surrogate = SurrogateWrapper.load(checkpoint_dir)

    observation = make_random_injection(surrogate, seed=args.seed,
                                          sigma_obs=args.sigma_obs)

    chain = run_nessai(
        observation["F_obs"], observation["sigma_obs"], surrogate,
        output_dir=run_dir / "nessai_run",
        seed=args.seed,
        n_live=args.n_live,
        n_pool=args.n_pool,
        resume=args.resume,
    )

    np.save(run_dir / "posterior_samples.npy", chain)

    results = credible_interval_check(chain, observation["theta_true"],
                                        PARAM_NAMES, ci=args.ci)
    results["_meta"] = dict(model_tag=model_tag, seed=args.seed,
                             sigma_obs=args.sigma_obs, n_live=args.n_live)
    with open(run_dir / "coverage_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "_meta"}, indent=2))

    plot_posterior_kde(chain, observation["theta_true"], PARAM_NAMES,
                        run_dir / "posterior_kde.png", ci=args.ci)


if __name__ == "__main__":
    main()