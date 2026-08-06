"""
run_inference.py — Stage 2 (options) CLI runner
====================================================

Mirrors SchwartzSmithFWDcheck/run_inference.py exactly in structure —
same output layout, same coverage_results.json + KDE plot pattern — just
wired to the options ensemble/likelihood instead of the curve surrogate.

Usage:
    python run_inference.py \\
        --checkpoint_dir /data/wiay/postgrads/shashwat/COMM_DATA/results/checkpoints/SchwartzSmithFWD_OOF/<tag> \\
        --seed 3 \\
        --output_root /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L2 \\
        --n_contracts 10 \\
        --n_live 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from surrogate import OptionsEnsembleWrapper, PARAM_NAMES
from gen_pseudo_data import make_random_injection
from sampler import run_nessai
from plot_utils import plot_posterior_kde, credible_interval_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True,
                     help="Path to one trained Stage 2 ensemble's checkpoint dir "
                          "(must contain model_*.pt or best_model.pt and norm_stats.npz).")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_contracts", type=int, default=10,
                     help="Number of simulated observed option contracts for this "
                          "synthetic 'day' — analogous to number of maturities on "
                          "a futures curve.")
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_pool", type=int, default=4)
    ap.add_argument("--ci", type=float, default=0.90)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    model_tag = checkpoint_dir.name

    run_dir = Path(args.output_root) / model_tag / f"inj_{args.seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_tag}")
    print(f"Injection seed: {args.seed}")
    print(f"Output dir: {run_dir}")

    surrogate = OptionsEnsembleWrapper.load(checkpoint_dir)

    observation = make_random_injection(surrogate, seed=args.seed,
                                          n_contracts=args.n_contracts)

    chain = run_nessai(
        observation["V_obs"], observation["contracts"], surrogate,
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
                             n_contracts=args.n_contracts, n_live=args.n_live)
    with open(run_dir / "coverage_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "_meta"}, indent=2))

    plot_posterior_kde(chain, observation["theta_true"], PARAM_NAMES,
                        run_dir / "posterior_kde.png", ci=args.ci)


if __name__ == "__main__":
    main()