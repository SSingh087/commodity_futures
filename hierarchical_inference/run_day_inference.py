"""
run_day_inference.py — CLI: Layer 1 PE for ONE pre-drawn Layer 2 day
=========================================================================

Unlike Layer 1's run_inference.py (which draws its OWN random theta_true
per seed), this reads a day's theta_true/F_obs that gen_population_injection.py
already generated and saved — because all D days must share consistency
with ONE Lambda_true, drawn together up front, not redrawn independently
per DAG node.

Usage:
    python run_day_inference.py \\
        --checkpoint_dir /path/to/SchwartzSmithFWD/<best_model> \\
        --day_dir /path/to/injection/day_0007 \\
        --output_root /path/to/layer1_per_day \\
        --n_live 1000 --n_pool 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from surrogate import SurrogateWrapper
from sampler import run_nessai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--day_dir", required=True,
                     help="Dir from gen_population_injection.py containing "
                          "theta_true.npy, F_obs.npy, meta.json for one day")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_pool", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    day_dir = Path(args.day_dir)
    day_tag = day_dir.name  # e.g. day_0007

    out_dir = Path(args.output_root) / day_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    F_obs = np.load(day_dir / "F_obs.npy")
    theta_true = np.load(day_dir / "theta_true.npy")
    meta = json.loads((day_dir / "meta.json").read_text())
    sigma_obs = meta["sigma_obs"]
    seed = meta["seed"]

    print(f"Day: {day_tag}  seed={seed}")

    surrogate = SurrogateWrapper.load(args.checkpoint_dir)

    chain = run_nessai(
        F_obs, sigma_obs, surrogate,
        output_dir=out_dir / "nessai_run",
        seed=seed, n_live=args.n_live, n_pool=args.n_pool,
        resume=args.resume,
    )

    np.save(out_dir / "posterior_samples.npy", chain)
    np.save(out_dir / "theta_true.npy", theta_true)  # copied through for convenience
    print(f"Saved posterior for {day_tag} to {out_dir}")


if __name__ == "__main__":
    main()