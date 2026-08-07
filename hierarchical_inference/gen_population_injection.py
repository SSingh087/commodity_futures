"""
gen_population_injection.py — Stage 3 Layer 2 synthetic-day generator
==========================================================================

Draws Lambda_true, then D days of theta_d ~ p(theta|Lambda_true), then
uses the Layer 1 surrogate to generate each day's synthetic F_obs,d.
Running Layer 1 PE on each of these days (reuse run_inference.py, one
call per day) gives the per-day posteriors that hierarchical_sampler.py
then consumes to try to recover Lambda_true — the Layer 2 analogue of
Layer 1's single-injection recovery test, just one level up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from surrogate import SurrogateWrapper, PARAM_NAMES
from population_model import sample_population, make_default_lambda_true


def generate_days(surrogate: SurrogateWrapper, lam_true: np.ndarray,
                    n_days: int, sigma_obs: float = 0.02,
                    seed: int = 0) -> list[dict]:
    """Returns a list of length n_days, each a dict with theta_true_d,
    F_obs, sigma_obs, maturities, seed — same shape as Layer 1's
    synthesize_observation output, so run_inference.py needs zero changes
    to consume these (just point it at pre-drawn theta rather than
    drawing from the flat prior)."""
    rng = np.random.default_rng(seed)
    thetas = sample_population(lam_true, rng, n=n_days)  # (n_days, 8)

    days = []
    for d in range(n_days):
        theta_d = thetas[d]
        day_seed = seed * 100_000 + d
        day_rng = np.random.default_rng(day_seed)
        F_clean = surrogate.predict(theta_d)
        noise = day_rng.normal(0.0, sigma_obs, size=F_clean.shape)
        F_obs = F_clean + noise
        days.append(dict(
            day_index=d, seed=day_seed,
            theta_true=theta_d, F_obs=F_obs, sigma_obs=sigma_obs,
            maturities=surrogate.maturities,
        ))
    return days


def save_population_injection(lam_true: np.ndarray, days: list[dict],
                                 out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lambda_true.npy", lam_true)
    for day in days:
        d = day["day_index"]
        day_dir = out_dir / f"day_{d:04d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        np.save(day_dir / "theta_true.npy", day["theta_true"])
        np.save(day_dir / "F_obs.npy", day["F_obs"])
        with open(day_dir / "meta.json", "w") as f:
            json.dump(dict(seed=day["seed"], sigma_obs=day["sigma_obs"]), f, indent=2)
    print(f"Saved {len(days)} synthetic days + lambda_true to {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_days", type=int, default=30)
    ap.add_argument("--sigma_obs", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    surrogate = SurrogateWrapper.load(args.checkpoint_dir)
    lam_true = make_default_lambda_true()
    days = generate_days(surrogate, lam_true, args.n_days, args.sigma_obs, args.seed)
    save_population_injection(lam_true, days, args.out_dir)