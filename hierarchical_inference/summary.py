"""
plot_layer2_summary.py — the "tell me the story" plot for Layer 2
=======================================================================

Two figures:

1. day_evolution.png — one panel per physical parameter (kappa, sigma_chi,
   sigma_xi, rho get their own row since they're population-correlated;
   mu_xi, lambda_chi, chi0, xi0 below). X-axis = day index. For each day:
   the Layer 1 posterior median as a point, with a vertical bar for its
   90% CI — literally the same visual grammar as a GW parameter's
   posterior-over-time plot across a run of events. A shaded horizontal
   band shows the RECOVERED population distribution (mean +/- 1 sigma,
   from the Lambda posterior, transformed back into physical units) —
   so you can see at a glance whether per-day scatter is consistent with
   what the hierarchical fit inferred. If synthetic, the true theta_d
   values are overlaid as small crosses.

2. lambda_posterior.png — KDE posterior over all 22 Lambda components
   (reuses plot_utils.plot_posterior_kde's style), so you can see how
   tightly each population hyperparameter is actually constrained.

Usage:
    python plot_layer2_summary.py \\
        --day_posteriors_root /path/to/layer1_per_day \\
        --lambda_posterior_path /path/to/layer2_hierarchical/lambda_posterior_samples.npy \\
        --injection_dir /path/to/injection \\
        --out_dir /path/to/plots/layer2_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from surrogate import PARAM_NAMES
from population_model import (
    LAMBDA_NAMES, unpack_lambda, from_unconstrained_corr, CORR_PARAM_NAMES,
    INDEP_PARAM_NAMES,
)
from plot_utils import plot_posterior_kde


def load_day_posteriors(root: Path):
    day_dirs = sorted(root.glob("day_*"))
    days = []
    for d in day_dirs:
        p = d / "posterior_samples.npy"
        if not p.exists():
            continue
        chain = np.load(p)
        theta_true_path = d / "theta_true.npy"
        theta_true = np.load(theta_true_path) if theta_true_path.exists() else None
        days.append(dict(day_dir=d, chain=chain, theta_true=theta_true))
    return days


def population_band_physical(lambda_chain: np.ndarray, param_name: str,
                               n_draws: int = 2000, seed: int = 0):
    """
    Draw n_draws (mu, mu+/-1sigma) triples in PHYSICAL units for one
    parameter, by transforming samples from the Lambda posterior back
    through the population model's parameterisation. Returns
    (median_of_mean, lo, hi) as a representative band to plot.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lambda_chain), size=min(n_draws, len(lambda_chain)), replace=False)

    means = []
    for i in idx:
        lam = lambda_chain[i]
        mu_corr, L, mu_indep, sigma_indep = unpack_lambda(lam)
        if param_name in CORR_PARAM_NAMES:
            j = CORR_PARAM_NAMES.index(param_name)
            # mean and std in UNCONSTRAINED space for this component
            Sigma = L @ L.T
            mu_u, std_u = mu_corr[j], np.sqrt(Sigma[j, j])
            # map mean+/-1sigma in unconstrained space back to physical
            # by transforming the two endpoints (approximate, since the
            # transform is nonlinear, but fine for a visual band)
            u_lo, u_hi = mu_u - std_u, mu_u + std_u
            full_lo = mu_corr.copy(); full_lo[j] = u_lo
            full_hi = mu_corr.copy(); full_hi[j] = u_hi
            full_mid = mu_corr.copy()
            phys_mid = from_unconstrained_corr(full_mid)[j]
            phys_lo = from_unconstrained_corr(full_lo)[j]
            phys_hi = from_unconstrained_corr(full_hi)[j]
            means.append((phys_mid, phys_lo, phys_hi))
        else:
            j = INDEP_PARAM_NAMES.index(param_name)
            means.append((mu_indep[j], mu_indep[j] - sigma_indep[j],
                          mu_indep[j] + sigma_indep[j]))

    means = np.array(means)  # (n_draws, 3): mid, lo, hi
    return (np.median(means[:, 0]), np.median(means[:, 1]), np.median(means[:, 2]))


def plot_day_evolution(days: list[dict], lambda_chain: np.ndarray | None,
                         out_path: str | Path, ci: float = 0.90):
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    n_params = len(PARAM_NAMES)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    day_indices = np.arange(len(days))

    for j, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        medians = np.array([np.median(d["chain"][:, j]) for d in days])
        los = np.array([np.quantile(d["chain"][:, j], lo_q) for d in days])
        his = np.array([np.quantile(d["chain"][:, j], hi_q) for d in days])

        ax.errorbar(day_indices, medians,
                     yerr=[medians - los, his - medians],
                     fmt="o", ms=4, color="steelblue", ecolor="steelblue",
                     alpha=0.8, capsize=2, label=f"Layer 1 posterior ({int(ci*100)}% CI)")

        if days[0]["theta_true"] is not None:
            true_vals = np.array([d["theta_true"][j] for d in days])
            ax.scatter(day_indices, true_vals, marker="x", color="crimson",
                        s=40, zorder=5, label="true theta_d")

        if lambda_chain is not None:
            mid, lo, hi = population_band_physical(lambda_chain, name)
            ax.axhline(mid, color="darkorange", lw=1.5, ls="--",
                        label="recovered population mean")
            ax.axhspan(lo, hi, color="darkorange", alpha=0.15,
                        label="recovered population 1σ band")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Day index")
        if j == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("Per-day parameter evolution vs. recovered population distribution",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved day-evolution plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day_posteriors_root", required=True)
    ap.add_argument("--lambda_posterior_path", default=None)
    ap.add_argument("--injection_dir", default=None,
                     help="Not strictly needed if theta_true.npy already copied "
                          "into each day folder by run_day_inference.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ci", type=float, default=0.90)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = load_day_posteriors(Path(args.day_posteriors_root))
    if not days:
        raise FileNotFoundError(f"No day posteriors found under {args.day_posteriors_root}")
    print(f"Loaded {len(days)} days")

    lambda_chain = None
    if args.lambda_posterior_path:
        lambda_chain = np.load(args.lambda_posterior_path)
        print(f"Loaded Lambda posterior: {lambda_chain.shape}")

    plot_day_evolution(days, lambda_chain, out_dir / "day_evolution.png", ci=args.ci)

    if lambda_chain is not None:
        # Reuse the same KDE-plot style as Layer 1's posteriors, applied
        # to Lambda instead of theta. lambda_true is optional (only for
        # synthetic validation); pass zeros as a no-op if not available.
        lambda_true_path = Path(args.injection_dir) / "lambda_true.npy" if args.injection_dir else None
        if lambda_true_path and lambda_true_path.exists():
            lambda_true = np.load(lambda_true_path)
            plot_posterior_kde(lambda_chain, lambda_true, LAMBDA_NAMES,
                                 out_dir / "lambda_posterior.png", ci=args.ci)
        else:
            print("No lambda_true available — plotting Lambda posterior without "
                  "true-value markers (expected for real data).")
            # Minimal fallback plot without requiring a true value per component
            fig, axes = plt.subplots(4, 6, figsize=(22, 12))
            axes = axes.flatten()
            for j, (ax, name) in enumerate(zip(axes, LAMBDA_NAMES)):
                samples = lambda_chain[:, j]
                kde = gaussian_kde(samples)
                xs = np.linspace(samples.min(), samples.max(), 300)
                ax.plot(xs, kde(xs), color="steelblue")
                ax.fill_between(xs, kde(xs), alpha=0.25, color="steelblue")
                ax.set_title(name, fontsize=9)
            for ax in axes[len(LAMBDA_NAMES):]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / "lambda_posterior.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved Lambda posterior plot to {out_dir / 'lambda_posterior.png'}")


if __name__ == "__main__":
    main()