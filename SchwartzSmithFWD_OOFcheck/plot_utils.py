"""
plot_utils.py — shared plotting / coverage utilities for Stage 3
====================================================================

Used by both run_stage3_layer1.py (single-injection diagnostic plot) and
collect_pp_plot.py (aggregating many injections into a PP plot).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_posterior_kde(chain: np.ndarray, theta_true: np.ndarray,
                        param_names: list[str], out_path: str | Path,
                        ci: float = 0.90):
    """
    KDE version of the posterior plot — smooth density instead of raw
    histogram bars, with the true value, posterior median, and CI% bounds
    all annotated directly on each panel. Same role as a corner-plot
    marginal panel, just not yet the full 2D corner (that's a nice-to-have
    for later, noted but not urgent).
    """
    n_dim = len(param_names)
    n_cols = 4
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    for j, (ax, name) in enumerate(zip(axes, param_names)):
        samples = chain[:, j]
        lo, hi = np.quantile(samples, [lo_q, hi_q])
        median = np.median(samples)

        # KDE over a padded range so tails aren't clipped
        pad = 0.05 * (samples.max() - samples.min() + 1e-12)
        xs = np.linspace(samples.min() - pad, samples.max() + pad, 400)
        kde = gaussian_kde(samples)
        density = kde(xs)

        ax.plot(xs, density, color="steelblue", lw=1.8)
        ax.fill_between(xs, density, alpha=0.25, color="steelblue")
        ax.axvline(theta_true[j], color="crimson", lw=2, label="true")
        ax.axvline(median, color="black", lw=1.2, ls="--", label="median")
        ax.axvspan(lo, hi, color="gray", alpha=0.15, label=f"{int(ci*100)}% CI")

        ax.set_title(name, fontsize=11)
        ax.text(0.02, 0.95,
                 f"true={theta_true[j]:.3g}\nmed={median:.3g}\n"
                 f"CI=[{lo:.3g}, {hi:.3g}]",
                 transform=ax.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0.5))
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    for ax in axes[n_dim:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved KDE posterior plot to {out_path}")


def credible_interval_check(chain: np.ndarray, theta_true: np.ndarray,
                              param_names: list[str], ci: float = 0.90) -> dict:
    """
    Per-parameter coverage check AND the insertion percentile (fraction of
    posterior samples <= theta_true) — the insertion percentile is what
    actually feeds a PP plot; the covered/not-covered flag at one fixed CI
    is just a convenience summary for quick reading.
    """
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    results = {}
    for j, name in enumerate(param_names):
        samples = chain[:, j]
        lo, hi = np.quantile(samples, [lo_q, hi_q])
        median = np.median(samples)
        covered = bool(lo <= theta_true[j] <= hi)
        insertion_percentile = float(np.mean(samples <= theta_true[j]))
        results[name] = dict(
            true=float(theta_true[j]), median=float(median),
            ci_lo=float(lo), ci_hi=float(hi), covered=covered,
            insertion_percentile=insertion_percentile,
        )
    return results