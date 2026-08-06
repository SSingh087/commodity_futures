"""
collect_pp_plot.py — aggregate many injections into a PP plot
==================================================================

What a PP plot actually checks
-------------------------------
For a well-calibrated PE pipeline, if you run N independent injections
and for each one compute the "insertion percentile" of theta_true within
its own marginal posterior (i.e. what fraction of posterior samples fall
below the true value), those N percentiles should themselves be
UNIFORMLY distributed on [0, 1] — for every parameter independently.

Why: theta_true's location within its own recovered posterior is, under
correct calibration, no more likely to sit at the 10th percentile than
the 90th, across many repeated injections. Concretely: at any confidence
level x%, YOU should find theta_true inside the recovered x% credible
interval in x% of injections — no more, no less. A PP plot just makes
this same statement visually and continuously, sweeping over ALL
confidence levels at once instead of checking one fixed CI.

If you plot the empirical CDF of those percentiles against the
theoretical uniform CDF, a well-calibrated pipeline gives a curve that
hugs the diagonal y=x. Consistent deviation ABOVE the diagonal at low x
means the posteriors are systematically too narrow/overconfident (true
values falling near the edges more often than they should); deviation
BELOW means posteriors are too wide/underconfident. The grey band is the
expected sampling scatter for N injections under the null hypothesis of
correct calibration (via a binomial/Kolmogorov-Smirnov-style bound), so
that you can tell a genuine problem apart from noise.

This script implements that directly (no bilby dependency) so you can
see exactly how it works, then shows the one-liner bilby equivalent at
the bottom for when you just want the result. Both operate on identical
inputs (insertion percentiles per parameter, already computed and saved
into each injection's coverage_results.json by run_stage3_layer1.py).

Usage:
    python collect_pp_plot.py \\
        --results_dir /path/to/CALLIBRATION_S3_L1/<model_tag> \\
        --out_path /path/to/plots/pp_plot_<model_tag>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from surrogate import PARAM_NAMES


def load_insertion_percentiles(results_dir: Path) -> dict[str, list[float]]:
    """
    Walk all inj_XXXX/coverage_results.json under results_dir and collect
    the insertion_percentile for each parameter across all injections.
    """
    percentiles = {name: [] for name in PARAM_NAMES}
    injection_dirs = sorted(results_dir.glob("inj_*"))
    if not injection_dirs:
        raise FileNotFoundError(f"No inj_* subfolders found under {results_dir}")

    for inj_dir in injection_dirs:
        result_path = inj_dir / "coverage_results.json"
        if not result_path.exists():
            print(f"[warn] missing {result_path}, skipping")
            continue
        results = json.loads(result_path.read_text())
        for name in PARAM_NAMES:
            percentiles[name].append(results[name]["insertion_percentile"])

    n = len(injection_dirs)
    print(f"Loaded {n} injections from {results_dir}")
    return percentiles


def plot_pp(percentiles: dict[str, list[float]], out_path: str | Path,
            model_tag: str = ""):
    """
    Custom PP plot: empirical CDF of insertion percentiles vs uniform,
    one line per parameter, with a combined-N binomial confidence band.
    """
    n_params = len(percentiles)
    n_injections = len(next(iter(percentiles.values())))

    fig, ax = plt.subplots(figsize=(7, 7))

    xs = np.linspace(0, 1, 1000)
    # Expected sampling scatter band for n_injections draws from Uniform(0,1),
    # via the binomial distribution at each x (same logic bilby uses):
    # at true CDF value x, the number of samples <= x is Binomial(n, x).
    lower = stats.binom.ppf(0.005, n_injections, xs) / n_injections
    upper = stats.binom.ppf(0.995, n_injections, xs) / n_injections
    ax.fill_between(xs, lower, upper, color="gray", alpha=0.2,
                     label="99% band (n={})".format(n_injections))
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    # Track max KS-style deviation per parameter for a quick numeric summary
    summary = {}
    cmap = plt.get_cmap("tab10")
    for i, (name, vals) in enumerate(percentiles.items()):
        vals = np.sort(np.asarray(vals))
        empirical_cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, empirical_cdf, color=cmap(i % 10), lw=1.5, label=name)

        ks_stat, ks_pvalue = stats.kstest(vals, "uniform")
        summary[name] = dict(ks_stat=float(ks_stat), ks_pvalue=float(ks_pvalue))

    ax.set_xlabel("Credible interval (theoretical)")
    ax.set_ylabel("Fraction of injections within CI (empirical)")
    ax.set_title(f"PP plot — {model_tag}" if model_tag else "PP plot")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PP plot to {out_path}")

    print("\nPer-parameter KS test against Uniform(0,1) (low p-value = miscalibrated):")
    for name, s in summary.items():
        flag = "  <-- check this" if s["ks_pvalue"] < 0.05 else ""
        print(f"  {name:12s}  D={s['ks_stat']:.3f}  p={s['ks_pvalue']:.3f}{flag}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                     help="Directory containing inj_XXXX subfolders for one model, "
                          "e.g. .../CALLIBRATION_S3_L1/<model_tag>")
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    model_tag = results_dir.name
    percentiles = load_insertion_percentiles(results_dir)
    plot_pp(percentiles, args.out_path, model_tag=model_tag)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# The bilby shortcut, for reference / cross-check against the manual version
# above. bilby.core.result.make_pp_plot() expects a list of bilby Result
# objects, each carrying .posterior (a DataFrame of samples) and
# .injection_parameters (a dict of true values) — NOT nessai's raw output
# directly, so you'd wrap each injection's chain like this first:
#
#   import bilby
#   import pandas as pd
#
#   results = []
#   for inj_dir in sorted(results_dir.glob("inj_*")):
#       chain = np.load(inj_dir / "posterior_samples.npy")
#       cov = json.loads((inj_dir / "coverage_results.json").read_text())
#       posterior_df = pd.DataFrame(chain, columns=PARAM_NAMES)
#       injection_parameters = {name: cov[name]["true"] for name in PARAM_NAMES}
#       r = bilby.core.result.Result(
#           label=inj_dir.name,
#           posterior=posterior_df,
#           search_parameter_keys=PARAM_NAMES,
#           injection_parameters=injection_parameters,
#       )
#       results.append(r)
#
#   bilby.core.result.make_pp_plot(
#       results, filename=str(out_path), keys=PARAM_NAMES,
#       confidence_interval=[0.68, 0.90, 0.99],
#   )
#
# This does essentially the same computation as plot_pp() above internally
# (insertion percentile per parameter per injection -> PP curve), just with
# bilby's own styling/CI-band conventions. Worth running both once and
# comparing — they should agree.
# ---------------------------------------------------------------------------