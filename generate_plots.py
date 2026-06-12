"""
generate_plots.py
=================
Self-contained demo: generates all illustrative plots without training.

Uses:
  - Analytical Schwartz-Smith model (numpy only, no PyTorch)
  - emcee for MCMC posterior (no surrogate needed)

Run:
    python demo/generate_plots.py

Produces all plots in results/plots/
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.schwartz_smith import (
    SSParams, SchwartzSmithModel,
    log_posterior, price_european_call_on_futures
)
# from src.inference.mcmc import BayesianCalibrator, coverage_test, HierarchicalCalibrator
from src.utils.plotting import (
    plot_forward_curves, plot_parameter_sensitivity,
    plot_analytical_gradients, plot_svd_reconstruction,
    plot_posterior_corner, plot_calibration_coverage,
    plot_posterior_vs_pointestimate, plot_uncertainty_propagation,
    plot_parameter_time_series, plot_mc_convergence,
    plot_volatility_smile, set_style
)

OUTDIR = Path("results/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

MATURITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
BASE = SSParams(kappa=1.5, mu_xi=0.04, sigma_chi=0.28,
                sigma_xi=0.18, rho=-0.30, lambda_chi=0.10,
                chi0=0.05, xi0=3.50)
THETA_TRUE = BASE.to_array()
CHI0, XI0 = BASE.chi0, BASE.xi0

def save(fig, name):
    path = OUTDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {path}")

set_style()

# ════════════════════════════════════════════════════════════════
print("\n[Stage 1] Forward curve structure")
# ════════════════════════════════════════════════════════════════

print("  1/10  Forward curves (kappa & rho variation)")
fig = plot_forward_curves(MATURITIES, [])
plt.show()
# save(fig, "01_forward_curves.png")

# print("  2/10  Parameter sensitivity")
# fig = plot_parameter_sensitivity(MATURITIES, BASE)
# save(fig, "02_parameter_sensitivity.png")

# print("  3/10  Analytical gradients (differential loss targets)")
# fig = plot_analytical_gradients(MATURITIES, BASE)
# save(fig, "03_analytical_gradients.png")

# print("  4/10  SVD compression (ROQ basis analogy)")
# # Generate a small set of curves for the SVD demo
# rng = np.random.default_rng(42)
# n_demo = 3000
# theta_demo = rng.uniform(
#     [0.3, 0.01, 0.10, 0.08, -0.7, -0.3, -0.3, 3.0],
#     [3.0, 0.08, 0.50, 0.35,  0.7,  0.3,  0.3, 4.0],
#     size=(n_demo, 8)
# )
# F_demo = np.array([
#     SchwartzSmithModel(SSParams(
#         kappa=t[0], mu_xi=t[1], sigma_chi=t[2], sigma_xi=t[3],
#         rho=t[4], lambda_chi=t[5], chi0=t[6], xi0=t[7]
#     )).forward_curve(MATURITIES)
#     for t in theta_demo
# ])
# fig = plot_svd_reconstruction(MATURITIES, F_demo)
# save(fig, "04_svd_reconstruction.png")

# # ════════════════════════════════════════════════════════════════
# print("\n[Stage 2] Options on futures")
# # ════════════════════════════════════════════════════════════════

# print("  5/10  MC convergence (why surrogate needed)")
# fig = plot_mc_convergence(
#     BASE,
#     {"moneyness": 1.0, "T_option": 0.5, "T_futures": 1.0},
#     path_counts=[500, 1000, 5000, 10000, 50000]
# )
# save(fig, "05_mc_convergence.png")

# print("  6/10  Implied volatility smile")
# moneyness_grid = np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
# fig = plot_volatility_smile(BASE, [1.0, 2.0], moneyness_grid, T_option=0.5, n_paths=15000)
# save(fig, "06_volatility_smile.png")

# # ════════════════════════════════════════════════════════════════
# print("\n[Stage 3] Bayesian calibration")
# # ════════════════════════════════════════════════════════════════

# print("  7/10  Running MCMC (this takes ~2 min)...")
# sigma_obs = 0.02
# F_obs = (SchwartzSmithModel(BASE).forward_curve(MATURITIES)
#          * np.exp(sigma_obs * rng.standard_normal(len(MATURITIES))))

# calibrator = BayesianCalibrator(MATURITIES, sigma_obs, chi0=CHI0, xi0=XI0)
# sampler = calibrator.run_mcmc(F_obs, n_walkers=48, n_steps=3000, n_burn=800)
# samples  = calibrator.get_posterior_samples(sampler)
# theta_ls = calibrator._least_squares_init(F_obs)

# print(f"     {len(samples)} posterior samples obtained")
# print("     Posterior medians vs truth:")
# names = ["kappa", "mu_xi", "sigma_chi", "sigma_xi", "rho", "lambda_chi"]
# for i, n in enumerate(names):
#     lo, med, hi = np.percentile(samples[:, i], [5, 50, 95])
#     print(f"       {n:12s}: {med:.4f} [{lo:.4f},{hi:.4f}]  true={THETA_TRUE[i]:.4f}")

# print("  8/10  Corner plot")
# fig = plot_posterior_corner(samples, theta_true=THETA_TRUE)
# save(fig, "07_posterior_corner.png")

# print("  9/10  Bayesian vs least-squares comparison")
# fig = plot_posterior_vs_pointestimate(
#     MATURITIES, F_obs, samples, theta_ls, THETA_TRUE, CHI0, XI0
# )
# save(fig, "08_bayesian_vs_pointestimate.png")

# print("  10/10  Two-level uncertainty propagation")
# fig = plot_uncertainty_propagation(MATURITIES, samples, chi0=CHI0, xi0=XI0, n_draw=300)
# save(fig, "09_uncertainty_propagation.png")

# # ════════════════════════════════════════════════════════════════
# print("\n[Stage 3 — Hierarchical] Calibration over trading days")
# # ════════════════════════════════════════════════════════════════

# print("  Running 20-day hierarchical calibration...")
# n_days = 20
# theta_series = np.empty((n_days, 6))
# theta = THETA_TRUE.copy()
# daily_std = np.array([0.04, 0.001, 0.006, 0.004, 0.025, 0.006])
# F_series = np.empty((n_days, len(MATURITIES)))

# for d in range(n_days):
#     theta = theta + daily_std * rng.standard_normal(6)
#     lo = np.array([0.1, 0.0, 0.05, 0.05, -0.9, -0.5])
#     hi = np.array([5.0, 0.10, 0.60, 0.40, 0.9, 0.5])
#     theta = np.clip(theta, lo, hi)
#     theta_series[d] = theta
#     p = SSParams.from_array(theta, CHI0, XI0)
#     F = SchwartzSmithModel(p).forward_curve(MATURITIES)
#     F_series[d] = F * np.exp(sigma_obs * rng.standard_normal(len(MATURITIES)))

# # Run per-day MCMC (short chains for demo speed)
# hier = HierarchicalCalibrator(calibrator)
# samplers_hier = hier.run(
#     F_series, n_walkers=32, n_steps=800, n_burn=200
# )
# summary = hier.population_summary(samplers_hier)

# fig = plot_parameter_time_series(
#     summary["param_median_series"],
#     summary["param_std_series"],
# )
# fig.savefig(OUTDIR / "10_parameter_time_series.png", dpi=150, bbox_inches="tight")
# plt.close("all")
# print(f"  Saved {OUTDIR / '10_parameter_time_series.png'}")

# # ════════════════════════════════════════════════════════════════
# print("\n[Coverage / PP-plot]")
# # ════════════════════════════════════════════════════════════════

# print("  Running injection-recovery coverage test (50 trials)...")
# cov = coverage_test(
#     calibrator,
#     n_trials=50,
#     coverage_levels=[0.68, 0.90, 0.95],
#     n_walkers=32, n_steps=1200, n_burn=300,
# )
# cov["n_trials"] = 50
# fig = plot_calibration_coverage(cov)
# fig.savefig(OUTDIR / "11_coverage_ppplot.png", dpi=150, bbox_inches="tight")
# plt.close("all")
# print(f"  Saved {OUTDIR / '11_coverage_ppplot.png'}")
# print("  Coverage achieved:")
# for lv, ac in zip(cov["levels"], cov["achieved_coverage"]):
#     print(f"    {lv*100:.0f}% CI: {ac*100:.1f}%")

# # ════════════════════════════════════════════════════════════════
# print(f"\n✓ All plots saved to {OUTDIR.resolve()}")
# print("  Files:")
# for p in sorted(OUTDIR.glob("*.png")):
#     print(f"    {p.name}")
