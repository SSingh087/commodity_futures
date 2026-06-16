"""
test_mc_functions.py
====================
Visual test of simulate_state_paths and price_european_call_on_futures.

Paste your Schwartz_Smith.py in the same directory (or adjust the import),
then run:
    python test_mc_functions.py

Produces four plots in a single figure:
  1. Fan plot  — chi and xi sample paths (like a GW time-domain waveform)
  2. Terminal distributions of chi_T and xi_T
  3. Option payoff distribution and MC convergence (stderr vs n_paths)
  4. Implied vol smile across moneyness levels
"""

import sys, os
sys.path.insert(0, os.path.abspath('../'))
from plotting_scripts.__plotting_imports__ import *

from schwartz_smith import SSParams, SchwartzSmithModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

from schwartz_smith import (
    SSParams,
    SchwartzSmithModel,
    simulate_state_paths,
    price_european_call_on_futures,
)

# ── Baseline parameters (crude-oil-ish) ───────────────────────────────────
BASE = SSParams(
    kappa       = 1.5,    # mean-reversion: ~8-month half-life
    mu_xi       = 0.04,   # long-run drift 4 %/yr
    sigma_chi   = 0.28,   # short-term vol
    sigma_xi    = 0.18,   # long-term vol
    rho         = -0.30,  # typical negative correlation
    lambda_chi  = 0.10,   # market price of short-term risk
    chi0        = 0.05,   # start slightly above equilibrium
    xi0         = 3.50,   # ln(~33 $/bbl)
)

# ── Simulation settings ───────────────────────────────────────────────────
N_PATHS_FULL  = 20_000   # for price estimation
N_PATHS_FAN   = 30       # for the fan plot
N_STEPS       = 252      # daily steps (1 year)
T_OPTION      = 1.0      # option expires in 1 year
T_FUTURES     = 1.5      # futures delivers in 1.5 years
SEED          = 42

# ─────────────────────────────────────────────────────────────────────────
# FIGURE SETUP
# ─────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

ax_chi   = fig.add_subplot(gs[0, 0])   # chi sample paths
ax_xi    = fig.add_subplot(gs[0, 1])   # xi sample paths
ax_term  = fig.add_subplot(gs[0, 2])   # terminal distributions
ax_pay   = fig.add_subplot(gs[1, 0])   # payoff histogram
ax_conv  = fig.add_subplot(gs[1, 1])   # MC convergence
ax_smile = fig.add_subplot(gs[1, 2])   # implied vol smile

BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"
t_grid = np.linspace(0, T_OPTION, N_STEPS + 1)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 1 & 2  –  Fan plots for chi and xi
# (GW analogy: each path is like a noise realisation; the fan width is
#  determined by the process PSD — Lorentzian for chi, flat for xi)
# ═════════════════════════════════════════════════════════════════════════
chi_fan, xi_fan = simulate_state_paths(BASE, T_OPTION,
                                       n_paths=N_PATHS_FAN,
                                       n_steps=N_STEPS, seed=SEED)

# Plot individual paths (thin, transparent)
for i in range(N_PATHS_FAN):
    ax_chi.plot(t_grid, chi_fan[i], lw=0.6, alpha=0.35, color=BLUE)
    ax_xi.plot(t_grid, xi_fan[i], lw=0.6, alpha=0.35, color=ORANGE)

# Overlay analytic mean ± 1σ envelope
# chi: E[chi_t] = chi0 * exp(-kappa*t)
# Var[chi_t]   = sigma_chi^2 / (2*kappa) * (1 - exp(-2*kappa*t))
k, s_c, s_x = BASE.kappa, BASE.sigma_chi, BASE.sigma_xi
mean_chi = BASE.chi0 * np.exp(-k * t_grid)
std_chi  = s_c * np.sqrt((1 - np.exp(-2 * k * t_grid)) / (2 * k + 1e-12))
mean_xi  = BASE.xi0 + BASE.mu_xi * t_grid
std_xi   = s_x * np.sqrt(t_grid)

ax_chi.plot(t_grid, mean_chi, "k--", lw=1.8, label="Analytic mean")
ax_chi.fill_between(t_grid, mean_chi - std_chi, mean_chi + std_chi,
                    alpha=0.20, color=BLUE, label="±1σ envelope")
ax_chi.set_xlabel("Time (years)"); ax_chi.set_ylabel("χ(t)")
ax_chi.set_title("Short-term deviation χ(t)\n"
                 "[OU = coloured noise, Lorentzian PSD]", fontsize=9)
ax_chi.legend(fontsize=8)

ax_xi.plot(t_grid, mean_xi, "k--", lw=1.8, label="Analytic mean")
ax_xi.fill_between(t_grid, mean_xi - std_xi, mean_xi + std_xi,
                   alpha=0.20, color=ORANGE, label="±1σ envelope")
ax_xi.set_xlabel("Time (years)"); ax_xi.set_ylabel("ξ(t)")
ax_xi.set_title("Long-term equilibrium ξ(t)\n"
                "[GBM = white-noise integral, flat PSD]", fontsize=9)
ax_xi.legend(fontsize=8)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 3  –  Terminal distributions  (check MC matches analytic Gaussian)
# (GW analogy: verify your noise simulator matches the target PSD)
# ═════════════════════════════════════════════════════════════════════════
chi_full, xi_full = simulate_state_paths(BASE, T_OPTION,
                                         n_paths=N_PATHS_FULL,
                                         n_steps=N_STEPS, seed=SEED)
chi_T = chi_full[:, -1]
xi_T  = xi_full[:, -1]

# Analytic terminal moments
mean_chi_T = BASE.chi0 * np.exp(-k * T_OPTION)
std_chi_T  = s_c * np.sqrt((1 - np.exp(-2 * k * T_OPTION)) / (2 * k))
mean_xi_T  = BASE.xi0 + BASE.mu_xi * T_OPTION
std_xi_T   = s_x * np.sqrt(T_OPTION)

x_chi = np.linspace(chi_T.min(), chi_T.max(), 200)
x_xi  = np.linspace(xi_T.min(), xi_T.max(), 200)

ax_term.hist(chi_T, bins=80, density=True, alpha=0.55, color=BLUE,  label="MC $\\chi_T$")
ax_term.hist(xi_T,  bins=80, density=True, alpha=0.55, color=ORANGE, label="MC $\\xi_T$")
ax_term.plot(x_chi, norm.pdf(x_chi, mean_chi_T, std_chi_T),
             "b-", lw=2, label="Analytic $\\chi_T$")
ax_term.plot(x_xi,  norm.pdf(x_xi,  mean_xi_T,  std_xi_T),
             "-",  lw=2, color="darkorange", label="Analytic $\\xi_T$")
ax_term.set_xlabel("State value at T = 1 yr")
ax_term.set_ylabel("Density")
ax_term.set_title("Terminal distributions\n[MC vs analytic Gaussian — sanity check]",
                  fontsize=9)
ax_term.legend(fontsize=7)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 4  –  Payoff histogram  (what the NN surrogate must learn to price)
# ═════════════════════════════════════════════════════════════════════════
model = SchwartzSmithModel(BASE)
F_0_fut = model.futures_price(np.array([T_FUTURES]))[0]
K_atm   = 1.0 * F_0_fut                   # at-the-money strike

tau_rem = T_FUTURES - T_OPTION
e1      = np.exp(-BASE.kappa * tau_rem)
A_val   = model.A(np.array([tau_rem]))[0]
ln_F_T  = e1 * chi_T + xi_T + A_val
F_T     = np.exp(ln_F_T)

payoffs  = np.maximum(F_T - K_atm, 0.0)
discount = np.exp(-0.05 * T_OPTION)
mc_price = discount * payoffs.mean()
mc_std   = discount * payoffs.std() / np.sqrt(N_PATHS_FULL)

# Separate zero payoffs (option expires OTM) from positive payoffs
positive_payoffs = payoffs[payoffs > 0]

ax_pay.hist(positive_payoffs, bins=60, density=True, color=GREEN, alpha=0.75,
            label=f"ITM payoffs ({100*len(positive_payoffs)/N_PATHS_FULL:.1f}% of paths)")
ax_pay.axvline(mc_price / discount, color=RED, lw=2, ls="--",
               label=f"E[payoff] = {mc_price/discount:.3f}")
ax_pay.set_xlabel("Payoff (ITM paths only)")
ax_pay.set_ylabel("Density")
ax_pay.set_title(f"Option payoff distribution\n"
                 f"Price = {mc_price:.4f}  ±  {mc_std:.4f}  (N={N_PATHS_FULL:,})",
                 fontsize=9)
ax_pay.legend(fontsize=8)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 5  –  MC convergence  (stderr ~ 1/√N — the label noise you must model)
# (GW analogy: this is the SNR scaling; surrogate must account for this noise)
# ═════════════════════════════════════════════════════════════════════════
n_list    = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
prices_n  = []
stderrs_n = []

for n in n_list:
    price_n, stderr_n = price_european_call_on_futures(
        BASE, strike_ratio=1.0,
        T_option=T_OPTION, T_futures=T_FUTURES,
        n_paths=n, n_steps=N_STEPS, seed=SEED
    )
    prices_n.append(price_n)
    stderrs_n.append(stderr_n)

prices_n  = np.array(prices_n)
stderrs_n = np.array(stderrs_n)
n_arr     = np.array(n_list, dtype=float)

ax_conv.errorbar(n_arr, prices_n, yerr=1.96 * stderrs_n,
                 fmt="o-", color=BLUE, capsize=4, label="MC price ± 1.96σ")
ax_conv.axhline(prices_n[-1], ls="--", color="gray", lw=1, label="Best estimate")
ax_conv.set_xscale("log")
ax_conv.set_xlabel("Number of MC paths (log scale)")
ax_conv.set_ylabel("Option price")
ax_conv.set_title("MC convergence: stderr ~ 1/√N\n"
                  "[This is the aleatoric label noise the ensemble must model]",
                  fontsize=9)
ax_conv.legend(fontsize=8)

# Inset: log(stderr) vs log(N) should be slope -0.5
ax_ins = ax_conv.inset_axes([0.55, 0.55, 0.42, 0.38])
ax_ins.loglog(n_arr, stderrs_n, "o-", color=ORANGE, ms=4)
fit = np.polyfit(np.log(n_arr), np.log(stderrs_n), 1)
ax_ins.loglog(n_arr, np.exp(np.polyval(fit, np.log(n_arr))),
              "k--", lw=1, label=f"slope={fit[0]:.2f}")
ax_ins.set_title("log stderr vs log N", fontsize=7)
ax_ins.legend(fontsize=6)


# ═════════════════════════════════════════════════════════════════════════
# PANEL 6  –  Implied volatility smile
# (The surface the NN surrogate must learn — analogue of the waveform manifold)
# ═════════════════════════════════════════════════════════════════════════
def bs_call(F, K, T, r, sigma):
    """Black-Scholes call price (used to back out implied vol)."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def implied_vol(price, F, K, T, r, tol=1e-6):
    from scipy.optimize import brentq
    try:
        return brentq(lambda s: bs_call(F, K, T, r, s) - price, 0.01, 5.0, xtol=tol)
    except Exception:
        return np.nan

moneyness = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
T_futs    = [T_FUTURES, 2.5]

for j, T_fut in enumerate(T_futs):
    F_atm = model.futures_price(np.array([T_fut]))[0]
    ivols = []
    for m in moneyness:
        p, _ = price_european_call_on_futures(
            BASE, strike_ratio=m,
            T_option=T_OPTION, T_futures=T_fut,
            n_paths=10_000, n_steps=N_STEPS, seed=j * 100
        )
        iv = implied_vol(p, F_atm, m * F_atm, T_OPTION, 0.05)
        ivols.append(iv)
    color = [BLUE, GREEN][j]
    ax_smile.plot(moneyness, np.array(ivols) * 100, "o-",
                  color=color, label=f"T_fut = {T_fut}y")

ax_smile.axvline(1.0, ls="--", color="gray", lw=1, alpha=0.6)
ax_smile.set_xlabel("Moneyness  K / $\\mathcal{F}(0, T_{\\text{fut}})$")
ax_smile.set_ylabel("Implied volatility (%)")
ax_smile.set_title("Implied vol smile\n"
                   "[The manifold Stage 2 surrogate must interpolate]", fontsize=9)
ax_smile.legend(fontsize=8)


# ─────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Schwartz-Smith MC Functions — Visual Test\n"
    f"κ={BASE.kappa}, $\\sigma_\\chi={BASE.sigma_chi}, $\\sigma_\\xi={BASE.sigma_xi}, ρ={BASE.rho}",
    fontsize=11, fontweight="bold", y=1.01
)
fig.savefig("../plots/test_mc_functions.png", dpi=150, bbox_inches="tight")

plt.show()