import sys, os
sys.path.insert(0, os.path.abspath('../'))
from __plotting_imports__ import *

from schwartz_smith import SSParams, SchwartzSmithModel

MATURITIES = np.linspace(0.1, 10.0, 100)

# set_style()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
kappas = [0.3, 1.0, 2.0, 4.0]
for k, kap in enumerate(kappas):
    p = SSParams(kappa=kap, mu_xi=0.04, sigma_chi=0.3,
                    sigma_xi=0.2, rho=-0.3, lambda_chi=0.1,
                    chi0=0.1, xi0=3.5)
    F = SchwartzSmithModel(p).log_forward_curve(MATURITIES)
    ax.plot(MATURITIES, F, color=PALETTE[k], label=f"$\kappa$ = {kap}")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Futures Price $\mathcal{F}(T)$")
ax.set_title("Effect of Mean-Reversion Speed $\kappa$")
ax.legend(title="$\kappa$ (1/year)")
ax.set_xscale("log")

ax = axes[1]
RHOS = [-0.8, -0.3, 0.0, 0.5, 0.8]
for k, rho in enumerate(RHOS):
    p = SSParams(kappa=1.5, mu_xi=0.04, sigma_chi=0.3,
                    sigma_xi=0.2, rho=rho, lambda_chi=0.1,
                    chi0=0.05, xi0=3.5)
    F = SchwartzSmithModel(p).log_forward_curve(MATURITIES)
    ax.plot(MATURITIES, F, color=PALETTE[k], label=f"$\\rho$ = {rho:+.1f}")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Futures Price $\mathcal{F}(T)$")
ax.set_title("Effect of Correlation $\\rho$")
ax.set_xscale("log")
ax.legend(title="$\\rho$")
fig.tight_layout()

plt.savefig("../plots/SchwartzSmithModel.png", dpi=300)
plt.show()