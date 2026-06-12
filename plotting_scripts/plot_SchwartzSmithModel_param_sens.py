import sys, os
sys.path.insert(0, os.path.abspath('../'))
from __plotting_imports__ import *

from schwartz_smith import SSParams, SchwartzSmithModel

MATURITIES = np.linspace(0.1, 10.0, 100)
BASE_PARAMS = SSParams(kappa=1.5, mu_xi=0.04, sigma_chi=0.3,
                       sigma_xi=0.2, rho=-0.3, lambda_chi=0.1,
                       chi0=0.05, xi0=3.5
                       )
fig, axes = plt.subplots(2, 3, figsize=(12, 10))
axes = axes.flatten()
model0 = SchwartzSmithModel(BASE_PARAMS)
F0 = model0.log_forward_curve(MATURITIES)

param_info = [
    ("kappa",     "$\\kappa$",         0.5, BASE_PARAMS.kappa),
    ("sigma_chi", "$\\sigma_\\chi$",       0.05, BASE_PARAMS.sigma_chi),
    ("sigma_xi",  "$\\sigma_\\xi$",       0.05, BASE_PARAMS.sigma_xi),
    ("rho",       "$\\rho$",         0.15, BASE_PARAMS.rho),
    ("lambda_chi","$\\lambda_\\chi$",       0.1,  BASE_PARAMS.lambda_chi),
    ("mu_xi",     "$\\mu_\\xi$",       0.01, BASE_PARAMS.mu_xi),
]
for ax, (attr, label, delta, base_val) in zip(axes, param_info):
    for sign, col, ls in [(+1, SURROGATE_RED, "-"), (-1, GW_BLUE, "--")]:
        p2 = SSParams(
            kappa=BASE_PARAMS.kappa, mu_xi=BASE_PARAMS.mu_xi,
            sigma_chi=BASE_PARAMS.sigma_chi, sigma_xi=BASE_PARAMS.sigma_xi,
            rho=BASE_PARAMS.rho, lambda_chi=BASE_PARAMS.lambda_chi,
            chi0=BASE_PARAMS.chi0, xi0=BASE_PARAMS.xi0,
        )
        setattr(p2, attr, base_val + sign * delta)
        F2 = SchwartzSmithModel(p2).log_forward_curve(MATURITIES)
        ax.plot(MATURITIES, F2, color=col, ls=ls,
                label=f"{label} {'+' if sign>0 else '-'} {delta:.2g}")

    ax.plot(MATURITIES, F0, "k-", lw=2, label="Baseline", zorder=5)
    ax.set_title(f"Sensitivity to {label}")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("$\\mathcal{F}$(T)")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("../plots/SchwartzSmithModel_param_sens.png", dpi=300)
plt.show()