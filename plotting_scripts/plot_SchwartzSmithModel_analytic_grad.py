import sys, os
sys.path.insert(0, os.path.abspath('../'))
from __plotting_imports__ import *

from schwartz_smith import SSParams, SchwartzSmithModel

MATURITIES = np.linspace(0.1, 10.0, 100)
BASE_PARAMS = SSParams(kappa=1.5, mu_xi=0.04, sigma_chi=0.3,
                       sigma_xi=0.2, rho=-0.3, lambda_chi=0.1,
                       chi0=0.05, xi0=3.5
                       )

model = SchwartzSmithModel(BASE_PARAMS)
grads = model.all_gradients(MATURITIES)

labels = {
    "kappa":     r"$\partial F / \partial \kappa$",
    "sigma_chi": r"$\partial F / \partial \sigma_\chi$",
    "sigma_xi":  r"$\partial F / \partial \sigma_\xi$",
    "rho":       r"$\partial F / \partial \rho$",
}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for ax, (key, label) in zip(axes.flatten(), labels.items()):
    ax.plot(MATURITIES, grads[key], color=STAGE_COLORS[1], lw=2.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Maturity (years)")
    ax.fill_between(MATURITIES, grads[key], alpha=0.15, color=STAGE_COLORS[1])

# Set y-label on the left column
axes[0, 0].set_ylabel(r"$\partial \mathcal{F} / \partial \theta$  (price / parameter unit)")
axes[1, 0].set_ylabel(r"$\partial \mathcal{F} / \partial \theta$  (price / parameter unit)")

plt.tight_layout()
plt.savefig("../plots/SchwartzSmithModel_analytic_grad.png", dpi=300)
plt.show()