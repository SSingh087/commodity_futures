"""
gen_pseudo_data.py — Stage 2 (options) data generation
==========================================================

One "day" = one shared (kappa, sigma_chi, sigma_xi, rho) plus SEVERAL
observed option contracts (each with its own known moneyness/T_option/
T_futures/r) all priced under that same theta. This mirrors having
several maturities on one futures curve, except here it's several
option contracts instead of curve points.
"""

from __future__ import annotations

import numpy as np

from surrogate import (
    OptionsEnsembleWrapper, PARAM_NAMES, PRIOR_BOUNDS,
    CONTRACT_RANGES, TAU_EXTRA,
)


def sample_contracts(rng: np.random.Generator, n_contracts: int) -> np.ndarray:
    """Draw n_contracts random (moneyness, T_option, T_futures, r) tuples —
    stands in for "today's actual traded option contracts", known/observed,
    not inferred."""
    moneyness = rng.uniform(*CONTRACT_RANGES["moneyness"], size=n_contracts)
    T_option = rng.uniform(*CONTRACT_RANGES["T_option"], size=n_contracts)
    T_futures = T_option + rng.uniform(*TAU_EXTRA, size=n_contracts)
    r = rng.uniform(*CONTRACT_RANGES["r"], size=n_contracts)
    return np.stack([moneyness, T_option, T_futures, r], axis=1)  # (n_contracts, 4)


def make_random_injection(surrogate: OptionsEnsembleWrapper, seed: int,
                            n_contracts: int = 10) -> dict:
    """
    Draw theta_true ~ prior, draw n_contracts random known contract terms,
    price them via the ensemble MEAN (the "clean" price), add Gaussian
    noise using the ensemble's own predicted total sigma per contract —
    i.e. the injected noise level matches what the surrogate itself
    believes its uncertainty is, the self-consistent choice absent a
    separately-known real bid-ask spread.
    """
    rng = np.random.default_rng(seed)

    theta_true = np.array([rng.uniform(*PRIOR_BOUNDS[name]) for name in PARAM_NAMES])
    contracts = sample_contracts(rng, n_contracts)

    mu_clean, sigma_pred = surrogate.predict_batch(theta_true[None, :], contracts)
    mu_clean = mu_clean[0]      # (n_contracts,)
    sigma_pred = sigma_pred[0]  # (n_contracts,)

    noise = rng.normal(0.0, sigma_pred)
    V_obs = mu_clean + noise

    return dict(
        V_obs=V_obs,
        sigma_obs=sigma_pred,     # per-contract, not a single scalar
        contracts=contracts,
        theta_true=theta_true,
        source="synthetic_injection",
        seed=seed,
        n_contracts=n_contracts,
    )