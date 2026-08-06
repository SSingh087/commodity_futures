"""
gen_pseudo_data.py — Data generation / loading for Stage 3 Layer 1
======================================================================

Unchanged in spirit from before: this module's ONLY job is to produce
F_obs (from wherever), and know nothing about the sampler.

New: make_random_injection(seed) draws a DIFFERENT theta_true per seed,
sampled from the prior. This is what a proper PP-plot coverage test
needs — reusing one fixed theta_true for every injection would only test
calibration at one point in parameter space, not across it. Contrast
with make_default_injection, which is kept for quick single-run sanity
checks where you want a reproducible fixed answer to eyeball.
"""

from __future__ import annotations

import numpy as np

from surrogate import SurrogateWrapper, PARAM_NAMES, PRIOR_BOUNDS  # noqa: F401


def synthesize_observation(
    theta_true: np.ndarray,
    surrogate: SurrogateWrapper,
    sigma_obs: float,
    seed: int = 0,
) -> dict:
    """
    Inject a known theta_true, add Gaussian noise at the assumed
    observation-noise level -> synthetic F_obs.
    """
    rng = np.random.default_rng(seed)
    F_clean = surrogate.predict(theta_true)
    noise = rng.normal(0.0, sigma_obs, size=F_clean.shape)
    F_obs = F_clean + noise

    return dict(
        F_obs=F_obs,
        sigma_obs=sigma_obs,
        maturities=surrogate.maturities,
        theta_true=theta_true,
        source="synthetic_injection",
        seed=seed,
    )


def make_random_injection(surrogate: SurrogateWrapper, seed: int,
                            sigma_obs: float = 0.02) -> dict:
    """
    Draw theta_true ~ prior (same seed used for both the draw and the
    noise realisation, so the whole injection is fully reproducible from
    the seed alone — the standard convention for an injection campaign).
    """
    rng = np.random.default_rng(seed)
    theta_true = np.array([
        rng.uniform(*PRIOR_BOUNDS[name]) for name in PARAM_NAMES
    ])
    return synthesize_observation(theta_true, surrogate, sigma_obs, seed=seed)


def make_default_injection(surrogate: SurrogateWrapper, seed: int = 1) -> dict:
    """Fixed, reproducible theta_true for quick single-run sanity checks."""
    theta_true = np.array([1.2, 0.03, 0.35, 0.20, 0.4, 0.0, 0.0, 3.5])
    sigma_obs = 0.02
    return synthesize_observation(theta_true, surrogate, sigma_obs, seed=seed)


def load_real_day(
    csv_path: str,
    date: str,
    maturities: np.ndarray,
    sigma_obs: float,
) -> dict:
    """STUB — see previous version's docstring; fill in with real market data."""
    raise NotImplementedError(
        "Point this at your actual market-data source once ready. Must "
        "return: F_obs, sigma_obs, maturities, theta_true=None, "
        "source='real:<date>'."
    )