"""
sampler.py — Stage 3 Layer 1 PE using nessai
================================================

Unchanged in spirit: still knows nothing about where F_obs came from.
n_live/n_pool are now parameters (not hardcoded) so DAG jobs can tune
them per model if needed without editing this file.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
from nessai.flowsampler import FlowSampler
from nessai.model import Model as NessaiModel
from nessai.utils import setup_logger

from surrogate import SurrogateWrapper, PARAM_NAMES, PRIOR_BOUNDS, N_DIM


class Stage1CalibrationModel(NessaiModel):
    """
    nessai Model wrapping the Stage 1 surrogate as the likelihood.
    Gaussian residuals between F_obs and the surrogate's predicted curve —
    the direct analogue of a matched-filter likelihood with independent
    per-bin noise, except "bins" = maturities rather than frequencies.
    """

    def __init__(self, F_obs: np.ndarray, sigma_obs: float,
                 surrogate: SurrogateWrapper):
        self.F_obs = F_obs
        self.sigma_obs = sigma_obs
        self.surrogate = surrogate
        self.names = list(PARAM_NAMES)
        self.bounds = {name: list(PRIOR_BOUNDS[name]) for name in PARAM_NAMES}
        super().__init__()

    def log_prior(self, x) -> np.ndarray:
        log_p = np.zeros(x.size)
        log_p[~self.in_bounds(x)] = -np.inf
        return log_p

    def log_likelihood(self, x) -> np.ndarray:
        thetas = np.stack([x[name] for name in self.names], axis=-1)
        F_pred = self.surrogate.predict_batch(thetas)
        resid = (self.F_obs[None, :] - F_pred) / self.sigma_obs
        return -0.5 * np.sum(resid ** 2, axis=-1)


def run_nessai(F_obs: np.ndarray, sigma_obs: float, surrogate: SurrogateWrapper,
                output_dir: str | Path, seed: int = 0, resume: bool = False,
                n_live: int = 1000, n_pool: int = 4) -> np.ndarray:
    """
    Returns posterior samples as a plain (n_samples, 8) array in the same
    column order as PARAM_NAMES.
    """
    setup_logger(output=str(output_dir))

    model = Stage1CalibrationModel(F_obs, sigma_obs, surrogate)

    fs = FlowSampler(
        model,
        output=str(output_dir),
        resume=resume,
        seed=seed,
        nlive=n_live,
        n_pool=n_pool,
    )
    fs.run()

    posterior = fs.posterior_samples
    chain = np.stack([posterior[name] for name in PARAM_NAMES], axis=-1)
    return chain