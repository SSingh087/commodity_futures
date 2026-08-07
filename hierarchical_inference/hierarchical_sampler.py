"""
hierarchical_sampler.py — Stage 3 Layer 2 (hierarchical population inference)
================================================================================

Reuses Layer 1's ALREADY-COMPUTED per-day posterior samples. No surrogate
calls happen here at all — this is pure importance-reweighting over
existing samples, so it's cheap regardless of how many days you have.

log p({F_obs,d} | Lambda) = sum_d log[ mean_i p(theta_d,i | Lambda) ]

where theta_d,i are day d's Layer 1 posterior samples (drawn under a flat
prior, so the reweighting ratio collapses to just p(theta|Lambda) itself
— see population_model.py's docstring for the derivation).
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from nessai.flowsampler import FlowSampler
from nessai.model import Model as NessaiModel
from nessai.utils import setup_logger

from population_model import log_prob_population, LAMBDA_NAMES, LAMBDA_BOUNDS, N_LAMBDA


class HierarchicalModel(NessaiModel):
    def __init__(self, day_posteriors: list[np.ndarray]):
        """day_posteriors: list of (N_d, 8) arrays, one per day, from
        Layer 1's run_inference.py output (posterior_samples.npy)."""
        self.day_posteriors = day_posteriors
        self.names = list(LAMBDA_NAMES)
        self.bounds = {name: list(b) for name, b in zip(LAMBDA_NAMES, LAMBDA_BOUNDS)}
        super().__init__()

    def log_prior(self, x) -> np.ndarray:
        log_p = np.zeros(x.size)
        log_p[~self.in_bounds(x)] = -np.inf
        return log_p

    def _log_likelihood_one(self, lam: np.ndarray) -> float:
        total = 0.0
        for theta_samples in self.day_posteriors:
            log_p_i = log_prob_population(theta_samples, lam)  # (N_d,)
            # log-mean-exp: log( mean_i p(theta_i|Lambda) )
            log_mean = logsumexp(log_p_i) - np.log(len(log_p_i))
            total += log_mean
        return total

    def log_likelihood(self, x) -> np.ndarray:
        x = np.atleast_1d(x)
        out = np.empty(x.size)
        for i, xi in enumerate(x):
            lam = np.array([xi[name] for name in self.names])
            out[i] = self._log_likelihood_one(lam)
        return out if x.size > 1 else out[0]


def run_hierarchical_nessai(day_posteriors: list[np.ndarray],
                              output_dir: str | Path, seed: int = 0,
                              n_live: int = 1000, n_pool: int = 4,
                              resume: bool = False) -> np.ndarray:
    setup_logger(output=str(output_dir))
    model = HierarchicalModel(day_posteriors)
    fs = FlowSampler(model, output=str(output_dir), resume=resume,
                      seed=seed, nlive=n_live, n_pool=n_pool)
    fs.run()
    posterior = fs.posterior_samples
    chain = np.stack([posterior[name] for name in LAMBDA_NAMES], axis=-1)
    return chain