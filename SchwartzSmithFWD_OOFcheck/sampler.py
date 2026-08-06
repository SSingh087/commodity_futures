"""
sampler.py — Stage 2 (options) PE using nessai
===================================================

Same pattern as Stage 1's sampler.py, but the likelihood sums Gaussian
residuals across MULTIPLE observed option contracts (each with its own
sigma from the ensemble's predictive uncertainty), all sharing one
theta = (kappa, sigma_chi, sigma_xi, rho).
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

from surrogate import OptionsEnsembleWrapper, PARAM_NAMES, PRIOR_BOUNDS


class Stage2CalibrationModel(NessaiModel):
    """
    Likelihood combines the surrogate's OWN predicted uncertainty
    (epistemic + aleatoric, from OptionsEnsembleWrapper.predict_batch)
    as the per-contract noise term — same idea as folding in both
    detector noise and waveform-model uncertainty into one effective
    sigma when the model itself isn't perfectly known.
    """

    def __init__(self, V_obs: np.ndarray, contracts: np.ndarray,
                 surrogate: OptionsEnsembleWrapper,
                 obs_sigma: np.ndarray | None = None):
        self.V_obs = V_obs
        self.contracts = contracts
        self.surrogate = surrogate
        # obs_sigma: the ADDITIONAL observation noise (e.g. bid-ask spread),
        # separate from the surrogate's own predictive uncertainty. If not
        # given, only the surrogate's predictive sigma is used.
        self.obs_sigma = obs_sigma

        self.names = list(PARAM_NAMES)
        self.bounds = {name: list(PRIOR_BOUNDS[name]) for name in PARAM_NAMES}
        super().__init__()

    def log_prior(self, x) -> np.ndarray:
        log_p = np.zeros(x.size)
        log_p[~self.in_bounds(x)] = -np.inf
        return log_p

    def log_likelihood(self, x) -> np.ndarray:
        # nessai calls this once with a SINGLE point first (to verify the
        # function is vectorised), which collapses x to a 0-d structured
        # element and makes thetas come out as shape (4,) instead of
        # (1, 4). predict_batch's np.repeat(..., axis=0) requires 2D, so
        # force it here regardless of which call pattern triggered this.
        thetas = np.atleast_2d(
            np.stack([x[name] for name in self.names], axis=-1)
        )  # (n, 4), n=1 for the single-point verification call
        mu_pred, sigma_pred = self.surrogate.predict_batch(thetas, self.contracts)
        # (n, n_contracts) each

        total_sigma = sigma_pred
        if self.obs_sigma is not None:
            total_sigma = np.sqrt(sigma_pred ** 2 + self.obs_sigma[None, :] ** 2)

        resid = (self.V_obs[None, :] - mu_pred) / total_sigma
        log_l = -0.5 * np.sum(resid ** 2, axis=-1)
        # nessai expects a scalar back for the single-point verification
        # call, and an (n,) array for real batched calls.
        return log_l[0] if log_l.shape[0] == 1 and np.ndim(x) == 0 else log_l


def run_nessai(V_obs: np.ndarray, contracts: np.ndarray,
                surrogate: OptionsEnsembleWrapper,
                output_dir: str | Path, seed: int = 0, resume: bool = False,
                n_live: int = 1000, n_pool: int = 4,
                obs_sigma: np.ndarray | None = None) -> np.ndarray:
    setup_logger(output=str(output_dir))

    model = Stage2CalibrationModel(V_obs, contracts, surrogate, obs_sigma=obs_sigma)

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