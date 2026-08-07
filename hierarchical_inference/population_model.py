"""
population_model.py — Stage 3 Layer 2 population distribution
==================================================================

p(theta | Lambda), where theta = (kappa, mu_xi, sigma_chi, sigma_xi, rho,
lambda_chi, chi0, xi0), same 8 params as Layer 1.

Correlated block (kappa, sigma_chi, sigma_xi, rho): each transformed to
an unconstrained space (log for the positive vol/rate-like params, logit
for the bounded correlation), then modelled as a joint 4D Normal with
mean vector mu and covariance Sigma. This captures population-level
correlation/degeneracy between them (e.g. kappa-sigma_chi), same role as
correlated population hyperparameters in a GW mass/spin population model.

Independent block (mu_xi, lambda_chi, chi0, xi0): plain Normal(mean, std)
each, no cross-correlation modelled — these are more nuisance/state-like
than physically coupled.

Lambda layout (flat vector, for the sampler):
    [0:4]   mu_corr        - means of the 4 transformed correlated params
                              order: ln(kappa), ln(sigma_chi), ln(sigma_xi), logit(rho)
    [4:14]  L_flat         - 10 entries: lower-triangular Cholesky factor
                              of the 4x4 covariance Sigma (4 diag + 6 off-diag)
    [14:18] mu_indep       - means of mu_xi, lambda_chi, chi0, xi0
    [18:22] sigma_indep    - stds of the same, in order
"""

from __future__ import annotations

import numpy as np

CORR_PARAM_NAMES = ["kappa", "sigma_chi", "sigma_xi", "rho"]
INDEP_PARAM_NAMES = ["mu_xi", "lambda_chi", "chi0", "xi0"]
PARAM_NAMES = ["kappa", "mu_xi", "sigma_chi", "sigma_xi", "rho", "lambda_chi",
               "chi0", "xi0"]  # matches Layer 1's theta ordering

N_CORR = 4
N_TRIL = N_CORR * (N_CORR + 1) // 2  # 10

RHO_SCALE = 0.999  # keep logit(rho) finite; rho in (-RHO_SCALE, RHO_SCALE)


# ── Transforms: physical <-> unconstrained ─────────────────────────────

def to_unconstrained_corr(theta_corr: np.ndarray) -> np.ndarray:
    """theta_corr: (..., 4) physical [kappa, sigma_chi, sigma_xi, rho]
    -> (..., 4) unconstrained [ln kappa, ln sigma_chi, ln sigma_xi, logit(rho/scale)]"""
    kappa, sigma_chi, sigma_xi, rho = np.moveaxis(theta_corr, -1, 0)
    rho_clipped = np.clip(rho / RHO_SCALE, -0.999999, 0.999999)
    out = np.stack([
        np.log(kappa), np.log(sigma_chi), np.log(sigma_xi),
        np.log((1 + rho_clipped) / (1 - rho_clipped)),  # logit
    ], axis=-1)
    return out


def from_unconstrained_corr(u: np.ndarray) -> np.ndarray:
    """Inverse of to_unconstrained_corr."""
    ln_k, ln_sc, ln_sx, logit_rho = np.moveaxis(u, -1, 0)
    rho = RHO_SCALE * (2.0 / (1.0 + np.exp(-logit_rho)) - 1.0)  # sigmoid -> (-1,1)
    return np.stack([np.exp(ln_k), np.exp(ln_sc), np.exp(ln_sx), rho], axis=-1)


def jacobian_log_det_corr(theta_corr: np.ndarray) -> np.ndarray:
    """log|d(unconstrained)/d(physical)| for the change of variables, needed
    to convert a density in unconstrained space into one in physical space.
    d(ln x)/dx = 1/x for the three log params; d(logit)/d(rho) picks up a
    1/(1-rho'^2)-type factor for the bounded one."""
    kappa, sigma_chi, sigma_xi, rho = np.moveaxis(theta_corr, -1, 0)
    rho_clipped = np.clip(rho / RHO_SCALE, -0.999999, 0.999999)
    log_det = (
        -np.log(kappa) - np.log(sigma_chi) - np.log(sigma_xi)
        - np.log(RHO_SCALE) - np.log(1 - rho_clipped ** 2)
        + np.log(2.0)
    )
    return log_det


# ── Lambda packing/unpacking ─────────────────────────────────────────

N_LAMBDA = N_CORR + N_TRIL + len(INDEP_PARAM_NAMES) * 2  # 4+10+4+4 = 22

def unpack_lambda(lam: np.ndarray):
    mu_corr = lam[0:4]
    L_flat = lam[4:14]
    mu_indep = lam[14:18]
    sigma_indep = lam[18:22]

    L = np.zeros((4, 4))
    tril_idx = np.tril_indices(4)
    L[tril_idx] = L_flat
    # Diagonal must be positive for a valid Cholesky factor — enforce via
    # softplus-like exponentiation on the diagonal entries specifically.
    diag_idx = np.diag_indices(4)
    L[diag_idx] = np.exp(L[diag_idx])

    return mu_corr, L, mu_indep, sigma_indep


LAMBDA_BOUNDS = (
    # mu_corr: reasonable range in unconstrained space (roughly covers the
    # Layer 1 prior box once log/logit transformed)
    [(-2.5, 1.7)] * 1 +   # ln(kappa),   kappa in ~(0.08, 5.5)
    [(-3.5, 0.0)] * 1 +   # ln(sigma_chi)
    [(-3.5, 0.0)] * 1 +   # ln(sigma_xi)
    [(-6.0, 6.0)] * 1 +   # logit(rho)
    # L_flat: diagonal entries stored as log(std), off-diag as raw covariance terms
    [(-3.0, 1.5)] * 4 +   # log-diagonal of L (population scatter per param)
    [(-2.0, 2.0)] * 6 +   # off-diagonal L entries (correlation structure)
    # mu_indep / sigma_indep for mu_xi, lambda_chi, chi0, xi0
    [(0.0, 0.10)] +       # mu_xi mean
    [(-0.5, 0.5)] +       # lambda_chi mean
    [(-0.5, 0.5)] +       # chi0 mean
    [(2.5, 5.0)] +        # xi0 mean
    [(0.001, 0.05)] +     # mu_xi std
    [(0.001, 0.3)] +      # lambda_chi std
    [(0.001, 0.3)] +      # chi0 std
    [(0.001, 0.5)]        # xi0 std
)
LAMBDA_NAMES = (
    ["mu_ln_kappa", "mu_ln_sigma_chi", "mu_ln_sigma_xi", "mu_logit_rho"]
    + [f"L_{i}" for i in range(N_TRIL)]
    + ["mu_mu_xi", "mu_lambda_chi", "mu_chi0", "mu_xi0"]
    + ["sigma_mu_xi", "sigma_lambda_chi", "sigma_chi0", "sigma_xi0"]
)
assert len(LAMBDA_BOUNDS) == N_LAMBDA == len(LAMBDA_NAMES)


# ── Population density and sampling ────────────────────────────────────

def log_prob_population(theta: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """
    theta: (..., 8) physical params, order matches PARAM_NAMES
    lam:   (22,) flat Lambda vector
    Returns: (...,) log p(theta | Lambda)
    """
    mu_corr, L, mu_indep, sigma_indep = unpack_lambda(lam)
    Sigma = L @ L.T
    Sigma_inv = np.linalg.inv(Sigma)
    log_det_Sigma = 2 * np.sum(np.log(np.diag(L)))

    idx = {name: i for i, name in enumerate(PARAM_NAMES)}
    theta_corr = theta[..., [idx["kappa"], idx["sigma_chi"], idx["sigma_xi"], idx["rho"]]]
    theta_indep = theta[..., [idx["mu_xi"], idx["lambda_chi"], idx["chi0"], idx["xi0"]]]

    u = to_unconstrained_corr(theta_corr)                    # (..., 4)
    diff = u - mu_corr
    maha = np.einsum("...i,ij,...j->...", diff, Sigma_inv, diff)
    log_p_corr = (
        -0.5 * maha - 0.5 * log_det_Sigma - 2.0 * np.log(2 * np.pi)
        + jacobian_log_det_corr(theta_corr)
    )

    log_p_indep = -0.5 * np.sum(
        ((theta_indep - mu_indep) / sigma_indep) ** 2
        + 2 * np.log(sigma_indep) + np.log(2 * np.pi),
        axis=-1,
    )

    return log_p_corr + log_p_indep


def sample_population(lam: np.ndarray, rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Draw n samples of theta ~ p(theta | Lambda). Returns (n, 8)."""
    mu_corr, L, mu_indep, sigma_indep = unpack_lambda(lam)

    z = rng.standard_normal((n, 4))
    u = mu_corr[None, :] + z @ L.T
    theta_corr = from_unconstrained_corr(u)  # (n, 4)

    theta_indep = mu_indep[None, :] + rng.standard_normal((n, 4)) * sigma_indep[None, :]

    theta = np.zeros((n, 8))
    idx = {name: i for i, name in enumerate(PARAM_NAMES)}
    theta[:, [idx["kappa"], idx["sigma_chi"], idx["sigma_xi"], idx["rho"]]] = theta_corr
    theta[:, [idx["mu_xi"], idx["lambda_chi"], idx["chi0"], idx["xi0"]]] = theta_indep
    return theta


def make_default_lambda_true() -> np.ndarray:
    """A representative Lambda_true for injection-recovery testing — modest
    population scatter, mild kappa/sigma_chi anti-correlation to make the
    recovery test non-trivial."""
    mu_corr = np.array([np.log(1.2), np.log(0.30), np.log(0.20), np.log(3.0)])
    # last entry above is logit(rho) target ~ log(3) corresponds to rho~0.5;
    # recompute properly:
    mu_corr[3] = np.log((1 + 0.4 / RHO_SCALE) / (1 - 0.4 / RHO_SCALE))

    # Modest scatter, mild kappa-sigma_chi anti-correlation
    std = np.array([0.25, 0.20, 0.20, 0.5])
    corr = np.array([
        [1.0, -0.3, 0.0, 0.0],
        [-0.3, 1.0, 0.1, 0.0],
        [0.0, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Sigma = np.outer(std, std) * corr
    L = np.linalg.cholesky(Sigma)

    L_flat = np.zeros(N_TRIL)
    tril_idx = np.tril_indices(4)
    L_flat_full = L.copy()
    diag_idx = np.diag_indices(4)
    L_flat_full[diag_idx] = np.log(L_flat_full[diag_idx])  # store diag as log
    L_flat = L_flat_full[tril_idx]

    mu_indep = np.array([0.03, 0.0, 0.0, 3.5])
    sigma_indep = np.array([0.01, 0.15, 0.1, 0.3])

    return np.concatenate([mu_corr, L_flat, mu_indep, sigma_indep])