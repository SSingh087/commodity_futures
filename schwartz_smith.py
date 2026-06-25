"""
schwartz_smith.py
=================
Analytical implementation of the Schwartz-Smith (2000) two-factor commodity model.

The log spot price is decomposed as:
    ln S(t) = chi(t) + xi(t)

where:
    chi(t) : short-term mean-reverting factor   (Ornstein-Uhlenbeck)
    xi(t)  : long-term equilibrium factor       (Geometric Brownian Motion)


"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Parameter container ───────────────────────────────────────────────────────

@dataclass
class SSParams:
    """
    Parameters for the Schwartz-Smith two-factor model.

    Physical measure dynamics:
        dchi = -kappa * chi dt + sigma_chi dW_chi
        dxi  =  mu_xi         dt + sigma_xi  dW_xi
        dW_chi dW_xi = rho dt

    Risk-neutral adjustment: market price of risk lambda_chi for the short
    factor shifts the drift in the Q-measure.
    """
    kappa:      float = 1.5     # mean-reversion speed      (1/year), typical ~1-3
    mu_xi:      float = 0.05    # long-term log-price drift  (1/year)
    sigma_chi:  float = 0.30    # short-term vol             (1/sqrt(year))
    sigma_xi:   float = 0.20    # long-term vol              (1/sqrt(year))
    rho:        float = -0.30   # chi-xi correlation
    lambda_chi: float = 0.10    # market price of short-term risk
    lambda_xi:  float = 0.00    # market price of long-term risk (often set 0)
    chi0:       float = 0.00    # initial short-term factor
    xi0:        float = 3.50    # initial long-term factor  (ln S0 ~ 3.5 → S0~33)

    def to_array(self) -> np.ndarray:
        """Return parameter vector (excluding chi0, xi0 for surrogate input)."""
        return np.array([
            self.kappa, self.mu_xi, self.sigma_chi,
            self.sigma_xi, self.rho, self.lambda_chi
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray, chi0: float = 0.0, xi0: float = 3.5,
                   lambda_xi: float = 0.0) -> "SSParams":
        return cls(
            kappa=arr[0], mu_xi=arr[1], sigma_chi=arr[2],
            sigma_xi=arr[3], rho=arr[4], lambda_chi=arr[5],
            lambda_xi=lambda_xi, chi0=chi0, xi0=xi0
        )

    # Default prior bounds (used in data generation and MCMC)
    PRIOR_BOUNDS: dict = field(default_factory=lambda: {
        "kappa":      (0.1,  5.0),
        "mu_xi":      (0.0,  0.10),
        "sigma_chi":  (0.05, 0.60),
        "sigma_xi":   (0.05, 0.40),
        "rho":        (-0.9, 0.90),
        "lambda_chi": (-0.5, 0.50),
    })


# ── Forward curve (analytical) ────────────────────────────────────────────────

class SchwartzSmithModel:
    """
    Analytical Schwartz-Smith two-factor forward curve model.

    All quantities are in log-price space. The futures price for delivery
    at time T, observed at time t, given state (chi_t, xi_t) is:

        F(t, T) = exp( e^{-kappa*tau} * chi_t + xi_t + A(tau) )

    where tau = T - t and A(tau) is the risk-neutral drift correction.
    """

    def __init__(self, params: SSParams):
        self.p = params

    def A(self, tau: np.ndarray) -> np.ndarray:
        """
        Compute the deterministic drift correction A(tau) in the
        Schwartz-Smith futures pricing formula.

        A(tau) = (mu_xi - lambda_xi)*tau
                 - (lambda_chi/kappa)*(1 - e^{-kappa*tau})
                 + 0.5 * [sigma_chi^2*(1-e^{-2*kappa*tau})/(2*kappa)
                           + sigma_xi^2 * tau
                           + 2*rho*sigma_chi*sigma_xi*(1-e^{-kappa*tau})/kappa]

        Returns: array of shape (len(tau),)
        """
        p = self.p
        tau = np.asarray(tau, dtype=float)
        e1  = np.exp(-p.kappa * tau)
        e2  = np.exp(-2.0 * p.kappa * tau)

        drift_xi   = (p.mu_xi - p.lambda_xi) * tau
        drift_chi  = -(p.lambda_chi / p.kappa) * (1.0 - e1)

        var_chi    = (p.sigma_chi**2 / (2.0 * p.kappa)) * (1.0 - e2)
        var_xi     = p.sigma_xi**2 * tau
        cov_term   = 2.0 * p.rho * p.sigma_chi * p.sigma_xi / p.kappa * (1.0 - e1)

        return drift_xi + drift_chi + 0.5 * (var_chi + var_xi + cov_term)

    def log_futures_price(self, tau: np.ndarray) -> np.ndarray:
        """
        Compute ln F(t, T) for a vector of time-to-maturities tau = T - t.

        Returns: array of shape (len(tau),)
        """
        p   = self.p
        tau = np.asarray(tau, dtype=float)
        return np.exp(-p.kappa * tau) * p.chi0 + p.xi0 + self.A(tau)

    def futures_price(self, tau: np.ndarray) -> np.ndarray:
        """
        Compute F(t, T) = exp(log_futures_price(tau)).
        """
        return np.exp(self.log_futures_price(tau))

    def forward_curve(self, maturities: np.ndarray) -> np.ndarray:
        """
        Return the full forward curve F(T_1), ..., F(T_n).

        Parameters
        ----------
        maturities : array-like of time-to-maturities in years

        Returns
        -------
        np.ndarray of shape (n,) — futures prices
        """
        return self.futures_price(np.asarray(maturities))

    def log_forward_curve(self, maturities: np.ndarray) -> np.ndarray:
        """Return ln F(T_i) for each maturity."""
        return self.log_futures_price(np.asarray(maturities))

    # ── Analytical Greeks ────────────────────────────────────────────────────

    def dF_dkappa(self, tau: np.ndarray) -> np.ndarray:
        """
        Analytical derivative of the forward curve w.r.t. kappa.
        Used as the target for the differential loss in Stage 1.
        """
        p   = self.p
        tau = np.asarray(tau, dtype=float)
        F   = self.futures_price(tau)
        e1  = np.exp(-p.kappa * tau)
        e2  = np.exp(-2.0 * p.kappa * tau)

        # d(ln F)/d(kappa)
        # Derived term-by-term from ln F = e^{-kappa*tau}*chi0 + xi0 + A(tau):
        #   d/dkappa[e^{-kappa*tau}*chi0]           = -tau * e1 * chi0
        #   d/dkappa[-(lambda_chi/kappa)(1-e1)]     = lambda_chi*(1-e1)/kappa^2 - lambda_chi*tau*e1/kappa
        #   d/dkappa[sigma_chi^2*(1-e2)/(4*kappa)]  = sigma_chi^2/4 * [2*tau*e2/kappa - (1-e2)/kappa^2]
        #   d/dkappa[rho*sigma_chi*sigma_xi*(1-e1)/kappa] = rho*sc*sx * [tau*e1/kappa - (1-e1)/kappa^2]
        d_log_F  = -tau * e1 * p.chi0
        d_log_F += p.lambda_chi * (1.0 - e1) / p.kappa**2 \
                   - p.lambda_chi * tau * e1 / p.kappa
        d_log_F += p.sigma_chi**2 / 4.0 * (
            2.0 * tau * e2 / p.kappa - (1.0 - e2) / p.kappa**2
        )
        d_log_F += p.rho * p.sigma_chi * p.sigma_xi * (
            tau * e1 / p.kappa - (1.0 - e1) / p.kappa**2
        )
        return F * d_log_F

    def dF_dsigma_chi(self, tau: np.ndarray) -> np.ndarray:
        """Analytical dF/d(sigma_chi)."""
        p   = self.p
        tau = np.asarray(tau, dtype=float)
        F   = self.futures_price(tau)
        e1  = np.exp(-p.kappa * tau)
        e2  = np.exp(-2.0 * p.kappa * tau)

        d_log_F = (
            p.sigma_chi / (2.0 * p.kappa) * (1.0 - e2)  # ← σ_χ(1−e2)/(2κ)
            + p.rho * p.sigma_xi / p.kappa * (1.0 - e1)
        )
        return F * d_log_F

    def dF_dsigma_xi(self, tau: np.ndarray) -> np.ndarray:
        """Analytical dF/d(sigma_xi)."""
        p   = self.p
        tau = np.asarray(tau, dtype=float)
        F   = self.futures_price(tau)
        d_log_F = p.sigma_xi * tau + p.rho * p.sigma_chi / p.kappa * (
            1.0 - np.exp(-p.kappa * tau)
        )
        return F * d_log_F

    def dF_drho(self, tau: np.ndarray) -> np.ndarray:
        """Analytical dF/d(rho)."""
        p   = self.p
        tau = np.asarray(tau, dtype=float)
        F   = self.futures_price(tau)
        d_log_F = p.sigma_chi * p.sigma_xi / p.kappa * (
            1.0 - np.exp(-p.kappa * tau)
        )
        return F * d_log_F

    def all_gradients(self, tau: np.ndarray) -> dict[str, np.ndarray]:
        """Return dict of all analytical parameter gradients."""
        return {
            "kappa":      self.dF_dkappa(tau),
            "sigma_chi":  self.dF_dsigma_chi(tau),
            "sigma_xi":   self.dF_dsigma_xi(tau),
            "rho":        self.dF_drho(tau),
        }


# ── Monte Carlo simulation ─────────────────────────────────────────────────

def simulate_state_paths(
    params: SSParams,
    T: float,
    n_paths: int = 50000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate (chi_t, xi_t) paths under the physical measure.

    Uses exact OU discretisation for chi (no Euler error):
        chi_{t+dt} = chi_t * exp(-kappa*dt)
                     + sigma_chi * sqrt((1-exp(-2*kappa*dt))/(2*kappa)) * Z_chi

    Returns
    -------
    chi_paths : (n_paths, n_steps+1)
    xi_paths  : (n_paths, n_steps+1)

    """
    rng = np.random.default_rng(seed)
    p   = params
    dt  = T / n_steps

    chi = np.full(n_paths, p.chi0)
    xi  = np.full(n_paths, p.xi0)

    chi_paths = np.empty((n_paths, n_steps + 1))
    xi_paths  = np.empty((n_paths, n_steps + 1))
    chi_paths[:, 0] = chi
    xi_paths[:, 0]  = xi

    e1        = np.exp(-p.kappa * dt)
    std_chi   = p.sigma_chi * np.sqrt((1.0 - np.exp(-2.0 * p.kappa * dt)) / (2.0 * p.kappa))
    std_xi    = p.sigma_xi * np.sqrt(dt)

    for i in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        Z_chi = Z1
        Z_xi  = p.rho * Z1 + np.sqrt(1.0 - p.rho**2) * Z2

        chi = e1 * chi + std_chi * Z_chi
        xi  = xi + p.mu_xi * dt + std_xi * Z_xi

        chi_paths[:, i + 1] = chi
        xi_paths[:, i + 1]  = xi

    return chi_paths, xi_paths


def price_european_call_on_futures(
    params: SSParams,
    strike_ratio: float,          # K / F(0, T_fut)  i.e. moneyness
    T_option: float,              # option expiry (years)
    T_futures: float,             # futures delivery date (years) >= T_option
    risk_free_rate: float = 0.05,
    n_paths: int = 50000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> tuple[float, float]:
    """
    Price a European call option on a commodity futures contract via MC.

    The option pays max(F(T_opt, T_fut) - K, 0) at T_opt.
    F(T_opt, T_fut) is computed analytically from the simulated state.

    GW analogy: this is the "expensive likelihood" that requires full
    waveform simulation — the quantity the surrogate will replace.

    Returns
    -------
    price : float — discounted expected payoff
    stderr: float — MC standard error (~1/sqrt(n_paths))
    """
    assert T_futures >= T_option, "Futures delivery must be after option expiry."
    model_t0 = SchwartzSmithModel(params)
    atm_price = model_t0.futures_price(np.array([T_futures]))[0]
    K = strike_ratio * atm_price

    chi_paths, xi_paths = simulate_state_paths(
        params, T_option, n_paths, n_steps, seed
    )
    chi_T = chi_paths[:, -1]
    xi_T  = xi_paths[:, -1]

    # Analytical futures price at T_option for each path
    tau_remain = T_futures - T_option
    # Use params at time T_option (chi_T, xi_T) as new initial state
    tmp_params = SSParams(
        kappa=params.kappa, mu_xi=params.mu_xi,
        sigma_chi=params.sigma_chi, sigma_xi=params.sigma_xi,
        rho=params.rho, lambda_chi=params.lambda_chi,
        lambda_xi=params.lambda_xi, chi0=0.0, xi0=0.0  # placeholder
    )
    # Build per-path model — vectorised via broadcasting
    e1 = np.exp(-params.kappa * tau_remain)
    A_val = SchwartzSmithModel(tmp_params).A(np.array([tau_remain]))[0]
    ln_F = e1 * chi_T + xi_T + A_val
    F_T  = np.exp(ln_F)

    payoff = np.maximum(F_T - K, 0.0)
    discount = np.exp(-risk_free_rate * T_option)
    price  = discount * payoff.mean()
    stderr = discount * payoff.std() / np.sqrt(n_paths)
    return price, stderr


def price_spread_option(
    params1: SSParams,
    params2: SSParams,
    strike: float,
    T_option: float,
    T_futures: float,
    risk_free_rate: float = 0.05,
    n_paths: int = 50000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> tuple[float, float]:
    """
    Price a spread option: max(F1(T_opt, T_fut) - F2(T_opt, T_fut) - K, 0).

    Example: crack spread (crude oil minus gasoline).
    Two independent Schwartz-Smith models with the same number of steps.

    Returns
    -------
    price, stderr
    """
    chi1, xi1 = simulate_state_paths(params1, T_option, n_paths, n_steps, seed)
    chi2, xi2 = simulate_state_paths(params2, T_option, n_paths, n_steps,
                                     seed + 1 if seed is not None else None)
    tau = T_futures - T_option
    A1  = SchwartzSmithModel(params1).A(np.array([tau]))[0]
    A2  = SchwartzSmithModel(params2).A(np.array([tau]))[0]
    e1  = np.exp(-params1.kappa * tau)
    e2  = np.exp(-params2.kappa * tau)

    F1 = np.exp(e1 * chi1[:, -1] + xi1[:, -1] + A1)
    F2 = np.exp(e2 * chi2[:, -1] + xi2[:, -1] + A2)
    payoff = np.maximum(F1 - F2 - strike, 0.0)
    discount = np.exp(-risk_free_rate * T_option)
    price  = discount * payoff.mean()
    stderr = discount * payoff.std() / np.sqrt(n_paths)
    return price, stderr
