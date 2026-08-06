"""
surrogate.py — Stage 2 (options-on-futures ensemble) wrapper
================================================================

Key difference from Stage 1: the network's 8 inputs are NOT all
inference targets. Only (kappa, sigma_chi, sigma_xi, rho) are unknown
Schwartz-Smith parameters we calibrate for. The other four
(moneyness, T_option, T_futures, r) are KNOWN contract terms — you
observe them directly off the traded option (its strike, its
expiries, the risk-free rate) same way you know a detector's PSD
without inferring it.

So a single "day" of options calibration means: you have several
observed option contracts, each with its own KNOWN contract terms and
an OBSERVED price V_obs, and ONE shared (kappa, sigma_chi, sigma_xi,
rho) explaining all of them simultaneously — the calibration analogue
of using several detectors/contracts to jointly constrain one set of
source parameters.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import os, sys
sys.path.insert(0, os.path.abspath('../'))

try:
    from models import SurrogateMLP  # noqa: F401  (needed for unpickling)
except ImportError as e:
    print(f"[warn] Could not import SurrogateMLP: {e}")

# The four SS parameters actually being calibrated in Stage 3 Layer 1 (options).
PARAM_NAMES = ["kappa", "sigma_chi", "sigma_xi", "rho"]

PRIOR_BOUNDS = {
    "kappa":     (0.1, 5.0),
    "sigma_chi": (0.05, 0.60),
    "sigma_xi":  (0.05, 0.40),
    "rho":       (-0.9, 0.9),
}

N_DIM = len(PARAM_NAMES)

# Contract-term ranges used only to SIMULATE a realistic set of observed
# option contracts for one synthetic "day" (see gen_pseudo_data.py) — NOT
# inferred, just needs plausible ranges matching what the ensemble was
# trained on (from your options-data generator's CONTRACT dict).
CONTRACT_RANGES = {
    "moneyness": (0.7, 1.3),
    "T_option":  (0.05, 2.0),
    "r":         (0.00, 0.06),
}
TAU_EXTRA = (0.05, 1.0)  # T_futures = T_option + Uniform(*TAU_EXTRA)


@dataclass
class OptionsEnsembleWrapper:
    """
    Loads M ensemble members from one Stage 2 checkpoint dir. Each member
    outputs [mu, log_var] (heteroscedastic head, as trained). Ensemble
    mean = predicted price; combined variance = epistemic (across
    members) + aleatoric (mean of each member's predicted variance) —
    same decomposition used in your training/calibration plots.
    """
    models: list
    X_mean: np.ndarray
    X_std: np.ndarray
    V_mean: float
    V_std_norm: float
    col_names: list

    @classmethod
    def load(cls, checkpoint_dir: str | Path):
        checkpoint_dir = Path(checkpoint_dir)

        # Each ensemble member lives in its own subfolder:
        #   <checkpoint_dir>/model_00/best_model.pt
        #   <checkpoint_dir>/model_01/best_model.pt
        #   ...
        # Number of members varies by run (n_ensemble was itself swept:
        # some runs have model_00..model_02 (n_ensemble=3), others
        # model_00..model_06 (n_ensemble=7) — so just glob whatever
        # subfolders exist rather than assuming a fixed count.
        member_dirs = sorted(
            d for d in checkpoint_dir.glob("model_*") if d.is_dir()
        )
        if not member_dirs:
            raise FileNotFoundError(
                f"No model_XX/ subfolders found under {checkpoint_dir} — "
                f"expected e.g. model_00/best_model.pt, model_01/best_model.pt, ..."
            )

        models = []
        for member_dir in member_dirs:
            model_path = member_dir / "best_model.pt"
            m = torch.load(model_path, map_location="cpu", weights_only=False)
            m.eval()
            models.append(m)
        print(f"Loaded {len(models)} ensemble members from {checkpoint_dir.name}: "
              f"{[d.name for d in member_dirs]}")

        norm = np.load(checkpoint_dir / "norm_stats.npz", allow_pickle=True)
        return cls(
            models=models,
            X_mean=norm["X_mean"],
            X_std=norm["X_std"],
            V_mean=float(norm["V_mean"]),
            V_std_norm=float(norm["V_std_norm"]),
            col_names=list(norm["col_names"]),
        )

    def predict_batch(self, theta: np.ndarray, contracts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        theta:     (n_walkers, 4)  -> kappa, sigma_chi, sigma_xi, rho
        contracts: (n_contracts, 4) -> moneyness, T_option, T_futures, r
                   (FIXED, known, same across all walkers — today's
                   actual traded contracts)

        Returns:
            mu_ens:    (n_walkers, n_contracts) ensemble-mean predicted price
            sigma_ens: (n_walkers, n_contracts) total predictive std
                       (epistemic + aleatoric)
        """
        n_walkers = theta.shape[0]
        n_contracts = contracts.shape[0]

        # Broadcast into full (n_walkers * n_contracts, 8) input matrix
        theta_rep = np.repeat(theta, n_contracts, axis=0)               # (W*C, 4)
        contracts_rep = np.tile(contracts, (n_walkers, 1))              # (W*C, 4)
        X = np.concatenate([theta_rep, contracts_rep], axis=1)          # (W*C, 8)

        X_norm = (X - self.X_mean) / self.X_std
        x_t = torch.tensor(X_norm, dtype=torch.float32)

        mu_members, logvar_members = [], []
        with torch.no_grad():
            for m in self.models:
                out = m(x_t).numpy()          # (W*C, 2)
                mu_members.append(out[:, 0])
                logvar_members.append(out[:, 1])
        mu_members = np.stack(mu_members, axis=0)          # (M, W*C)
        logvar_members = np.stack(logvar_members, axis=0)  # (M, W*C)

        mu_ens_norm = mu_members.mean(axis=0)                       # (W*C,)
        epistemic_var = mu_members.var(axis=0)
        aleatoric_var = np.exp(logvar_members).mean(axis=0)
        sigma_ens_norm = np.sqrt(epistemic_var + aleatoric_var)

        # De-normalise back to physical price units
        mu_ens = mu_ens_norm * self.V_std_norm + self.V_mean
        sigma_ens = sigma_ens_norm * self.V_std_norm

        return (mu_ens.reshape(n_walkers, n_contracts),
                sigma_ens.reshape(n_walkers, n_contracts))