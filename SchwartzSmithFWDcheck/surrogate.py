"""
stage3_surrogate.py — Trained Stage 1 surrogate wrapper + prior definition
============================================================================

Shared by both stage3_data.py (to generate synthetic F_obs) and
stage3_sampler.py (to evaluate the likelihood). Kept separate so both
modules import the SAME normalisation logic — no risk of the data
generator and the sampler disagreeing about units/log-transforms.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import os, sys
sys.path.insert(0, os.path.abspath('../'))

try:
    from models import SurrogateMLP  # noqa: F401  (needed for unpickling)
except ImportError as e:
    print(f"[warn] Could not import SurrogateMLP: {e}. "
          f"Run from the same directory (or sys.path setup) as your "
          f"training scripts.")


# All EIGHT inputs the network takes (matches input_dim=8 in your Stage 1
# training script): the six physical model parameters PLUS the two
# initial-state values chi0, xi0 needed to evaluate today's curve.
PARAM_NAMES = ["kappa", "mu_xi", "sigma_chi", "sigma_xi", "rho", "lambda_chi",
               "chi0", "xi0"]

PRIOR_BOUNDS = {
    "kappa":      (0.1, 5.0),
    "mu_xi":      (0.0, 0.10),
    "sigma_chi":  (0.05, 0.60),
    "sigma_xi":   (0.05, 0.40),
    "rho":        (-0.9, 0.9),
    "lambda_chi": (-0.5, 0.5),
    "chi0":       (-0.5, 0.5),
    "xi0":        (2.5, 5.0),
}

N_DIM = len(PARAM_NAMES)


@dataclass
class SurrogateWrapper:
    """
    Thin wrapper around the trained Stage 1 model (`SurrogateMLP` — plain
    MLP, no SVD/basis compression anywhere in this pipeline).

    Must exactly mirror SchwartzSmithModelFWD_train.py's normalisation:
      - theta column `kappa_index` is ln(kappa) INSIDE the network. theta
        passed into predict()/predict_batch() here is always physical
        units (raw kappa) — we log() it ourselves before normalising.
      - The network's OUTPUT is normalised log(F), not normalised F. We
        invert with exp() after un-normalising.
      - Stats live in norm_stats.npz (theta_mean, theta_std, log_F_mean,
        log_F_std, maturities, kappa_is_log, kappa_index).
    """
    model: torch.nn.Module
    theta_mean: np.ndarray
    theta_std: np.ndarray
    log_F_mean: np.ndarray
    log_F_std: np.ndarray
    maturities: np.ndarray
    kappa_index: int = 0

    @classmethod
    def load(cls, checkpoint_dir: str | Path):
        checkpoint_dir = Path(checkpoint_dir)
        # weights_only=False: this checkpoint is a full pickled model
        # object (torch.save(model, ...)) from your own training run,
        # not a state_dict, and not from an untrusted source.
        model = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu",
                            weights_only=False)
        model.eval()

        norm = np.load(checkpoint_dir / "norm_stats.npz", allow_pickle=True)
        assert bool(norm["kappa_is_log"]), (
            "norm_stats.npz has kappa_is_log=False — this wrapper assumes "
            "the ln(kappa) reparametrization. Update _to_network_theta() "
            "if that's changed."
        )

        return cls(
            model=model,
            theta_mean=norm["theta_mean"],
            theta_std=norm["theta_std"],
            log_F_mean=norm["log_F_mean"],
            log_F_std=norm["log_F_std"],
            maturities=norm["maturities"],
            kappa_index=int(norm["kappa_index"]),
        )

    def _to_network_theta(self, theta: np.ndarray) -> np.ndarray:
        theta_net = theta.copy()
        theta_net[..., self.kappa_index] = np.log(theta_net[..., self.kappa_index])
        return theta_net

    def predict(self, theta: np.ndarray) -> np.ndarray:
        """theta: (8,) physical units (raw kappa) -> F(T): (n_maturities,) physical prices."""
        theta_net = self._to_network_theta(theta)
        theta_norm = (theta_net - self.theta_mean) / self.theta_std
        with torch.no_grad():
            x = torch.tensor(theta_norm, dtype=torch.float32).unsqueeze(0)
            log_F_norm = self.model(x).numpy()[0]
        log_F = log_F_norm * self.log_F_std + self.log_F_mean
        return np.exp(log_F)

    def predict_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorised — thetas: (n, 8) -> (n, n_maturities). Used by nessai's
        vectorised likelihood evaluation for a big speed win over per-point calls."""
        theta_net = self._to_network_theta(thetas)
        theta_norm = (theta_net - self.theta_mean) / self.theta_std
        with torch.no_grad():
            x = torch.tensor(theta_norm, dtype=torch.float32)
            log_F_norm = self.model(x).numpy()
        log_F = log_F_norm * self.log_F_std + self.log_F_mean
        return np.exp(log_F)