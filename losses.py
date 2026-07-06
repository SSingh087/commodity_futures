"""
losses.py
=========
Loss functions for training the neural network surrogates.

DifferentialLoss
----------------
The standard MSE loss on F(T) is augmented with terms that penalise
errors in the model's gradients dF/dtheta. This forces the surrogate
to learn the *geometry* of the forward curve surface, not just its values.

    L = MSE(F) + alpha * MSE(dF/d(ln kappa)) + beta * MSE(dF/dsigma_chi) + ...

Note the kappa channel is now dF/d(ln kappa), not dF/dkappa — reparametrized
to tame the intrinsic 1/kappa divergence in the raw gradient (kappa's
sensitivity blows up as mean-reversion weakens, the continuous-time analogue
of a Fisher-matrix element diverging near a near-unit-root / long-relaxation-
time boundary). See generate_forward_curve_data.py for where kappa * dF/dkappa
is computed.


EnsembleLoss with Label Noise
----------------------
For MC-labelled training data, the labels V_i are themselves noisy with
known standard error V_std_i. We explicitly model this aleatoric noise:

    L = sum_i [  (V_i - V_hat_i)^2 / (sigma_model_i^2 + sigma_label_i^2)
               + log(sigma_model_i^2 + sigma_label_i^2)  ]

where sigma_model is the predicted uncertainty from the network and
sigma_label = V_std is the MC label noise.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DifferentialLoss(nn.Module):
        """
        MSE loss on forward curve values + weighted MSE on analytical gradients.

        Parameters
        ----------
        alpha : weight on d(ln kappa) gradient term
        beta  : weight on sigma_chi gradient term
        gamma : weight on sigma_xi gradient term
        delta : weight on rho gradient term
        """

        def __init__(
            self,
            alpha: float = 0.10,
            beta:  float = 0.10,
            gamma: float = 0.05,
            delta: float = 0.05,
        ):
            super().__init__()
            self.weights = {
                "dF_dlnkappa":   alpha,
                "dF_dsigma_chi": beta,
                "dF_dsigma_xi":  gamma,
                "dF_drho":       delta,
            }

        def forward(
            self,
            F_pred: torch.Tensor,
            F_true: torch.Tensor,
            pred_grads: dict[str, torch.Tensor],
            true_grads: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, float]]:
            """
            Parameters
            ----------
            F_pred     : (batch, M) predicted forward curves
            F_true     : (batch, M) true forward curves
            pred_grads : dict of (batch, M) predicted gradients (via autograd)
            true_grads : dict of (batch, M) analytical gradient targets

            Returns
            -------
            total_loss : scalar tensor
            loss_dict  : breakdown for logging
            """
            mse_F = F.mse_loss(F_pred, F_true)
            loss_dict = {"mse_F": mse_F.item()}
            total = mse_F

            for key, weight in self.weights.items():
                if key in pred_grads and key in true_grads:
                    g_loss = F.mse_loss(pred_grads[key], true_grads[key])
                    total = total + weight * g_loss
                    loss_dict[key] = g_loss.item()

            loss_dict["total"] = total.item()
            return total, loss_dict

    class EnsembleLossWithLabelNoise(nn.Module):
        """
        Heteroscedastic loss for Stage 2 options surrogate.

        Network predicts (mean, log_var). Label noise V_std is known
        from MC sampling and is treated as fixed aleatoric uncertainty.

        Loss = sum [ (V - mu)^2 / (exp(log_var) + sigma_label^2)
                     + log(exp(log_var) + sigma_label^2) ]
        """

        def forward(
            self,
            mu_pred: torch.Tensor,
            log_var_pred: torch.Tensor,
            V_true: torch.Tensor,
            V_std: torch.Tensor,
        ) -> torch.Tensor:
            sigma2_model = torch.exp(log_var_pred)
            sigma2_label = V_std ** 2
            sigma2_total = sigma2_model + sigma2_label + 1e-8

            loss = ((V_true - mu_pred) ** 2 / sigma2_total
                    + torch.log(sigma2_total)).mean()
            return loss


    class CalibrationLoss(nn.Module):
        """
        Auxiliary calibration loss that penalises miscoverage.

        After training, we check: P(V* in [mu +/- z*sigma]) ~= target_coverage.
        This loss can be used to fine-tune the uncertainty estimates.
        """

        def __init__(self, target_coverage: float = 0.90):
            super().__init__()
            from scipy.stats import norm
            self.z = norm.ppf(0.5 + target_coverage / 2.0)
            self.target = target_coverage

        def forward(
            self,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            V_true: torch.Tensor,
        ) -> torch.Tensor:
            in_interval = ((V_true >= mu - self.z * sigma) &
                           (V_true <= mu + self.z * sigma)).float()
            achieved_coverage = in_interval.mean()
            return (achieved_coverage - self.target) ** 2

except ImportError:
    pass