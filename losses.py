"""
losses.py
=========
Loss functions for training the neural network surrogates.

DifferentialLoss
----------------
The standard MSE loss on F(T) is augmented with terms that penalise
errors in the model's gradients dF/dtheta. This forces the surrogate
to learn the *geometry* of the forward curve surface, not just its values.

    L = MSE(F) + alpha * MSE(dF/dkappa) + beta * MSE(dF/dsigma_chi)

GW analogy: this is the equivalent of not just fitting the waveform h(t)
but also constraining dh/dM and dh/dchi. It ensures the Fisher information
matrix (curvature of the likelihood) is correctly captured, which is
essential for accurate posterior shapes in the MCMC step.

EnsembleLoss (Stage 2)
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
        alpha : weight on kappa gradient term
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
                "dF_dkappa":    alpha,
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


except ImportError:
    pass  # No-op if PyTorch not installed
