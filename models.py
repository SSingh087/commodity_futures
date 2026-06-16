import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    
    class SurrogateMLP(nn.Module):
        """
        Multi-layer perceptron surrogate for the Schwartz-Smith forward curve.
        Input  : 8 parameters [kappa, mu_xi, sigma_chi, sigma_xi, rho, lambda_chi, chi0, xi0]
        Output : forward curve vector F(T_1), ..., F(T_n)  — one value per maturity
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            hidden_dims: list[int] = (64, 128, 128, 64),
            activation: str = "gelu",
        ):
            super().__init__()
            activations = {
                "gelu":    nn.GELU(),
                "relu":    nn.ReLU(),
                "silu":    nn.SiLU(),
                "tanh":    nn.Tanh(),
            }
            act = activations[activation]

            layers = []
            in_dim = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(in_dim, h), type(act)()]
                in_dim = h
            layers.append(nn.Linear(in_dim, output_dim))

            self.net = nn.Sequential(*layers)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def predict_with_gradients(
            self, x: torch.Tensor, param_indices: list[int]
        ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
            """
            Forward pass returning output and per-maturity gradients
            dF_m/dtheta_idx for each requested parameter index.

            Returns
            -------
            out   : (batch, M)
            grads : dict {param_idx: (batch, M) tensor of dF_m/dtheta_idx}
            """
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                out = self.forward(x)  # (batch, M)

                M = out.shape[1]
                grads = {idx: [] for idx in param_indices}

                for m in range(M):
                    g = torch.autograd.grad(
                        out[:, m].sum(), x, create_graph=True, retain_graph=True
                    )[0]  # (batch, 8)
                    for idx in param_indices:
                        grads[idx].append(g[:, idx])  # (batch,)

            grads = {idx: torch.stack(vals, dim=1) for idx, vals in grads.items()}  # (batch, M)
            return out, grads