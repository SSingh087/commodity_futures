"""
trainer.py
==========
Training loop for neural network surrogates (Stages 1 and 2).

Features
--------
- Cosine annealing learning rate scheduler
- Early stopping on validation loss
- Checkpoint saving (best model + each epoch)
- Gradient clipping
- Per-epoch logging of all loss components
- Calibration check after training (coverage validation)

"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.abspath('./'))
from losses import DifferentialLoss

class Trainer:
    """
    Generic trainer for neural network surrogates.

    Handles both Schwartz-Smith forward curve and  E options via
    the loss_fn parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        checkpoint_dir: str | Path = "results/checkpoints",
        device: str = "auto",
        grad_clip: float = 1.0,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model  = model.to(device)
        self.loss_fn = loss_fn

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip

        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "lr": []
        }
        self.best_val_loss = np.inf

    def _scheduler(self, n_epochs: int) -> "torch.optim.lr_scheduler.LRScheduler":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-6
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 200,
        patience: int = 20,
        verbose: bool = True,
    ) -> dict[str, list]:
        """
        Run the training loop.

        Returns the training history dict.
        """
        scheduler = self._scheduler(n_epochs)
        patience_counter = 0
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            # ── Train ──────────────────────────────────
            self.model.train()
            train_losses = []

            for batch in train_loader:
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                train_losses.append(loss.item())

            # ── Validate ───────────────────────────────
            val_loss = self._validate(val_loader)
            train_loss = np.mean(train_losses)
            current_lr = scheduler.get_last_lr()[0]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)
            scheduler.step()

            # ── Checkpoint ────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model, self.checkpoint_dir / "best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(f"Epoch {epoch:4d}/{n_epochs} | "
                        f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                        f"LR: {current_lr:.2e} | Elapsed: {elapsed:.0f}s")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)")
                break

        self._save_history()
        print(f"Training complete. Best val loss: {self.best_val_loss:.5f}")
        return self.history

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        """Override this in subclasses for different stages."""
        raise NotImplementedError

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_loss(batch)
                losses.append(loss.item())
        return float(np.mean(losses))

    def _save_history(self):
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f)


class SchwartzSmithTrainer(Trainer):
    """
    Trainer for the forward curve surrogate (Stage 1).
    Uses the DifferentialLoss with gradient matching.
    """

    def __init__(self, model, alpha=0.1, beta=0.1, grad_keys=None, **kwargs):
        loss_fn = DifferentialLoss(alpha=alpha, beta=beta)
        super().__init__(model, loss_fn, **kwargs)
        # grad_keys order must match the last axis of grads_stacked,
        # and param_indices must match the position in theta
        self.grad_keys = grad_keys or [
            "dF_dkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"
        ]
        self.param_indices = {
            "dF_dkappa":     0,
            "dF_dsigma_chi": 2,
            "dF_dsigma_xi":  3,
            "dF_drho":       4,
        }

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        theta, F_true, grads_true = batch
        theta      = theta.to(self.device)
        F_true     = F_true.to(self.device)
        grads_true = grads_true.to(self.device)   # (batch, M, 4)

        # Unstack into dict keyed by grad name
        true_grads = {
            k: grads_true[..., i] for i, k in enumerate(self.grad_keys)
        }

        # Get predictions + autograd gradients
        F_pred, pred_grads = self.model.predict_with_gradients(
            theta, list(self.param_indices.values())
        )
        pred_grads_named = {
            name: pred_grads[idx]
            for name, idx in self.param_indices.items()
        }

        loss, _ = self.loss_fn(F_pred, F_true, pred_grads_named, true_grads)
        return loss

# except ImportError:
#     pass  # No-op if PyTorch not installed
