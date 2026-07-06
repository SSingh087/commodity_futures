import sys, os
sys.path.insert(0, os.path.abspath('./'))

from __training_imports__ import *
from models import SurrogateMLP
from trainer import OptionsOnFuturesTrainer
from losses import EnsembleLossWithLabelNoise

import glob
from scipy.stats import norm as scipy_norm


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="/home/2673888s/commodity_futures/config/SchwartzSmithFWD.yaml")
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)

plot_dir = Path(cfg["output"]["plot_dir"])
ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
plot_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)


# ── Fixed trainer: correct held-out validation metric ──────────────────
# OptionsOnFuturesTrainer inherits Trainer._validate, which does a raw
# MSE(model(theta), F_true) — a shape mismatch here, since model(X) is
# (batch, 2) = [mu, log_var] but V_true is (batch, 1). That silently
# broadcasts into a meaningless number and is what early-stopping /
# "best_model" checkpointing currently keys off. We override it to score
# validation under the same heteroscedastic NLL used in training — the
# equivalent of scoring PE convergence on whitened residuals rather than
# raw strain.
class Stage2TrainerFixed(OptionsOnFuturesTrainer):
    def _validate(self, val_loader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                X, V_true, V_std = batch
                X, V_true, V_std = X.to(self.device), V_true.to(self.device), V_std.to(self.device)
                out = self.model(X)
                mu, log_var = out[:, :1], out[:, 1:]
                loss = self.loss_fn(mu, log_var, V_true, V_std)
                losses.append(loss.item())
        return float(np.mean(losses))


def plot_training_loss(histories: list[dict], figsize: tuple = (10, 4)):
    """
    Overlay train/val heteroscedastic-NLL curves for every ensemble member —
    the trace-plot-overlay analogue of checking M independent MCMC chains
    for agreement before trusting the pooled posterior.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for m, history in enumerate(histories):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], color=GW_BLUE,   alpha=0.5,
                 label="Train" if m == 0 else None)
        ax1.plot(epochs, history["val_loss"],   color=SURROGATE_RED, alpha=0.5,
                 label="Val" if m == 0 else None)
        ax2.plot(epochs, history["lr"], color=TRUTH_GREEN, alpha=0.5)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Heteroscedastic NLL")
    ax1.set_title("Stage 2 Training — All Ensemble Members"); ax1.legend()
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate")
    ax2.set_title("Cosine LR Schedule")

    fig.suptitle("Stage 2: Options-on-Futures Ensemble Training", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_calibration(V_true, mu, sigma_total, figsize: tuple = (12, 5)):
    """Accuracy scatter + calibrated-UQ errorbar plot (90% CI)."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n_show = min(500, len(V_true))
    axes[0].scatter(V_true[:n_show], mu[:n_show], s=5, alpha=0.5, color=GW_BLUE)
    lims = [V_true[:n_show].min(), V_true[:n_show].max()]
    axes[0].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlabel("MC price (true)"); axes[0].set_ylabel("Surrogate prediction")
    axes[0].set_title("Surrogate Accuracy"); axes[0].legend()

    n_err = min(100, len(V_true))
    axes[1].errorbar(
        V_true[:n_err], mu[:n_err], yerr=1.64 * sigma_total[:n_err],
        fmt="o", ms=3, alpha=0.5, capsize=2, color=SURROGATE_RED, label="90% CI"
    )
    elims = [V_true[:n_err].min(), V_true[:n_err].max()]
    axes[1].plot(elims, elims, "r--", lw=1.5)
    axes[1].set_xlabel("MC price (true)"); axes[1].set_ylabel("Surrogate prediction")
    axes[1].set_title("Ensemble Uncertainty (90% CI, epistemic + aleatoric)")
    axes[1].legend()

    fig.suptitle("Stage 2: Options Surrogate — Accuracy & Calibrated UQ", fontweight="bold")
    fig.tight_layout()
    return fig


# ── Load data (all HTCondor chunks) ────────────────────────────────────
# Each chunk was generated with an independent child SeedSequence — the
# non-overlapping-state analogue of orthogonal detector-noise
# realisations — so concatenating chunks gives i.i.d. draws from the
# prior, same as pooling independent noise-injection runs.
chunk_paths = sorted(glob.glob(
    os.path.join(cfg["data"]["save_path"], "options_on_futures_data_chunk_*.pt")
))
if len(chunk_paths) == 0:
    raise FileNotFoundError(
        f"No chunk files found under {cfg['data']['save_path']}. "
        "Run the data-generation script first."
    )
print(f"Found {len(chunk_paths)} data chunks.")

X_parts, V_parts, V_std_parts = [], [], []
for p in chunk_paths:
    chunk = torch.load(p, weights_only=False)
    X_parts.append(chunk["X"])
    V_parts.append(chunk["V"])
    V_std_parts.append(chunk["V_std"])
    col_names = chunk["col_names"]

X_all     = torch.cat(X_parts,     dim=0).numpy()
V_all     = torch.cat(V_parts,     dim=0).numpy()
V_std_all = torch.cat(V_std_parts, dim=0).numpy()
print(f"Total samples: {len(V_all):,}   input dim: {X_all.shape[1]}   cols: {col_names}")

# ── Normalise (recomputed globally post-concatenation) ─────────────────
X_mean = X_all.mean(axis=0).astype(np.float32)
X_std  = (X_all.std(axis=0) + 1e-8).astype(np.float32)
X_norm = (X_all - X_mean) / X_std

# V and V_std are kept in *linear* price units (not log, unlike Stage 1's
# F): the heteroscedastic loss needs V_std in the same units as V, and
# rescaling both by the same factor preserves that relationship exactly —
# like keeping a noise PSD and a strain in consistent physical units
# rather than log-compressing only one of them.
V_mean     = float(V_all.mean())
V_std_norm = float(V_all.std()) + 1e-8
V_norm       = (V_all - V_mean) / V_std_norm
V_std_scaled = V_std_all / V_std_norm

print("Normalisation sanity checks (should all be O(1)):")
print(f"  X_norm:  mean={X_norm.mean():.3f}  std={X_norm.std():.3f}")
print(f"  V_norm:  mean={V_norm.mean():.3f}  std={V_norm.std():.3f}")
print(f"  mean SNR (V_norm / V_std_scaled): {(np.abs(V_norm) / (V_std_scaled + 1e-8)).mean():.1f}")

# ── To tensors, train/val split ────────────────────────────────────────
X_t    = torch.tensor(X_norm,       dtype=torch.float32)
V_t    = torch.tensor(V_norm,       dtype=torch.float32).unsqueeze(-1)
Vstd_t = torch.tensor(V_std_scaled, dtype=torch.float32).unsqueeze(-1)

n_val = int(cfg["training"]["val_fraction"] * len(X_t))
perm  = torch.randperm(len(X_t), generator=torch.Generator().manual_seed(cfg["training"]["seed"]))
val_idx, train_idx = perm[:n_val], perm[n_val:]

train_set = TensorDataset(X_t[train_idx], V_t[train_idx], Vstd_t[train_idx])
val_set   = TensorDataset(X_t[val_idx],   V_t[val_idx],   Vstd_t[val_idx])

train_loader = DataLoader(
    train_set, batch_size=cfg["training"]["batch_size"],
    shuffle=True, num_workers=4,
)
val_loader = DataLoader(
    val_set, batch_size=cfg["training"]["batch_size"],
    shuffle=False, num_workers=4,
)

# ── Train the deep ensemble ─────────────────────────────────────────────
# Each member outputs (mu, log_var). mu is the price estimate; log_var is
# a *learned, per-sample* aleatoric term tracking the known MC label noise.
# The spread of mu across M independent members is the epistemic term —
# same split as separating statistical vs. systematic uncertainty on an
# inferred parameter, or "shot noise vs. calibration uncertainty" in a
# detector pipeline.
n_ensemble = cfg["model"]["n_ensemble"]
n_epochs   = cfg["training"]["n_epochs"]
patience   = cfg["training"].get("patience", 20)

models, histories = [], []
for m in range(n_ensemble):
    print(f"\n{'='*50}")
    print(f"Training ensemble member {m+1}/{n_ensemble}")
    torch.manual_seed(cfg["training"]["seed"] + m)

    model = SurrogateMLP(
        input_dim   = X_t.shape[1],
        output_dim  = 2,             # [mu, log_var] — no output activation,
                                      # log_var must be free to go negative
        hidden_dims = cfg["model"]["hidden_dims"],
        activation  = cfg["model"]["activation"],
    )
    trainer = Stage2TrainerFixed(
        model,
        lr             = cfg["training"]["learning_rate"],
        weight_decay   = cfg["training"]["weight_decay"],
        checkpoint_dir = ckpt_dir / f"model_{m:02d}",
    )
    history = trainer.train(train_loader, val_loader, n_epochs=n_epochs, patience=patience)

    models.append(trainer.model)
    histories.append(history)

fig = plot_training_loss(histories)
fig.savefig(plot_dir / "training_loss.png", dpi=150, bbox_inches="tight")

# ── Ensemble prediction on held-out validation set ──────────────────────
device = next(models[0].parameters()).device
for m in models:
    m.eval()

val_X = X_t[val_idx].to(device)
with torch.no_grad():
    mus, sigmas_aleatoric = [], []
    for m in models:
        out = m(val_X)
        mu_i, log_var_i = out[:, :1], out[:, 1:]
        mus.append(mu_i.cpu().numpy())
        sigmas_aleatoric.append(np.exp(0.5 * log_var_i.cpu().numpy()))

mus              = np.stack(mus, axis=0)               # (M, N, 1)
sigmas_aleatoric = np.stack(sigmas_aleatoric, axis=0)   # (M, N, 1)

mu_mean     = mus.mean(axis=0).flatten()
sigma_epist = mus.std(axis=0).flatten()
sigma_aleat = np.sqrt((sigmas_aleatoric ** 2).mean(axis=0)).flatten()
sigma_total = np.sqrt(sigma_epist**2 + sigma_aleat**2)

# Undo normalisation back to raw price units
V_true_raw      = V_all[val_idx.numpy()]
mu_mean_raw     = mu_mean * V_std_norm + V_mean
sigma_epist_raw = sigma_epist * V_std_norm
sigma_aleat_raw = sigma_aleat * V_std_norm
sigma_total_raw = sigma_total * V_std_norm

print(f"\nMean epistemic std : {sigma_epist_raw.mean():.5f}")
print(f"Mean aleatoric std : {sigma_aleat_raw.mean():.5f}  (should track mean MC stderr, "
      f"{V_std_all[val_idx.numpy()].mean():.5f})")

# ── Calibration check ────────────────────────────────────────────────────
# Same check as verifying frequentist coverage of a credible interval:
# P(theta_true in 90% CI) should equal 0.90 if your posterior is honest.
print("\nChecking ensemble calibration on validation set ...")
for cov_target in [0.68, 0.90, 0.95]:
    z = scipy_norm.ppf(0.5 + cov_target / 2)
    in_ci = np.abs(V_true_raw - mu_mean_raw) < z * sigma_total_raw
    print(f"  {cov_target*100:.0f}% CI coverage: {in_ci.mean()*100:.1f}%")

fig = plot_calibration(V_true_raw, mu_mean_raw, sigma_total_raw)
fig.savefig(plot_dir / "ensemble_accuracy.png", dpi=150, bbox_inches="tight")

print(f"\nAll Stage 2 plots saved to {plot_dir}")
print(f"Checkpoints at {ckpt_dir}")

# ── Save normalisation stats (needed at Stage 3 inference time) ─────────
norm_stats = {
    "X_mean":     X_mean,
    "X_std":      X_std,
    "V_mean":     V_mean,
    "V_std_norm": V_std_norm,
    "col_names":  col_names,
}
np.savez(ckpt_dir / "norm_stats.npz", **norm_stats)
print(f"Normalisation stats saved to {ckpt_dir}/norm_stats.npz")