import sys, os
sys.path.insert(0, os.path.abspath('./'))

from __training_imports__ import *
from models import SurrogateMLP
from trainer import SchwartzSmithTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="/home/2673888s/commodity_futures/config/SchwartzSmithFWD.yaml")
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)

plot_dir = Path(cfg["output"]["plot_dir"])
ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
plot_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)


def plot_training_loss(
    history: dict[str, list],
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot training and validation loss curves with LR schedule."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax1.semilogy(epochs, history["train_loss"], label="Train", color=GW_BLUE)
    ax1.semilogy(epochs, history["val_loss"],   label="Val",   color=SURROGATE_RED)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Training Loss"); ax1.legend()

    ax2.plot(epochs, history["lr"], color=TRUTH_GREEN)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate")
    ax2.set_title("Cosine LR Schedule")

    fig.suptitle("Stage 1 Training", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_surrogate_accuracy(
    F_true: np.ndarray,
    F_pred: np.ndarray,
    maturities: np.ndarray,
    n_examples: int = 6,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Compare surrogate predictions vs. analytical forward curves on held-out
    test samples. Shows both individual curves and residuals.
    """
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    idx  = np.random.choice(len(F_true), n_examples, replace=False)

    for i, (ax, j) in enumerate(zip(axes, idx)):
        ax.plot(maturities, F_true[j], "k-",  lw=2, label="Analytical")
        ax.plot(maturities, F_pred[j], "--",   lw=2, color=SURROGATE_RED, label="Surrogate")
        rel_err = (F_pred[j] - F_true[j]) / F_true[j] * 100
        ax2 = ax.twinx()
        ax2.bar(maturities, rel_err, width=0.15, alpha=0.25, color="gray", label="Rel. error %")
        ax2.set_ylabel("Rel. error (%)", color="gray", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="gray")
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("F(T)")

    fig.suptitle("Surrogate Accuracy on Held-Out Test Set\n"
                 "(each panel: one random parameter draw)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ── Load data ────────────────────────────────────────────────────────
data = torch.load(
    os.path.join(cfg["data"]["save_path"], "SchwartzSmithFWD_dataset.pt"),
    weights_only=False,
)

theta_mean = data["theta_mean"]
theta_std  = data["theta_std"]
maturities = data["maturities"]
grad_keys  = data["grad_keys"]          # ["dF_dkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"]
param_indices = [0, 2, 3, 4]           # positions of those params in theta

# ── Normalise theta ────────────────────────────────────────────────
theta_norm = (data["theta"] - theta_mean) / theta_std

# ── Log-transform and normalise F ─────────────────────────────────
# GW analogy: this is like working in log-amplitude space for h(t).
# F is always positive and spans ~decades over the prior, so log-space
# normalises the output distribution far better than raw F.
log_F      = np.log(data["F"])                         # (N, M)
log_F_mean = log_F.mean(axis=0)                        # (M,)  — per-maturity mean
log_F_std  = log_F.std(axis=0)                         # (M,)  — per-maturity std
F_norm     = (log_F - log_F_mean) / log_F_std          # (N, M)

# ── Normalise gradients (per-sample, chain rule exact) ────────────
# We need: d(log_F_norm)/d(theta_norm) = dF/dtheta * theta_std / (F * log_F_std)

theta_std_subset = theta_std[param_indices]            # (4,)
F_actual         = data["F"]                           # (N, M)  — raw prices

grads_norm = (
    data["grads_stacked"]                              # (N, M, 4) raw dF/dtheta
    / F_actual[:, :, None]                             # divide by ACTUAL F, not mean F
    * theta_std_subset[None, None, :]                  # scale by theta_std (chain rule)
    / log_F_std[None, :, None]                         # scale by log_F_std (chain rule)
)                                                      # -> (N, M, 4) dimensionless

print("Normalisation sanity checks (should all be O(1)):")
print(f"  theta_norm:  mean={theta_norm.mean():.3f}  std={theta_norm.std():.3f}")
print(f"  F_norm:      mean={F_norm.mean():.3f}  std={F_norm.std():.3f}")
print(f"  grads_norm:  mean={grads_norm.mean():.3f}  std={grads_norm.std():.3f}  max={np.abs(grads_norm).max():.1f}")

# If grads_norm.std() >> 5 or max >> 50, the differential loss will explode.
# In that case either (a) clip grads at data-generation time or
# (b) reduce alpha/beta/gamma/delta in the config until the loss terms are balanced.

# ── To tensors ────────────────────────────────────────────────────
theta_t = torch.tensor(theta_norm, dtype=torch.float32)
F_t     = torch.tensor(F_norm,     dtype=torch.float32)
grads_t = torch.tensor(grads_norm, dtype=torch.float32)  # (N, M, 4)

# ── Train/val split ───────────────────────────────────────────────
n_val = int(cfg["training"]["val_fraction"] * len(theta_t))
theta_train, theta_val = theta_t[:-n_val], theta_t[-n_val:]
F_train, F_val         = F_t[:-n_val],     F_t[-n_val:]
grads_train, grads_val = grads_t[:-n_val], grads_t[-n_val:]

train_set = TensorDataset(theta_train, F_train, grads_train)
val_set   = TensorDataset(theta_val,   F_val,   grads_val)
train_loader = DataLoader(
    train_set, batch_size=cfg["training"]["batch_size"],
    shuffle=True, num_workers=4,
)
val_loader = DataLoader(
    val_set, batch_size=cfg["training"]["batch_size"],
    shuffle=False, num_workers=4,
)

# ── Model ─────────────────────────────────────────────────────────
n_maturities = len(cfg["data"]["maturities"])
model = SurrogateMLP(
    input_dim   = 8,
    output_dim  = n_maturities,
    hidden_dims = cfg["model"]["hidden_dims"],
    activation  = cfg["model"]["activation"],
)

# ── Train ─────────────────────────────────────────────────────────
trainer = SchwartzSmithTrainer(
    model,
    alpha          = cfg["training"]["differential_loss_alpha"],
    beta           = cfg["training"]["differential_loss_beta"],
    lr             = cfg["training"]["learning_rate"],
    weight_decay   = cfg["training"]["weight_decay"],
    checkpoint_dir = ckpt_dir,
    grad_keys      = grad_keys,
)
history = trainer.train(train_loader, val_loader, n_epochs=cfg["training"]["n_epochs"])

# ── Validation plots ──────────────────────────────────────────────
fig = plot_training_loss(history)
fig.savefig(plot_dir / "training_loss.png", dpi=150, bbox_inches="tight")

device = next(model.parameters()).device
model.eval()

test_theta = torch.tensor(
    (data["theta"][:200] - theta_mean) / theta_std, dtype=torch.float32
).to(device)

with torch.no_grad():
    F_pred_norm = model(test_theta).cpu().numpy()      # (200, M) normalised log-price

# ── Invert log-transform to get prices ────────────────────────────
# Chain: network output (normalised) -> log F -> F
# log_F_mean and log_F_std are (M,) arrays saved above; used consistently.
log_F_pred = F_pred_norm * log_F_std + log_F_mean     # (200, M) log-price
F_pred     = np.exp(log_F_pred)                        # (200, M) price

fig = plot_surrogate_accuracy(data["F"][:200], F_pred, maturities)
fig.savefig(plot_dir / "surrogate_accuracy.png", dpi=150, bbox_inches="tight")

print(f"\nAll plots saved to {plot_dir}")
print(f"Best checkpoint at {ckpt_dir}/best_model.pt")

# ── Save normalisation stats for use at inference/Stage 3 ─────────
# Everything needed to (a) normalise inputs and (b) invert outputs
# must be saved alongside the model — exactly like saving PSD
# calibration data alongside a matched-filter template bank.
norm_stats = {
    "theta_mean": theta_mean,
    "theta_std":  theta_std,
    "log_F_mean": log_F_mean,   # (M,)
    "log_F_std":  log_F_std,    # (M,)
    "maturities": maturities,
}
np.savez(ckpt_dir / "norm_stats.npz", **norm_stats)
print(f"Normalisation stats saved to {ckpt_dir}/norm_stats.npz")