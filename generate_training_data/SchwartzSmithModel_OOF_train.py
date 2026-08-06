"""
SchwartzSmithModel_OOF_train.py — Options on Futures Ensemble Surrogate
"""

import sys, os
sys.path.insert(0, os.path.abspath('./'))

from __training_imports__ import *          # same shared imports as Stage 1
from models import SurrogateMLP
from trainer import train_ensemble

import glob
import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="/home/2673888s/commodity_futures/config/SchwartzSmithFWD.yaml")
# ── Sweep overrides ──────────────────────────────────────────────────────
parser.add_argument("--n_ensemble",   type=int,   default=None)
parser.add_argument("--batch_size",   type=int,   default=None)
parser.add_argument("--n_epochs",     type=int,   default=None)
parser.add_argument("--patience",     type=int,   default=None)
parser.add_argument("--lr",           type=float, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--tag",          type=str,   default=None,
                     help="HTCondor Cluster_Process, used ONLY as a "
                          "disambiguating suffix on top of the descriptive "
                          "hyperparameter tag built below.")
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)

overrides = {
    "n_ensemble":    args.n_ensemble,
    "batch_size":    args.batch_size,
    "n_epochs":      args.n_epochs,
    "patience":      args.patience,
    "learning_rate": args.lr,
    "weight_decay":  args.weight_decay,
}
for key, val in overrides.items():
    if val is not None:
        cfg["training"][key] = val


# ── Self-describing run tag (same fix as Stage 1) ────────────────────────
def make_run_tag(cfg: dict, condor_tag: str | None) -> str:
    t = cfg["training"]
    parts = [
        f"ens{t['n_ensemble']}",
        f"bs{t['batch_size']}",
        f"ep{t['n_epochs']}",
        f"pat{t['patience']}",
        f"lr{t['learning_rate']:.1e}",
        f"wd{t['weight_decay']:.1e}",
    ]
    tag = "_".join(parts)
    if condor_tag:
        tag += "_" + condor_tag.replace("_", "-")
    return tag


args.tag = make_run_tag(cfg, args.tag)

plot_dir = Path(cfg["output"]["plot_dir"])
ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / "SchwartzSmithFWD_OOF"
ckpt_dir = ckpt_dir / args.tag
plot_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

print(f"Job tag: {args.tag}")
print(f"Effective hyperparameters: n_ensemble={cfg['training']['n_ensemble']}  "
      f"batch_size={cfg['training']['batch_size']}  n_epochs={cfg['training']['n_epochs']}  "
      f"patience={cfg['training']['patience']}  lr={cfg['training']['learning_rate']}  "
      f"weight_decay={cfg['training']['weight_decay']}")

with open(ckpt_dir / "config.json", "w") as f:
    json.dump({
        "n_ensemble":   cfg["training"]["n_ensemble"],
        "batch_size":   cfg["training"]["batch_size"],
        "n_epochs":     cfg["training"]["n_epochs"],
        "patience":     cfg["training"]["patience"],
        "lr":           cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "condor_tag":   args.tag,
    }, f, indent=2)


def plot_training_loss(history: dict[str, list], figsize: tuple = (10, 4)) -> plt.Figure:
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
    fig.suptitle("Stage 2 Training", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_ensemble_calibration(V_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                               figsize: tuple = (12, 5)) -> plt.Figure:
    """
    Checks P(V_true in [mu +/- 1.64*sigma]) ~= 0.90 — the same coverage-test
    logic as an injection-recovery P-P plot, applied to option price
    predictions instead of MCMC posteriors.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    resid = (V_true - mu) / sigma
    ax1.hist(resid, bins=50, density=True, color=SURROGATE_RED, alpha=0.7)
    xs = np.linspace(-4, 4, 200)
    ax1.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k--", label="N(0,1)")
    ax1.set_xlabel("(V_true - mu) / sigma"); ax1.set_title("Standardised residuals")
    ax1.legend()

    frac_covered = np.mean(np.abs(resid) <= 1.64)
    ax2.bar(["Empirical", "Nominal (90%)"], [frac_covered, 0.90],
            color=[SURROGATE_RED, "gray"])
    ax2.set_ylim(0, 1); ax2.set_title("90% interval coverage")

    fig.suptitle("Stage 2 Ensemble Calibration", fontweight="bold")
    fig.tight_layout()
    return fig


# ── Load chunked MC data ────────────────────────────────────────────────
chunk_paths = sorted(glob.glob(os.path.join(cfg["data"]["save_path"],
                                             "options_on_futures_data_chunk_*.pt")))
if not chunk_paths:
    raise FileNotFoundError(
        f"No chunk files found under {cfg['data']['save_path']} — "
        f"run the data-generation sweep first."
    )
print(f"Loading {len(chunk_paths)} chunk(s)...")

X_list, V_list, V_std_list = [], [], []
for p in chunk_paths:
    chunk = torch.load(p, weights_only=False)
    X_list.append(chunk["X"])
    V_list.append(chunk["V"])
    V_std_list.append(chunk["V_std"])
    col_names = chunk["col_names"]  # assumed identical across chunks

X = torch.cat(X_list, dim=0)
V = torch.cat(V_list, dim=0)
V_std = torch.cat(V_std_list, dim=0)  # aleatoric label noise (MC stderr), NOT epistemic

# ── Normalise ────────────────────────────────────────────────────────────
X_mean = X.mean(dim=0)
X_std  = X.std(dim=0) + 1e-8
X_norm = (X - X_mean) / X_std

V_mean = V.mean()
V_std_norm = V.std() + 1e-8
V_norm = (V - V_mean) / V_std_norm
# label noise must be scaled by the SAME factor as V itself, so it stays
# in normalised-output units when fed into the heteroscedastic loss —
# the direct analogue of scaling detector-noise sigma consistently with
# whatever units/whitening you apply to the strain data itself.
V_std_norm_scaled = V_std / V_std_norm

print("Normalisation sanity checks:")
print(f"  X_norm: mean={X_norm.mean():.3f} std={X_norm.std():.3f}")
print(f"  V_norm: mean={V_norm.mean():.3f} std={V_norm.std():.3f}")
print(f"  Mean relative label noise (V_std/V): {(V_std / V.clamp(min=1e-8)).mean():.4f}")

n_val = int(cfg["training"]["val_fraction"] * len(X_norm))
X_train, X_val           = X_norm[:-n_val], X_norm[-n_val:]
V_train, V_val           = V_norm[:-n_val], V_norm[-n_val:]
Vstd_train, Vstd_val     = V_std_norm_scaled[:-n_val], V_std_norm_scaled[-n_val:]

train_set = TensorDataset(X_train, V_train, Vstd_train)
val_set   = TensorDataset(X_val,   V_val,   Vstd_val)
train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                           shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set, batch_size=cfg["training"]["batch_size"],
                           shuffle=False, num_workers=4)

# ── Train ensemble ───────────────────────────────────────────────────────
# GW analogy: M independently-seeded models playing the role of M
# independent MCMC/nested-sampling chains. Ensemble mean/variance at
# inference time separates epistemic uncertainty (chain-to-chain scatter)
# from the aleatoric MC label noise modelled explicitly in the loss.
model_kwargs = dict(
    input_dim   = X.shape[1],
    output_dim  = 2,  # [mu, log_var] per sample — heteroscedastic head
    hidden_dims = cfg["model"]["hidden_dims"],
    activation  = cfg["model"]["activation"],
)

models, histories = train_ensemble(
    SurrogateMLP,
    model_kwargs,
    train_loader,
    val_loader,
    n_models       = cfg["training"]["n_ensemble"],
    checkpoint_dir = ckpt_dir,
    n_epochs       = cfg["training"]["n_epochs"],
    patience       = cfg["training"]["patience"],
    lr             = cfg["training"]["learning_rate"],
    weight_decay   = cfg["training"]["weight_decay"],
    seed           = 0,
)

# ── Plots (using the first ensemble member's loss curve as representative) ──
fig = plot_training_loss(histories[0])
fig.savefig(plot_dir / f"training_loss_{args.tag}.png", dpi=150, bbox_inches="tight")

# ── Ensemble calibration check on held-out validation set ───────────────
device = next(models[0].parameters()).device
preds = []
with torch.no_grad():
    for m in models:
        m.eval()
        out = m(X_val.to(device))
        preds.append(out.cpu().numpy())
preds = np.stack(preds, axis=0)          # (n_ensemble, n_val, 2)
mu_members  = preds[..., 0]              # (n_ensemble, n_val)
logvar_members = preds[..., 1]

# Ensemble mean = average of member means (epistemic centre)
mu_ens = mu_members.mean(axis=0)
# Epistemic variance = variance across members
epistemic_var = mu_members.var(axis=0)
# Aleatoric variance = average of member-predicted variances
aleatoric_var = np.exp(logvar_members).mean(axis=0)
sigma_total = np.sqrt(epistemic_var + aleatoric_var)

V_val_denorm = (V_val.numpy() * V_std_norm.item()) + V_mean.item()
mu_denorm    = (mu_ens * V_std_norm.item()) + V_mean.item()
sigma_denorm = sigma_total * V_std_norm.item()

fig = plot_ensemble_calibration(V_val_denorm, mu_denorm, sigma_denorm)
fig.savefig(plot_dir / f"ensemble_calibration_{args.tag}.png", dpi=150, bbox_inches="tight")

print(f"\nAll plots saved to {plot_dir}")
print(f"Ensemble checkpoints saved under {ckpt_dir}/")

# ── Save normalisation stats for Stage 3 ─────────────────────────────────
norm_stats = {
    "X_mean": X_mean.numpy(),
    "X_std":  X_std.numpy(),
    "V_mean": V_mean.item(),
    "V_std_norm": V_std_norm.item(),
    "col_names": col_names,
}
np.savez(ckpt_dir / "norm_stats.npz", **norm_stats)
print(f"Normalisation stats saved to {ckpt_dir}/norm_stats.npz")