import sys, os
sys.path.insert(0, os.path.abspath('./'))

from __training_imports__ import *
from models import SurrogateMLP
from trainer import SchwartzSmithTrainer

import json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="/home/2673888s/commodity_futures/config/SchwartzSmithFWD.yaml")
# ── Sweep overrides ──────────────────────────────────────────────────────
# Optional CLI overrides for the hyperparameters you're sweeping over via
# HTCondor's queue-from-file. Each is None by default, meaning "use the
# yaml value" — so this script still runs standalone with just --config,
# exactly as before, and only deviates from the yaml when a sweep job
# explicitly passes a value. Same pattern as an injection script accepting
# optional --mass1/--mass2 overrides on top of a base config, rather than
# needing a fully separate yaml file per grid point.
parser.add_argument("--batch_size",   type=int,   default=None)
parser.add_argument("--alpha",        type=float, default=None)
parser.add_argument("--beta",         type=float, default=None)
parser.add_argument("--n_epochs",     type=int,   default=None)
parser.add_argument("--patience",     type=int,   default=None)
parser.add_argument("--lr",           type=float, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--tag",          type=str,   default=None,
                     help="HTCondor Cluster_Process, used ONLY as a "
                          "disambiguating suffix on top of the descriptive "
                          "hyperparameter tag built below — no longer the "
                          "primary run identifier.")
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)

# Apply overrides — only touches cfg["training"] keys explicitly passed.
overrides = {
    "batch_size":   args.batch_size,
    "differential_loss_alpha": args.alpha,
    "differential_loss_beta":  args.beta,
    "n_epochs":     args.n_epochs,
    "patience":     args.patience,
    "learning_rate": args.lr,
    "weight_decay": args.weight_decay,
}
for key, val in overrides.items():
    if val is not None:
        cfg["training"][key] = val


# ── Self-describing run tag ──────────────────────────────────────────────
# Built from the EFFECTIVE hyperparameters (post-override), so the
# directory name itself tells you what was run — no more cross-referencing
# the submit file's row order against Cluster_Process. The raw Condor tag
# (if given) is kept only as a short tie-breaker suffix in case two grid
# rows round to an identical string.
def make_run_tag(cfg: dict, condor_tag: str | None) -> str:
    t = cfg["training"]
    parts = [
        f"bs{t['batch_size']}",
        f"a{t['differential_loss_alpha']:.2f}",
        f"b{t['differential_loss_beta']:.2f}",
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
ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / "SchwartzSmithFWD"
ckpt_dir = ckpt_dir / args.tag
plot_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

print(f"Job tag: {args.tag}")
print(f"Effective hyperparameters: batch_size={cfg['training']['batch_size']}  "
      f"alpha={cfg['training']['differential_loss_alpha']}  "
      f"beta={cfg['training']['differential_loss_beta']}  "
      f"n_epochs={cfg['training']['n_epochs']}  patience={cfg['training']['patience']}  "
      f"lr={cfg['training']['learning_rate']}  weight_decay={cfg['training']['weight_decay']}")

# Save the exact effective config used for this run — the authoritative
# source for any downstream script (e.g. organize_stage1_sweep.py) that
# needs to know what hyperparameters produced this checkpoint, instead of
# reverse-engineering it from a directory name or the submit file.
with open(ckpt_dir / "config.json", "w") as f:
    json.dump({
        "batch_size":   cfg["training"]["batch_size"],
        "alpha":        cfg["training"]["differential_loss_alpha"],
        "beta":         cfg["training"]["differential_loss_beta"],
        "n_epochs":     cfg["training"]["n_epochs"],
        "patience":     cfg["training"]["patience"],
        "lr":           cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "condor_tag":   args.tag,
    }, f, indent=2)


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

# theta_net is the NETWORK-FACING array: identical to theta_raw except
# column 0 is ln(kappa) instead of raw kappa. theta_mean/theta_std were
# already computed on theta_net at generation time — this is the load-
# bearing consistency point: the same array must be used for normalising
# BOTH the training inputs and any inference-time inputs later, the same
# way a GW sampler must consistently work in ln(mass) throughout a run
# rather than switching coordinate conventions partway through.
theta_net  = data["theta_net"]
theta_raw  = data["theta_raw"]     # physical units — diagnostics/printouts only
theta_mean = data["theta_mean"]
theta_std  = data["theta_std"]
maturities = data["maturities"]
grad_keys  = data["grad_keys"]          # ["dF_dlnkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"]
param_indices = [0, 2, 3, 4]           # positions in theta_net: ln(kappa), sigma_chi, sigma_xi, rho

# ── Normalise theta_net ────────────────────────────────────────────
theta_norm = (theta_net - theta_mean) / theta_std

# ── Log-transform and normalise F ─────────────────────────────────
# GW analogy: this is like working in log-amplitude space for h(t).
# F is always positive and spans ~decades over the prior, so log-space
# normalises the output distribution far better than raw F.
log_F      = np.log(data["F"])                         # (N, M)
log_F_mean = log_F.mean(axis=0)                        # (M,)  — per-maturity mean
log_F_std  = log_F.std(axis=0)                         # (M,)  — per-maturity std
F_norm     = (log_F - log_F_mean) / log_F_std          # (N, M)

# ── Normalise gradients (per-sample, chain rule exact) ────────────
# We need: d(log_F_norm)/d(theta_norm) = dF/dtheta_net * theta_std / (F * log_F_std)
#
# This formula is UNCHANGED from before the reparametrization — it's already
# generic in terms of whichever "theta"/"theta_std"/"grads_stacked" it's
# handed. The reparametrization is entirely upstream: grads_stacked[...,0]
# is now dF/d(ln kappa) rather than dF/dkappa (computed at generation time
# as kappa * dF/dkappa), and theta_std[0] is now std(ln kappa) rather than
# std(kappa). Both changes are self-consistent by construction, the same
# way a Fisher-matrix computation doesn't need its own formula rewritten
# when you switch from raw-mass to ln-mass coordinates — only the inputs
# (the Jacobian-transformed derivative and the coordinate's own scale)
# change, not the machinery combining them.

theta_std_subset = theta_std[param_indices]            # (4,) — now includes std(ln kappa)
F_actual         = data["F"]                           # (N, M)  — raw prices

grads_norm = (
    data["grads_stacked"]                              # (N, M, 4) — col 0 now dF/d(ln kappa)
    / F_actual[:, :, None]                             # divide by ACTUAL F, not mean F
    * theta_std_subset[None, None, :]                  # scale by theta_std (chain rule)
    / log_F_std[None, :, None]                         # scale by log_F_std (chain rule)
)                                                      # -> (N, M, 4) dimensionless

print("Normalisation sanity checks (should all be O(1)):")
print(f"  theta_norm:  mean={theta_norm.mean():.3f}  std={theta_norm.std():.3f}")
print(f"  F_norm:      mean={F_norm.mean():.3f}  std={F_norm.std():.3f}")
print(f"  grads_norm:  mean={grads_norm.mean():.3f}  std={grads_norm.std():.3f}  max={np.abs(grads_norm).max():.1f}")

# Per-channel breakdown — this is the check that matters most right now:
# confirm the ln-kappa reparametrization actually tamed dF_dlnkappa's scale
# down to be comparable with the other three channels (which were all
# max <= 1.0 previously), rather than just trusting the aggregate number.
print("Per-channel |grads_norm| max (should now be comparable across channels):")
for i, key in enumerate(grad_keys):
    print(f"  {key:15s}: max={np.abs(grads_norm[..., i]).max():.2f}")

# If any channel's max is still wildly larger than the others, the
# reparametrization didn't fully fix it — go back to the stratified
# check_gradients.py diagnostic before turning alpha/beta up further.

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
# grad_keys passed straight through from the saved dataset — now
# ["dF_dlnkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"]. This must
# match EXACTLY what SchwartzSmithTrainer's param_indices dict and
# DifferentialLoss's weights dict use as keys (both patched to
# "dF_dlnkappa"), or the differential loss silently drops the kappa term
# again — same silent-skip failure mode as before, just relocated.
trainer = SchwartzSmithTrainer(
    model,
    alpha          = cfg["training"]["differential_loss_alpha"],
    beta           = cfg["training"]["differential_loss_beta"],
    lr             = cfg["training"]["learning_rate"],
    weight_decay   = cfg["training"]["weight_decay"],
    checkpoint_dir = ckpt_dir,
    grad_keys      = grad_keys,
)
history = trainer.train(train_loader, val_loader,
                         n_epochs=cfg["training"]["n_epochs"],
                         patience=cfg["training"]["patience"])

# ── Validation plots ──────────────────────────────────────────────
fig = plot_training_loss(history)
fig.savefig(plot_dir / f"training_loss_{args.tag}.png", dpi=150, bbox_inches="tight")

device = next(model.parameters()).device
model.eval()

# Test inputs must come from theta_net (ln-kappa space) — feeding raw
# theta_raw here would silently hand the network un-logged kappa values,
# which is the exact same class of mismatch as evaluating a trained
# ln-mass surrogate waveform model with a raw mass input: it won't error,
# it'll just silently produce a nonsense forward curve.
test_theta = torch.tensor(
    (theta_net[:200] - theta_mean) / theta_std, dtype=torch.float32
).to(device)

with torch.no_grad():
    F_pred_norm = model(test_theta).cpu().numpy()      # (200, M) normalised log-price

# ── Invert log-transform to get prices ────────────────────────────
log_F_pred = F_pred_norm * log_F_std + log_F_mean     # (200, M) log-price
F_pred     = np.exp(log_F_pred)                        # (200, M) price

fig = plot_surrogate_accuracy(data["F"][:200], F_pred, maturities)
fig.savefig(plot_dir / f"surrogate_accuracy_{args.tag}.png", dpi=150, bbox_inches="tight")

print(f"\nAll plots saved to {plot_dir}")
print(f"Best checkpoint at {ckpt_dir}/best_model_SchwartzSmithFWD.pt")
print("Physical-unit parameter ranges for these 200 test samples (theta_raw):")
for i, name in enumerate(cfg["param_names"]):
    print(f"  {name}: [{theta_raw[:200, i].min():.4f}, {theta_raw[:200, i].max():.4f}]")

# ── Save normalisation stats for use at inference/Stage 3 ─────────
# Everything needed to (a) normalise inputs and (b) invert outputs, in
# BOTH coordinate systems — Stage 3's MCMC will sample in physical kappa
# (the natural prior), so it needs to know to apply log() and these exact
# stats before calling the surrogate. Saving both conventions here avoids
# the sampler having to guess which space the checkpoint expects, the same
# way you'd save both the raw and whitened representations of a template
# bank rather than forcing every downstream consumer to re-derive one from
# the other.
norm_stats = {
    "theta_mean": theta_mean,   # ln(kappa) space
    "theta_std":  theta_std,    # ln(kappa) space
    "log_F_mean": log_F_mean,   # (M,)
    "log_F_std":  log_F_std,    # (M,)
    "maturities": maturities,
    "kappa_is_log": True,       # explicit flag: column 0 requires np.log() before normalising
    "kappa_index":  0,
}
np.savez(ckpt_dir / "norm_stats.npz", **norm_stats)
print(f"Normalisation stats saved to {ckpt_dir}/norm_stats.npz")