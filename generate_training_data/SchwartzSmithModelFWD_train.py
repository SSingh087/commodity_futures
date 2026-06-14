import sys, os
sys.path.insert(0, os.path.abspath('../'))

from __training_imports__ import *
from models import SurrogateMLP
from trainer import SchwartzSmithTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config/SchwartzSmithFWD.yaml")
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)

plot_dir = Path(cfg["output"]["plot_dir"])
ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
plot_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
data = torch.load(cfg["data"]["save_path"], weights_only=False)

theta_mean = data["theta_mean"]
theta_std  = data["theta_std"]
F_mean     = data["F_mean"]
F_std      = data["F_std"]
maturities = data["maturities"]
grad_keys  = data["grad_keys"]            # order: kappa, sigma_chi, sigma_xi, rho
param_indices = [0, 2, 3, 4]              # positions of those params in theta

# ── Normalise theta and F ───────────────────────────────────────────
theta_norm = (data["theta"] - theta_mean) / theta_std
F_norm     = (data["F"]     - F_mean)     / F_std

# ── Normalise gradients (chain rule scaling) ────────────────────────
theta_std_subset = theta_std[param_indices]                  # (4,)
grads_norm = data["grads_stacked"] \
             * theta_std_subset[None, None, :] \
             / F_std[None, :, None]                          # (N, M, 4)

# ── To tensors ───────────────────────────────────────────────────────
theta_t = torch.tensor(theta_norm, dtype=torch.float32)
F_t     = torch.tensor(F_norm,     dtype=torch.float32)
grads_t = torch.tensor(grads_norm, dtype=torch.float32)      # (N, M, 4)

# ── Train/val split ──────────────────────────────────────────────────
n_val = int(cfg["training"]["val_fraction"] * len(theta_t))
theta_train, theta_val = theta_t[:-n_val], theta_t[-n_val:]
F_train, F_val         = F_t[:-n_val],     F_t[-n_val:]
grads_train, grads_val = grads_t[:-n_val], grads_t[-n_val:]

train_set = TensorDataset(theta_train, F_train, grads_train)
val_set   = TensorDataset(theta_val,   F_val,   grads_val)

train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)

# ── Model ────────────────────────────────────────────────────────────
n_maturities = len(cfg["data"]["maturities"])
model = SurrogateMLP(
    input_dim   = 8,
    output_dim  = n_maturities,
    hidden_dims = cfg["model"]["hidden_dims"],
    activation  = cfg["model"]["activation"],
)

# ── Train ────────────────────────────────────────────────────────────
trainer = SchwartzSmithTrainer(
    model,
    alpha=cfg["training"]["differential_loss_alpha"],
    beta=cfg["training"]["differential_loss_beta"],
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
    checkpoint_dir=ckpt_dir,
    grad_keys=grad_keys,
)
history = trainer.train(train_loader, val_loader, n_epochs=cfg["training"]["n_epochs"])

# ── Validation plots ────────────────────────────────────────────────
import matplotlib.pyplot as plt
from src.utils.plotting import plot_training_loss, plot_surrogate_accuracy

fig = plot_training_loss(history)
fig.savefig(plot_dir / "training_loss.png", dpi=150, bbox_inches="tight")

model.eval()
test_theta = torch.tensor(
    (data["theta"][:200] - theta_mean) / theta_std, dtype=torch.float32
)
with torch.no_grad():
    F_pred_norm = model(test_theta).numpy()
F_pred = F_pred_norm * F_std + F_mean

fig = plot_surrogate_accuracy(data["F"][:200], F_pred, maturities)
fig.savefig(plot_dir / "surrogate_accuracy.png", dpi=150, bbox_inches="tight")

print(f"\nAll plots saved to {plot_dir}")
print(f"Best checkpoint at {ckpt_dir}/best_model.pt")