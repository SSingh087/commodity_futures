import sys, os
sys.path.insert(0, os.path.abspath('../'))

from __training_imports__ import *
from models import SurrogateMLP
from trainer import SchwartzSmithTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="../config/SchwartzSmithFWD.yaml")
args = parser.parse_args()

cfg_path = args.config
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)


plot_dir = Path(cfg["output"]["plot_dir"])
ckpt_dir = Path(cfg["output"]["checkpoint_dir"])

# load data 
data = torch.load(cfg["data"]["save_path"], weights_only=False)

theta = (data["theta"] - data["theta_mean"]) / data["theta_std"]
F = (data["F"] - data["F_mean"]) / data["F_std"]

# Convert to tensors
theta = torch.tensor(theta, dtype=torch.float32)
F = torch.tensor(F, dtype=torch.float32)

n_val = int(cfg["training"]["val_fraction"] * len(theta))
theta_train, theta_val = theta[:-n_val], theta[-n_val:]
F_train, F_val = F[:-n_val], F[-n_val:]


train_set = TensorDataset(theta_train, F_train)
val_set = TensorDataset(theta_val,   F_val)

train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)

n_maturities = len(cfg["data"]["maturities"])   # 11 in your yaml

model = SurrogateMLP(
    input_dim  = 8,             # 8 parameters, no T
    output_dim = n_maturities,  # 11 maturity outputs
    hidden_dims = cfg["model"]["hidden_dims"],   # [64, 128, 128, 64]
    activation  = cfg["model"]["activation"],    # gelu
)

trainer = SchwartzSmithTrainer(
    model,
    alpha=cfg["training"]["differential_loss_alpha"],
    beta=cfg["training"]["differential_loss_beta"],
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
    checkpoint_dir=ckpt_dir,
)
history = trainer.train(
    train_loader, val_loader,
    n_epochs=cfg["training"]["n_epochs"]
)

# ── 6. Validation plots ───────────────────────────────────────────────
import matplotlib.pyplot as plt
from src.utils.plotting import plot_training_loss

fig = plot_training_loss(history)
fig.savefig(plot_dir / "training_loss.png", dpi=150, bbox_inches="tight")

# Predict on test set
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
