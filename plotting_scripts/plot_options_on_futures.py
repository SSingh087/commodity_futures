import sys, os
sys.path.insert(0, os.path.abspath('../'))
from __plotting_imports__ import *
import torch

from schwartz_smith import SSParams, SchwartzSmithModel

def plot_coverage(X, V, V_std, col_names, save_dir="../plots"):
    """
    Two sets of plots:

    Plot 1 - Input marginals (one histogram per input dimension)
             Goal: each should look roughly uniform across its prior range.
             If any is skewed or has gaps, something is wrong with sampling
             or too many samples are being rejected.

    Plot 2 - Price and label noise diagnostics
             V histogram          : are prices spread across a useful range?
             V_std histogram      : is label noise consistently small?
             V vs moneyness       : price should fall smoothly with moneyness
             V_std vs V           : noise should be proportional to price
                                    (heteroscedastic - bigger prices, bigger noise)

    GW analogy: this is your injection parameter space coverage check -
    the same plot you'd make to confirm your waveform bank uniformly
    covers (m1, m2, chi) before trusting the surrogate.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Plot 1: input marginals ───────────────────────────────────────────
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7))
    axes = axes.flatten()

    for i, name in enumerate(col_names):
        axes[i].hist(X[:, i], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
        axes[i].set_title(name, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("count")
        # mark the mean
        axes[i].axvline(X[:, i].mean(), color="red", lw=1.5, ls="--", label="mean")
        axes[i].legend(fontsize=7)

    fig.suptitle(
        "Input marginals - should be uniform across prior range\n"
        "(red dashed = mean, should be near centre of range)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out1 = os.path.join(save_dir, "option_input_marginals.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out1}")

    # ── Plot 2: price and noise diagnostics ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 2a - price histogram
    axes[0, 0].hist(V, bins=50, color="#55A868", edgecolor="white", linewidth=0.4)
    axes[0, 0].set_xlabel("Option price V")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Price distribution\n(should be right-skewed, no spike at 0)")

    # 2b - label noise histogram
    axes[0, 1].hist(V_std, bins=50, color="#DD8452", edgecolor="white", linewidth=0.4)
    axes[0, 1].set_xlabel("MC stderr  σ_label")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Label noise distribution\n(should be small relative to V)")

    # 2c - price vs moneyness  (moneyness is col index 4)
    m_idx = col_names.index("moneyness")
    sort_idx = np.argsort(X[:, m_idx])
    # thin to 2000 points for scatter clarity
    thin = np.linspace(0, len(sort_idx)-1, min(2000, len(sort_idx)), dtype=int)
    axes[1, 0].scatter(
        X[sort_idx[thin], m_idx], V[sort_idx[thin]],
        alpha=0.3, s=8, color="#4C72B0"
    )
    axes[1, 0].set_xlabel("Moneyness  K/F")
    axes[1, 0].set_ylabel("Option price V")
    axes[1, 0].set_title("Price vs moneyness\n(should decrease as moneyness increases - OTM cheaper)")

    # 2d - V_std vs V  (noise proportional to price = heteroscedastic)
    thin2 = np.random.choice(len(V), min(2000, len(V)), replace=False)
    axes[1, 1].scatter(V[thin2], V_std[thin2], alpha=0.3, s=8, color="#C44E52")
    axes[1, 1].set_xlabel("Option price V")
    axes[1, 1].set_ylabel("MC stderr  σ_label")
    axes[1, 1].set_title("Label noise vs price\n(should be proportional - heteroscedastic)")

    fig.tight_layout()
    out2 = os.path.join(save_dir, "option_price_diagnostics.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out2}")

if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/SchwartzSmithFWD.yaml")
    args = parser.parse_args()
    cfg_path = args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # load saved data
    torch_file = os.path.join(cfg["data"]["save_path"], "options_on_futures_data.pt")
    torch_data = torch.load(torch_file)
    X, V, V_std = torch_data["X"], torch_data["V"], torch_data["V_std"]
    col_names = torch_data["col_names"]
    
    plot_coverage(
        X, V, V_std, col_names,
        save_dir=cfg["data"].get("plot_dir", "../plots"),
    )