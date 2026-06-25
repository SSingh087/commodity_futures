import sys, os
sys.path.insert(0, os.path.abspath('./'))
from __training_imports__ import *
from schwartz_smith import SSParams, SchwartzSmithModel, price_european_call_on_futures

import numpy as np
from pathlib import Path
from tqdm import tqdm

def sample_one(rng: np.random.Generator, PRIOR: dict, CONTRACT: dict, TAU_EXTRA: list) -> dict:
    """
    Sample one (theta, contract) pair and price it via MC.

    Returns a dict with X (8 inputs), V (price), V_std (stderr).
    Returns None if the pricing call fails for any reason.
    """
    # ── sample model parameters ──────────────────────────────────────────
    def u(key, lo, hi):
        return rng.uniform(lo, hi)

    params = SSParams(
        kappa      = u("kappa", *PRIOR["kappa"]),
        mu_xi      = u("mu_xi", *PRIOR["mu_xi"]),
        sigma_chi  = u("sigma_chi", *PRIOR["sigma_chi"]),
        sigma_xi   = u("sigma_xi", *PRIOR["sigma_xi"]),
        rho        = u("rho", *PRIOR["rho"]),
        lambda_chi = u("lambda_chi", *PRIOR["lambda_chi"]),
        chi0       = u("chi0", *PRIOR["chi0"]),
        xi0        = u("xi0", *PRIOR["xi0"]),
    )

    # ── sample contract parameters ────────────────────────────────────────
    moneyness = rng.uniform(*CONTRACT["moneyness"])
    T_option  = rng.uniform(*CONTRACT["T_option"])
    T_futures = T_option + rng.uniform(*TAU_EXTRA)   # always > T_option
    r         = rng.uniform(*CONTRACT["r"])

    try:
        price, stderr = price_european_call_on_futures(
            params         = params,
            strike_ratio   = moneyness,
            T_option       = T_option,
            T_futures      = T_futures,
            risk_free_rate = r,
            n_paths        = 20000,
            n_steps        = 252,
            seed           = None,
        )
    except Exception:
        return None
 
    if not np.isfinite(price) or not np.isfinite(stderr):
        return None
    if price < 0 or stderr < 0:
        return None
 
    X = np.array([
        params.kappa,
        params.sigma_chi,
        params.sigma_xi,
        params.rho,
        moneyness,
        T_option,
        T_futures,
        r,
    ], dtype=np.float32)
 
    return {"X": X, "V": float(price), "V_std": float(stderr)}



def generate(PRIOR: dict, CONTRACT: dict, TAU_EXTRA: list, n_samples: int, seed: int = 42, save_path: str = "./SchwartzSmithFWD_dataset.pt") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate n_samples (input, price, stderr).
    """
    rng = np.random.default_rng(seed)

    X_list     = []
    V_list     = []
    V_std_list = []

    n_attempts = 0
    pbar = tqdm(total=n_samples, desc="Pricing options via MC")

    while len(X_list) < n_samples:
        n_attempts += 1
        result = sample_one(rng, PRIOR, CONTRACT, TAU_EXTRA)
        if result is None:
            continue
        X_list.append(result["X"])
        V_list.append(result["V"])
        V_std_list.append(result["V_std"])
        pbar.update(1)

    pbar.close()

    X     = np.stack(X_list)
    V     = np.array(V_list,     dtype=np.float32)
    V_std = np.array(V_std_list, dtype=np.float32)
 
    col_names = ["kappa", "sigma_chi", "sigma_xi", "rho",
                 "moneyness", "T_option", "T_futures", "r"]


    # ── normalisation stats (computed here, stored for training) ──────────
    X_mean = X.mean(axis=0).astype(np.float32)
    X_std  = X.std(axis=0).astype(np.float32) + 1e-8
    V_mean = float(V.mean())
    V_std_norm = float(V.std()) + 1e-8

    print(f"\n  Samples generated : {n_samples:,}  (from {n_attempts:,} attempts)")
    print(f"  Price range       : [{V.min():.4f}, {V.max():.4f}]")
    print(f"  Mean price        : {V.mean():.4f}")
    print(f"  Mean MC stderr    : {V_std.mean():.5f}   (sigma_label)")
    print(f"  Mean SNR          : {(V / (V_std + 1e-8)).mean():.1f}   (V / sigma_label)")
 
    # ── save as .pt ───────────────────────────────────────────────────────
    torch.save({
        "X"         : torch.tensor(X),
        "V"         : torch.tensor(V),
        "V_std"     : torch.tensor(V_std),
        "col_names" : col_names,
        "X_mean"    : torch.tensor(X_mean),
        "X_std"     : torch.tensor(X_std),
        "V_mean"    : V_mean,
        "V_std_norm": V_std_norm,
    }, save_path)
    print(f"  Saved to: {save_path}")
 
    return X, V, V_std, col_names



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/SchwartzSmithFWD.yaml")
    args = parser.parse_args()
    cfg_path = args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["training"]["seed"])
    n_samples = cfg["training"]["n_samples"]

    PARAM_BOUNDS = np.array([
        cfg["data"]["priors"]["kappa"],
        cfg["data"]["priors"]["mu_xi"],
        cfg["data"]["priors"]["sigma_chi"],
        cfg["data"]["priors"]["sigma_xi"],
        cfg["data"]["priors"]["rho"],
        cfg["data"]["priors"]["lambda_chi"],
        cfg["data"]["priors"]["chi0"],
        cfg["data"]["priors"]["xi0"],
    ])

    PRIOR = {
        "kappa"     : PARAM_BOUNDS[0],
        "mu_xi"     : PARAM_BOUNDS[1],
        "sigma_chi" : PARAM_BOUNDS[2],
        "sigma_xi"  : PARAM_BOUNDS[3],
        "rho"       : PARAM_BOUNDS[4],
        "lambda_chi": PARAM_BOUNDS[5],
        "chi0"      : PARAM_BOUNDS[6],
        "xi0"       : PARAM_BOUNDS[7],
    }
    
    CONTRACT = {
        "moneyness" : cfg["data"]["contract"]["moneyness"],
        "T_option"  : cfg["data"]["contract"]["T_option"],
        "r"         : cfg["data"]["contract"]["r"],
    }

    TAU_EXTRA = cfg["data"]["TAU_EXTRA"]


    # Start small — 2000 samples is enough to check everything works
    # before committing to the full 200k run (which takes hours)
    file_path = os.path.join(cfg["data"]["save_path"], "options_on_futures_data.pt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    X, V, V_std, col_names = generate(
    PRIOR, CONTRACT, TAU_EXTRA,
    n_samples = cfg["training"]["n_samples"],
    seed      = cfg["training"]["seed"],
    save_path = file_path,
    )
