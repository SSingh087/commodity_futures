import sys, os
sys.path.insert(0, os.path.abspath('./'))
from __training_imports__ import *
from schwartz_smith import SSParams, price_european_call_on_futures

import numpy as np
from tqdm import tqdm


def sample_one(rng: np.random.Generator, PRIOR: dict, CONTRACT: dict, TAU_EXTRA: list) -> dict:
    """
    Sample one (theta, contract) pair and price it via MC.

    Returns a dict with X (8 inputs), V (price), V_std (stderr).
    Returns None if the pricing call fails for any reason.
    """
    def u(key, lo, hi):
        return rng.uniform(lo, hi)

    params = SSParams(
        kappa      = u("kappa",      *PRIOR["kappa"]),
        mu_xi      = u("mu_xi",      *PRIOR["mu_xi"]),
        sigma_chi  = u("sigma_chi",  *PRIOR["sigma_chi"]),
        sigma_xi   = u("sigma_xi",   *PRIOR["sigma_xi"]),
        rho        = u("rho",        *PRIOR["rho"]),
        lambda_chi = u("lambda_chi", *PRIOR["lambda_chi"]),
        chi0       = u("chi0",       *PRIOR["chi0"]),
        xi0        = u("xi0",        *PRIOR["xi0"]),
    )

    moneyness = rng.uniform(*CONTRACT["moneyness"])
    T_option  = rng.uniform(*CONTRACT["T_option"])
    T_futures = T_option + rng.uniform(*TAU_EXTRA)
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


def generate(
    PRIOR: dict,
    CONTRACT: dict,
    TAU_EXTRA: list,
    n_samples: int,
    rng: np.random.Generator,
    save_path: str,
) -> tuple:
    """
    Generate n_samples (input, price, stderr) using a pre-built RNG.
    The caller is responsible for constructing a job-specific RNG so that
    parallel workers never share state.
    """
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

    X_mean     = X.mean(axis=0).astype(np.float32)
    X_std      = X.std(axis=0).astype(np.float32) + 1e-8
    V_mean     = float(V.mean())
    V_std_norm = float(V.std()) + 1e-8

    print(f"\n  Samples generated : {n_samples:,}  (from {n_attempts:,} attempts)")
    print(f"  Price range       : [{V.min():.4f}, {V.max():.4f}]")
    print(f"  Mean price        : {V.mean():.4f}")
    print(f"  Mean MC stderr    : {V_std.mean():.5f}   (sigma_label)")
    print(f"  Mean SNR          : {(V / (V_std + 1e-8)).mean():.1f}   (V / sigma_label)")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
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
    parser.add_argument("--config",   default="./config/SchwartzSmithFWD.yaml")
    parser.add_argument("--job_id",   type=int, default=0,
                        help="0-indexed chunk ID (set by HTCondor via $(Process))")
    parser.add_argument("--n_jobs",   type=int, default=1,
                        help="Total number of parallel chunks")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Parallel-safe seeding via SeedSequence ────────────────────────────
    # SeedSequence.spawn(n_jobs) produces n_jobs independent child sequences
    # with non-overlapping internal state — analogous to choosing orthogonal
    # waveform polarisations so matched-filter outputs are uncorrelated.
    base_seed = cfg["training"]["seed"]
    ss        = np.random.SeedSequence(base_seed)
    child_seq = ss.spawn(args.n_jobs)[args.job_id]
    rng       = np.random.default_rng(child_seq)

    # ── Split total sample count across jobs ──────────────────────────────
    total_samples  = cfg["training"]["n_samples"]
    samples_per_job = total_samples // args.n_jobs
    # Give any remainder to the last job
    if args.job_id == args.n_jobs - 1:
        samples_per_job += total_samples % args.n_jobs

    PRIOR = {k: cfg["data"]["priors"][k] for k in
             ["kappa", "mu_xi", "sigma_chi", "sigma_xi",
              "rho", "lambda_chi", "chi0", "xi0"]}

    CONTRACT = {
        "moneyness" : cfg["data"]["contract"]["moneyness"],
        "T_option"  : cfg["data"]["contract"]["T_option"],
        "r"         : cfg["data"]["contract"]["r"],
    }
    TAU_EXTRA = cfg["data"]["TAU_EXTRA"]

    # Each chunk saves to its own file: options_on_futures_data_chunk_003.pt
    save_dir  = cfg["data"]["save_path"]
    file_path = os.path.join(
        save_dir,
        f"options_on_futures_data_chunk_{args.job_id:03d}.pt"
    )

    print(f"\n[Job {args.job_id}/{args.n_jobs}]  "
          f"generating {samples_per_job:,} / {total_samples:,} samples")

    generate(
        PRIOR, CONTRACT, TAU_EXTRA,
        n_samples = samples_per_job,
        rng       = rng,
        save_path = file_path,
    )