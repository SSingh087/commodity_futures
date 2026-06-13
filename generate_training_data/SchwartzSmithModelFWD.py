import sys, os
sys.path.insert(0, os.path.abspath('../'))
from __training_imports__ import *
from schwartz_smith import SSParams, SchwartzSmithModel

class SSForwardCurveDatasetNormalised(TorchDataset):
    """
    PyTorch Dataset for Schwartz-Smith forward curve data.
    Each item: (theta_normalised, F_normalised, grad_dict)
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        theta_mean: np.ndarray,
        theta_std: np.ndarray,
        F_mean: np.ndarray,
        F_std: np.ndarray,
    ):
        self.theta = torch.tensor(
            (data["theta"] - theta_mean) / theta_std, dtype=torch.float32
        )
        self.F = torch.tensor(
            (data["F"] - F_mean) / F_std, dtype=torch.float32
        )
        # Store available gradient targets
        self.grads = {
            k: torch.tensor(data[k] / F_std, dtype=torch.float32)
            for k in ["dF_dkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"]
            if k in data
        }

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, idx: int) -> tuple:
        grads = {k: v[idx] for k, v in self.grads.items()}
        return self.theta[idx], self.F[idx], grads


def sample_parameters(
    n: int,
    param_bounds: np.ndarray,
    state_bounds: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    oversample_degenerate: bool = True,
    degenerate_fraction: float = 0.15,
    
) -> np.ndarray:
    """
    Sample Schwartz-Smith parameters from the prior.

    Parameters
    ----------
    n                      : number of samples
    oversample_degenerate  : whether to oversample near kappa~0, rho~±1
    degenerate_fraction    : fraction of samples from degenerate regions

    Returns
    -------
    theta : (n, 8) array  [kappa, mu_xi, sigma_chi, sigma_xi, rho, lambda_chi, chi0, xi0]
    """
    rng = rng or np.random.default_rng()

    # Main uniform draw
    all_bounds = np.vstack([param_bounds, state_bounds])  # (8, 2)
    theta = rng.uniform(all_bounds[:, 0], all_bounds[:, 1], size=(n, 8))

    if oversample_degenerate:
        n_degen = int(n * degenerate_fraction)
        idx = rng.integers(0, n, size=n_degen)

        # Small kappa region (slow mean reversion — hard to constrain)
        n_small_k = n_degen // 2
        theta[idx[:n_small_k], 0] = rng.uniform(0.05, 0.5, n_small_k)

        # Near-unit correlation
        n_corr = n_degen - n_small_k
        theta[idx[n_small_k:], 4] = rng.choice([-1, 1], n_corr) * rng.uniform(0.7, 0.95, n_corr)

    return theta


def generate_forward_curve_data(
    n_samples: int,
    param_bounds: np.ndarray,
    state_bounds: np.ndarray,
    maturities: np.ndarray,
    compute_gradients: bool = True,
    rng: Optional[np.random.Generator] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Generate forward curve data using the Schwartz-Smith model.
    Returns
    -------
    dict with keys:
      "theta"       : (N, 8)    parameter vectors
      "F"           : (N, M)    forward curves (futures prices)
      "ln_F"        : (N, M)    log-forward curves
      "dF_dkappa"   : (N, M)    analytical gradient w.r.t. kappa
      "dF_dsigma_chi": (N, M)
      "dF_dsigma_xi": (N, M)
      "dF_drho"     : (N, M)
      "maturities"  : (M,)
    """
    rng = rng or np.random.default_rng(seed)
    theta = sample_parameters(n_samples, param_bounds=param_bounds, state_bounds=state_bounds, rng=rng)
    M = len(maturities)

    F = np.empty((n_samples, M))
    grads = {k: np.empty((n_samples, M)) for k in
                  ["dF_dkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"]}

    it = tqdm(range(n_samples), desc="Generating forward curves", disable=not verbose)
    for i in it:
        p = SSParams(
            kappa=theta[i, 0], mu_xi=theta[i, 1],
            sigma_chi=theta[i, 2], sigma_xi=theta[i, 3],
            rho=theta[i, 4], lambda_chi=theta[i, 5],
            chi0=theta[i, 6], xi0=theta[i, 7],
        )
        model  = SchwartzSmithModel(p)
        F[i]   = model.forward_curve(maturities)

        if compute_gradients:
            g = model.all_gradients(maturities)
            grads["dF_dkappa"][i]    = g["kappa"]
            grads["dF_dsigma_chi"][i]= g["sigma_chi"]
            grads["dF_dsigma_xi"][i] = g["sigma_xi"]
            grads["dF_drho"][i]      = g["rho"]

    result = {"theta": theta, "F": F, "ln_F": np.log(F), "maturities": maturities}
    if compute_gradients:
        result.update(grads)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/SchwartzSmithFWD.yaml")
    args = parser.parse_args()
    cfg_path = args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)


    # get the maturities and number of training samples from config
    maturities = np.array(cfg["data"]["maturities"])
    rng = np.random.default_rng(cfg["training"]["seed"])
    n_samples = cfg["training"]["n_samples"]

    PARAM_BOUNDS = np.array([
        cfg["data"]["priors"]["kappa"],
        cfg["data"]["priors"]["mu_xi"],
        cfg["data"]["priors"]["sigma_chi"],
        cfg["data"]["priors"]["sigma_xi"],
        cfg["data"]["priors"]["rho"],
        cfg["data"]["priors"]["lambda_chi"],
    ])

    STATE_BOUNDS = np.array([
        cfg["data"]["priors"]["chi0"],
        cfg["data"]["priors"]["xi0"],
    ])

    print(f"\nGenerating {n_samples:,} training samples...")
    data = generate_forward_curve_data(n_samples=n_samples, maturities=maturities, param_bounds=PARAM_BOUNDS, state_bounds=STATE_BOUNDS, compute_gradients=True, rng=rng, seed=42, verbose=True)
    
    print(f"Forward curve range: [{data['F'].min():.2f}, {data['F'].max():.2f}]")
    print(f"Log-forward curve range: [{data['ln_F'].min():.2f}, {data['ln_F'].max():.2f}]")
    print(f"Sample parameter ranges:")
    for i, name in enumerate(cfg["param_names"]):
        print(f"  {name}: [{data['theta'][:, i].min():.4f}, {data['theta'][:, i].max():.4f}]")  


    # normalize the forward curves and gradients
    theta_mean = data["theta"].mean(axis=0)
    theta_std  = data["theta"].std(axis=0) + 1e-8
    F_mean     = data["F"].mean(axis=0)
    F_std      = data["F"].std(axis=0) + 1e-8

    dataset = SSForwardCurveDatasetNormalised(data, theta_mean, theta_std, F_mean, F_std)
    # save the dataset and normalization stats as a torch file

    save_path = cfg["data"]["save_path"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "dataset": dataset,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "F_mean": F_mean,
        "F_std": F_std,
        "maturities": maturities,
    }, save_path)



