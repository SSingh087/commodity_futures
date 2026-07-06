import sys, os
sys.path.insert(0, os.path.abspath('./'))
from __training_imports__ import *
from schwartz_smith import SSParams, SchwartzSmithModel

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

    Returns
    -------
    theta : (n, 8) array  [kappa, mu_xi, sigma_chi, sigma_xi, rho, lambda_chi, chi0, xi0]
      Always in RAW physical units — kappa here is the actual mean-reversion
      rate, never log-transformed. The log transform is applied downstream,
      only to the copy that feeds the network.
    """
    rng = rng or np.random.default_rng()

    all_bounds = np.vstack([param_bounds, state_bounds])  # (8, 2)
    theta = rng.uniform(all_bounds[:, 0], all_bounds[:, 1], size=(n, 8))

    if oversample_degenerate:
        n_degen = int(n * degenerate_fraction)
        idx = rng.integers(0, n, size=n_degen)

        # Small kappa region (slow mean reversion — hard to constrain).
        # Hence oversample the slow-mean-reversion corner of the prior.
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
      "theta_raw"     : (N, 8)    RAW physical parameter vectors [kappa, mu_xi, sigma_chi, sigma_xi, rho, lambda_chi, chi0, xi0]
      "theta_net"     : (N, 8)    NETWORK-FACING parameter vectors [ln(kappa), mu_xi, sigma_chi, sigma_xi, rho, lambda_chi, chi0, xi0]
      "F"             : (N, M)    forward curves (futures prices) [F(T_1), ..., F(T_M)]
      "ln_F"          : (N, M)    log-forward curves [ln F(T_1), ..., ln F(T_M)]
      "dF_dlnkappa"   : (N, M)    dF/d(ln kappa) = kappa * dF/dkappa  (reparametrized)
      "dF_dsigma_chi" : (N, M)    dF/dsigma_chi  (raw)
      "dF_dsigma_xi"  : (N, M)    dF/dsigma_xi  (raw)
      "dF_drho"       : (N, M)    dF/drho  (raw)
      "maturities"    : (M,)      maturities T_1, ..., T_M
    """
    rng = rng or np.random.default_rng(seed)
    theta_raw = sample_parameters(
        n_samples, param_bounds=param_bounds, state_bounds=state_bounds, rng=rng
    )
    M = len(maturities)

    F = np.empty((n_samples, M))
    grads = {k: np.empty((n_samples, M)) for k in
                  ["dF_dlnkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"]}

    it = tqdm(range(n_samples), desc="Generating forward curves", disable=not verbose)
    for i in it:
        # Physical model call — ALWAYS raw kappa. The Schwartz-Smith formula
        # A(tau) has no concept of ln(kappa); it's defined in the physical
        # rate, the same way a ringdown model's A(t) = A0 exp(-t/tau) is
        # defined in the physical damping time tau, not ln(tau).
        p = SSParams(
            kappa=theta_raw[i, 0], mu_xi=theta_raw[i, 1],
            sigma_chi=theta_raw[i, 2], sigma_xi=theta_raw[i, 3],
            rho=theta_raw[i, 4], lambda_chi=theta_raw[i, 5],
            chi0=theta_raw[i, 6], xi0=theta_raw[i, 7],
        )
        model  = SchwartzSmithModel(p)
        F[i]   = model.forward_curve(maturities)

        if compute_gradients:
            g = model.all_gradients(maturities)   # raw dF/dtheta, all in physical units

            # ── The reparametrization step ──────────────────────────────
            # d F/d(ln kappa) = kappa * dF/dkappa   (chain rule)
            # This is the exact analogue of converting a Fisher-matrix row
            # from d/d(mass) to d/d(ln mass) — it cancels the intrinsic
            # 1/kappa divergence in the raw gradient (which comes from A(tau)
            # containing terms like sigma_chi^2*(1-e^{-2*kappa*tau})/(4*kappa)),
            # the same way working in ln(mass) tames Fisher-matrix
            # conditioning near extremal or near-degenerate corners of a
            # GW parameter space, rather than leaving a coordinate whose
            # sensitivity diverges as a rate parameter approaches zero.
            grads["dF_dlnkappa"][i]   = g["kappa"] * theta_raw[i, 0]
            grads["dF_dsigma_chi"][i] = g["sigma_chi"]
            grads["dF_dsigma_xi"][i]  = g["sigma_xi"]
            grads["dF_drho"][i]       = g["rho"]

    # ── Build the network-facing theta: same as theta_raw, except kappa
    #    column is replaced by ln(kappa) ─────────────────────────────────
    theta_net = theta_raw.copy()
    theta_net[:, 0] = np.log(theta_raw[:, 0])

    result = {
        "theta_raw": theta_raw,
        "theta_net": theta_net,
        "F": F,
        "ln_F": np.log(F),
        "maturities": maturities,
    }
    if compute_gradients:
        result.update(grads)
    return result


def fd_check_lnkappa_gradient(
    theta_raw_sample: np.ndarray,
    maturities: np.ndarray,
    eps: float = 1e-5,
    rtol: float = 1e-3,
) -> None:
    """
    Finite-difference check specifically for the reparametrized target:
    confirms dF/d(ln kappa) computed analytically (kappa * dF/dkappa) agrees
    with a numerical derivative taken directly in ln(kappa) space.
    """
    p_dict = dict(kappa=theta_raw_sample[0], mu_xi=theta_raw_sample[1],
                  sigma_chi=theta_raw_sample[2], sigma_xi=theta_raw_sample[3],
                  rho=theta_raw_sample[4], lambda_chi=theta_raw_sample[5],
                  chi0=theta_raw_sample[6], xi0=theta_raw_sample[7])

    ln_kappa = np.log(p_dict["kappa"])

    def F_of_lnkappa(lnk):
        pd = dict(p_dict); pd["kappa"] = np.exp(lnk)
        return SchwartzSmithModel(SSParams(**pd)).forward_curve(maturities)

    F_up = F_of_lnkappa(ln_kappa + eps)
    F_dn = F_of_lnkappa(ln_kappa - eps)
    fd_dlnkappa = (F_up - F_dn) / (2 * eps)

    model = SchwartzSmithModel(SSParams(**p_dict))
    analytical_dlnkappa = model.all_gradients(maturities)["kappa"] * p_dict["kappa"]

    rel_err = np.abs(analytical_dlnkappa - fd_dlnkappa) / (np.abs(fd_dlnkappa) + 1e-10)
    status = "OK" if rel_err.max() < rtol else "FAIL"
    print(f"  dF/d(ln kappa) FD check: max rel err = {rel_err.max():.2e}   [{status}]")
    assert rel_err.max() < rtol, "dF/d(ln kappa) reparametrization is wrong — check the multiply-by-kappa step"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/SchwartzSmithFWD.yaml")
    args = parser.parse_args()
    cfg_path = args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

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

    # ── Sanity check the reparametrized gradient BEFORE burning hours
    #    generating 500k samples — same discipline as validating a
    #    Jacobian-transformed Fisher element before a full PE run. ───────
    print("Running dF/d(ln kappa) sanity check on 5 random draws...")
    check_rng = np.random.default_rng(0)
    check_theta = sample_parameters(5, PARAM_BOUNDS, STATE_BOUNDS, rng=check_rng)
    for i in range(5):
        fd_check_lnkappa_gradient(check_theta[i], maturities)
    print("✓ dF/d(ln kappa) reparametrization verified.\n")

    print(f"Generating {n_samples:,} training samples...")
    data = generate_forward_curve_data(
        n_samples=n_samples, maturities=maturities,
        param_bounds=PARAM_BOUNDS, state_bounds=STATE_BOUNDS,
        compute_gradients=True, rng=rng, seed=42, verbose=True,
    )

    print(f"Forward curve range: [{data['F'].min():.2f}, {data['F'].max():.2f}]")
    print(f"Log-forward curve range: [{data['ln_F'].min():.2f}, {data['ln_F'].max():.2f}]")
    print("Sample parameter ranges (physical units, theta_raw):")
    for i, name in enumerate(cfg["param_names"]):
        print(f"  {name}: [{data['theta_raw'][:, i].min():.4f}, {data['theta_raw'][:, i].max():.4f}]")
    print(f"  ln(kappa) [theta_net col 0]: "
          f"[{data['theta_net'][:, 0].min():.4f}, {data['theta_net'][:, 0].max():.4f}]")

    # ── Normalise theta_net (the network-facing array) and F ───────────
    # theta_mean/theta_std are computed on theta_net, NOT theta_raw — this
    # is the load-bearing step. Everything downstream (stage1_train.py's
    # chain-rule scaling of grads_norm) is already generic in terms of
    # whichever theta/theta_std it's handed, so getting THIS step right is
    # what makes the whole reparametrization self-consistent, the same way
    # getting the coordinate choice right at the sampler-configuration
    # stage is what makes a Fisher-matrix-based proposal distribution
    # well-conditioned throughout an MCMC run.
    theta_net_mean = data["theta_net"].mean(axis=0)
    theta_net_std  = data["theta_net"].std(axis=0) + 1e-8
    F_mean         = data["F"].mean(axis=0)
    F_std          = data["F"].std(axis=0) + 1e-8

    save_path = cfg["data"]["save_path"]

    # Stack gradients — order matters, document it. Note "dF_dlnkappa" name,
    # not "dF_dkappa": deliberately renamed so any script still referencing
    # the old key fails LOUDLY (KeyError) rather than silently training as
    # if alpha=0 again, the same class of silent bug we already hit twice.
    grad_keys = ["dF_dlnkappa", "dF_dsigma_chi", "dF_dsigma_xi", "dF_drho"]
    grads_stacked = np.stack([data[k] for k in grad_keys], axis=-1)  # (N, M, 4)

    file_path = os.path.join(save_path, "SchwartzSmithFWD_dataset.pt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save({
        "theta_raw":     data["theta_raw"],    # physical units — for plots/Stage 3 interpretation
        "theta_net":     data["theta_net"],    # ln(kappa) in col 0 — what the network trains on
        "F":             data["F"],
        "grads_stacked": grads_stacked,        # raw (N, M, 4), col 0 = dF/d(ln kappa)
        "grad_keys":     grad_keys,
        "theta_mean":    theta_net_mean,        # computed on theta_net
        "theta_std":     theta_net_std,         # computed on theta_net
        "F_mean":        F_mean,
        "F_std":         F_std,
        "maturities":    maturities,
    }, file_path)

    print(f"\nSaved to {file_path}")
    # print("NOTE: 'theta_mean'/'theta_std' are in (ln kappa, mu_xi, sigma_chi, ...) space.")
    # print("      Use 'theta_raw' for anything needing physical kappa units.")