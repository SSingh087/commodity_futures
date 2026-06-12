from __future__ import annotations
from .__plotting_imports__ import *


# ── Stage 1 ───────────────────────────────────────────────────────────────

def plot_forward_curves(
    maturities: np.ndarray,
    param_sets: list[dict],
    title: str = "Schwartz-Smith Forward Curves",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot forward curves F(T) for multiple parameter sets.

    Shows how changing kappa (mean-reversion) and sigma shapes the curve.
    """
    from ..models.schwartz_smith import SSParams, SchwartzSmithModel

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: vary kappa, fix others
    ax = axes[0]
    kappas = [0.3, 1.0, 2.0, 4.0]
    for k, kap in enumerate(kappas):
        p = SSParams(kappa=kap, mu_xi=0.04, sigma_chi=0.3,
                     sigma_xi=0.2, rho=-0.3, lambda_chi=0.1,
                     chi0=0.1, xi0=3.5)
        F = SchwartzSmithModel(p).forward_curve(maturities)
        ax.plot(maturities, F, color=PALETTE[k], label=f"κ = {kap}")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Futures Price F(T)")
    ax.set_title("Effect of Mean-Reversion Speed κ")
    ax.legend(title="κ (1/year)")

    # Right: vary rho, fix others
    ax = axes[1]
    rhos = [-0.8, -0.3, 0.0, 0.5, 0.8]
    for k, rho in enumerate(rhos):
        p = SSParams(kappa=1.5, mu_xi=0.04, sigma_chi=0.3,
                     sigma_xi=0.2, rho=rho, lambda_chi=0.1,
                     chi0=0.05, xi0=3.5)
        F = SchwartzSmithModel(p).forward_curve(maturities)
        ax.plot(maturities, F, color=PALETTE[k], label=f"ρ = {rho:+.1f}")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Futures Price F(T)")
    ax.set_title("Effect of Correlation ρ")
    ax.legend(title="ρ")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()
    return fig


def plot_parameter_sensitivity(
    maturities: np.ndarray,
    base_params: "SSParams",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    4-panel plot showing how F(T) responds to each parameter ±1 std dev.
    Also shows the analytical gradient dF/dtheta for comparison with surrogate.

    GW analogy: this is the waveform derivative plot — dh/dM, dh/dchi —
    the key diagnostic for assessing parameter degeneracies.
    """
    from ..models.schwartz_smith import SSParams, SchwartzSmithModel

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    model0 = SchwartzSmithModel(base_params)
    F0 = model0.forward_curve(maturities)

    param_info = [
        ("kappa",     "κ",         0.5, base_params.kappa),
        ("sigma_chi", "σ_χ",       0.05, base_params.sigma_chi),
        ("sigma_xi",  "σ_ξ",       0.05, base_params.sigma_xi),
        ("rho",       "ρ",         0.15, base_params.rho),
        ("lambda_chi","λ_χ",       0.1,  base_params.lambda_chi),
        ("mu_xi",     "μ_ξ",       0.01, base_params.mu_xi),
    ]
    for ax, (attr, label, delta, base_val) in zip(axes, param_info):
        for sign, col, ls in [(+1, SURROGATE_RED, "-"), (-1, GW_BLUE, "--")]:
            p2 = SSParams(
                kappa=base_params.kappa, mu_xi=base_params.mu_xi,
                sigma_chi=base_params.sigma_chi, sigma_xi=base_params.sigma_xi,
                rho=base_params.rho, lambda_chi=base_params.lambda_chi,
                chi0=base_params.chi0, xi0=base_params.xi0,
            )
            setattr(p2, attr, base_val + sign * delta)
            F2 = SchwartzSmithModel(p2).forward_curve(maturities)
            ax.plot(maturities, F2, color=col, ls=ls,
                    label=f"{label} {'+' if sign>0 else '-'} {delta:.2g}")

        ax.plot(maturities, F0, "k-", lw=2, label="Baseline", zorder=5)
        ax.set_title(f"Sensitivity to {label}")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("F(T)")
        ax.legend(fontsize=9)

    fig.suptitle("Forward Curve Parameter Sensitivity\n"
                 "(analogue of dh/dθ in GW matched filtering)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_analytical_gradients(
    maturities: np.ndarray,
    params: "SSParams",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot dF/dkappa, dF/dsigma_chi, dF/dsigma_xi, dF/drho.

    These are the targets for the differential loss — what the surrogate
    must learn to match, in addition to F itself.
    """
    from ..models.schwartz_smith import SchwartzSmithModel

    set_style()
    model = SchwartzSmithModel(params)
    grads = model.all_gradients(maturities)

    labels = {
        "kappa":     r"$\partial F / \partial \kappa$",
        "sigma_chi": r"$\partial F / \partial \sigma_\chi$",
        "sigma_xi":  r"$\partial F / \partial \sigma_\xi$",
        "rho":       r"$\partial F / \partial \rho$",
    }
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for ax, (key, label) in zip(axes, labels.items()):
        ax.plot(maturities, grads[key], color=STAGE_COLORS[1], lw=2.5)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Maturity (years)")
        ax.fill_between(maturities, grads[key], alpha=0.15, color=STAGE_COLORS[1])

    axes[0].set_ylabel("dF/dθ  (price / parameter unit)")
    fig.suptitle("Analytical Forward Curve Gradients\n"
                 "(differential loss targets — equivalent to dh/dθ in GW)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


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


def plot_svd_reconstruction(
    maturities: np.ndarray,
    F_curves: np.ndarray,
    n_components_list: list[int] = (2, 4, 6, 8),
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Show how many SVD modes are needed to reconstruct the forward curve.

    GW analogy: ROQ basis truncation study — how many waveform modes
    are needed to represent the manifold to a given accuracy?
    """
    set_style()
    F_mean = F_curves.mean(axis=0)
    F_norm = (F_curves - F_mean) / (F_curves.std(axis=0) + 1e-8)
    U, s, Vt = np.linalg.svd(F_norm, full_matrices=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Variance explained
    var_explained = np.cumsum(s**2) / (s**2).sum()
    ax1.plot(np.arange(1, len(s)+1), var_explained * 100,
             "o-", color=STAGE_COLORS[1], ms=5)
    for k in n_components_list:
        ax1.axvline(k, ls="--", color="gray", alpha=0.5)
        ax1.axhline(var_explained[k-1]*100, ls="--", color="gray", alpha=0.5)
        ax1.text(k+0.1, var_explained[k-1]*100 - 2, f"{var_explained[k-1]*100:.1f}%",
                 fontsize=9, color="gray")
    ax1.set_xlabel("Number of SVD components K")
    ax1.set_ylabel("Cumulative variance explained (%)")
    ax1.set_title("SVD Compression\n(GW analogy: ROQ basis truncation)")
    ax1.set_xlim(0, 20)

    # Reconstruction example
    sample_idx = 0
    F_sample = F_curves[sample_idx]
    ax2.plot(maturities, F_sample, "k-", lw=2.5, label="True", zorder=5)
    colors = sns.color_palette("Blues_r", len(n_components_list))
    for col, K in zip(colors, n_components_list):
        coeffs = U[sample_idx, :K] * s[:K]
        F_rec  = coeffs @ Vt[:K] * (F_curves.std(axis=0) + 1e-8) + F_mean
        ax2.plot(maturities, F_rec, "--", color=col, lw=1.5, label=f"K={K}")
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("F(T)")
    ax2.set_title("Forward Curve SVD Reconstruction")
    ax2.legend()

    fig.tight_layout()
    return fig


# ── Stage 2 ───────────────────────────────────────────────────────────────

def plot_mc_convergence(
    params: "SSParams",
    option_params: dict,
    path_counts: list[int],
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Show MC option price convergence as a function of number of paths.
    Illustrates why we need a surrogate (seconds per evaluation → too slow for MCMC).
    """
    from ..models.schwartz_smith import price_european_call_on_futures
    set_style()
    prices, stderrs = [], []
    for n in path_counts:
        p, se = price_european_call_on_futures(
            params,
            option_params["moneyness"],
            option_params["T_option"],
            option_params["T_futures"],
            n_paths=n, seed=0
        )
        prices.append(p); stderrs.append(se)
    prices  = np.array(prices)
    stderrs = np.array(stderrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.errorbar(path_counts, prices, yerr=1.96*stderrs,
                 fmt="o-", color=STAGE_COLORS[2], capsize=5)
    ax1.axhline(prices[-1], ls="--", color="gray", label="Reference (N=100k)")
    ax1.set_xscale("log"); ax1.set_xlabel("MC paths N")
    ax1.set_ylabel("Option price"); ax1.set_title("MC Convergence")
    ax1.legend()

    ax2.loglog(path_counts, stderrs, "s-", color=SURROGATE_RED)
    x_theory = np.array(path_counts)
    ax2.loglog(x_theory, stderrs[0] * np.sqrt(path_counts[0] / x_theory),
               "--", color="gray", label="~1/√N")
    ax2.set_xlabel("MC paths N"); ax2.set_ylabel("Std error")
    ax2.set_title("MC Label Noise (σ ~ 1/√N)")
    ax2.legend()

    fig.suptitle("Stage 2: Monte Carlo Pricing — Why We Need a Surrogate",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_volatility_smile(
    params: "SSParams",
    maturities_fut: list[float],
    moneyness: np.ndarray,
    T_option: float = 0.5,
    n_paths: int = 30_000,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot the implied volatility smile for options at different moneyness levels.
    Illustrates the structure the surrogate must learn.
    """
    from ..models.schwartz_smith import price_european_call_on_futures, SchwartzSmithModel
    from scipy.stats import norm
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def bs_call(F, K, T, r, sigma):
        d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return np.exp(-r*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))

    def implied_vol(price, F, K, T, r):
        from scipy.optimize import brentq
        try:
            return brentq(lambda s: bs_call(F, K, T, r, s) - price, 0.01, 5.0, xtol=1e-6)
        except:
            return np.nan

    for j, T_fut in enumerate(maturities_fut[:2]):
        model = SchwartzSmithModel(params)
        F_atm = model.futures_price(np.array([T_fut]))[0]
        ivols = []
        prices_mc = []
        for m in moneyness:
            p, _ = price_european_call_on_futures(
                params, m, T_option, T_fut, n_paths=n_paths, seed=j
            )
            iv = implied_vol(p, F_atm, m*F_atm, T_option, 0.05)
            ivols.append(iv); prices_mc.append(p)
        axes[0].plot(moneyness, np.array(ivols)*100, "o-",
                     color=PALETTE[j], label=f"T_fut={T_fut}y")
        axes[1].plot(moneyness, prices_mc, "o-", color=PALETTE[j], label=f"T_fut={T_fut}y")

    axes[0].axvline(1.0, ls="--", color="gray", lw=1)
    axes[0].set_xlabel("Moneyness K/F"); axes[0].set_ylabel("Implied Vol (%)")
    axes[0].set_title("Implied Volatility Smile"); axes[0].legend()
    axes[1].set_xlabel("Moneyness K/F"); axes[1].set_ylabel("Option Price")
    axes[1].set_title("Option Price vs Moneyness"); axes[1].legend()

    fig.suptitle("Stage 2: Options on Futures — Surrogate Learning Target",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ── Stage 3 ───────────────────────────────────────────────────────────────

def plot_posterior_corner(
    samples: np.ndarray,
    theta_true: Optional[np.ndarray] = None,
    param_labels: Optional[list[str]] = None,
    title: str = "Schwartz-Smith Posterior",
    figsize: tuple = (12, 12),
) -> plt.Figure:
    """
    Corner plot of the MCMC posterior over Schwartz-Smith parameters.

    GW analogy: this IS your PE corner plot — identical to the ones you
    produce for EMRI parameter estimation in bilby/dynesty.
    """
    import corner
    set_style()
    if param_labels is None:
        param_labels = [r"$\kappa$", r"$\mu_\xi$", r"$\sigma_\chi$",
                        r"$\sigma_\xi$", r"$\rho$", r"$\lambda_\chi$"]
    kwargs = dict(
        labels=param_labels, show_titles=True,
        title_kwargs={"fontsize": 11},
        plot_datapoints=False, fill_contours=True,
        contourf_kwargs={"colors": [sns.color_palette("Blues")[2],
                                    sns.color_palette("Blues")[4],
                                    sns.color_palette("Blues")[5]]},
        levels=(0.68, 0.90, 0.95),
        smooth=1.2, bins=30
    )
    if theta_true is not None:
        kwargs["truths"] = theta_true
        kwargs["truth_color"] = TRUTH_GREEN

    fig = corner.corner(samples, **kwargs)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    return fig


def plot_calibration_coverage(
    coverage_results: dict,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    PP-plot: compare nominal vs achieved credible interval coverage.

    If the inference is well-calibrated, all points lie on the diagonal.
    This is the standard PP-plot used in GW PE pipelines (e.g. bilby).
    """
    set_style()
    levels   = coverage_results["levels"]
    achieved = coverage_results["achieved_coverage"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.scatter(levels, achieved, s=80, color=STAGE_COLORS[3],
               zorder=5, label="Surrogate-based calibration")
    ax.plot(levels, achieved, "-", color=STAGE_COLORS[3], alpha=0.7)

    # 1-sigma band from binomial uncertainty
    n = coverage_results.get("n_trials", 200)
    for lv, ac in zip(levels, achieved):
        se = np.sqrt(lv * (1 - lv) / n)
        ax.errorbar(lv, ac, yerr=1.96*se, fmt="none", color=STAGE_COLORS[3], capsize=4)

    ax.set_xlabel("Nominal credible level")
    ax.set_ylabel("Achieved coverage fraction")
    ax.set_title("Calibration PP-Plot\n"
                 "(GW analogy: standard injection-recovery diagnostic)")
    ax.legend()
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    return fig


def plot_posterior_vs_pointestimate(
    maturities: np.ndarray,
    F_obs: np.ndarray,
    samples: np.ndarray,
    theta_ls: np.ndarray,
    theta_true: Optional[np.ndarray] = None,
    chi0: float = 0.0,
    xi0: float = 3.5,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Compare the Bayesian posterior predictive forward curve against the
    least-squares point estimate.

    Left:  posterior predictive bands (honest uncertainty)
    Right: LS point estimate (no uncertainty — what industry currently does)

    This is the key result figure: shows why Bayesian calibration is better.
    """
    from ..models.schwartz_smith import SSParams, SchwartzSmithModel

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Draw random posterior samples for predictive
    n_draw = min(200, len(samples))
    idx    = np.random.choice(len(samples), n_draw, replace=False)
    F_post = np.array([
        SchwartzSmithModel(SSParams.from_array(samples[i], chi0, xi0))
        .forward_curve(maturities)
        for i in idx
    ])

    for ax, label, F_samples, ls_only in zip(
        axes,
        ["Bayesian Posterior Predictive", "Least-Squares Point Estimate"],
        [F_post, None],
        [False, True],
    ):
        if not ls_only:
            lo, hi = np.percentile(F_samples, [5, 95], axis=0)
            med    = np.median(F_samples, axis=0)
            ax.fill_between(maturities, lo, hi, alpha=0.25, color=GW_BLUE,
                            label="90% CI (posterior)")
            ax.plot(maturities, med, "-", color=GW_BLUE, lw=2, label="Posterior median")

        # LS estimate
        p_ls = SSParams.from_array(theta_ls, chi0, xi0)
        F_ls = SchwartzSmithModel(p_ls).forward_curve(maturities)
        ax.plot(maturities, F_ls, "--", color=SURROGATE_RED, lw=2,
                label="LS point estimate")

        # Observations
        ax.scatter(maturities, F_obs, s=50, color="black",
                   zorder=5, label="Observed F(T)")

        # True curve if known
        if theta_true is not None:
            p_true = SSParams.from_array(theta_true, chi0, xi0)
            F_true = SchwartzSmithModel(p_true).forward_curve(maturities)
            ax.plot(maturities, F_true, "g-", lw=2, label="True model", alpha=0.7)

        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Futures Price F(T)")
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.suptitle("Bayesian vs. Point-Estimate Calibration\n"
                 "(the core result: honest uncertainty vs. false precision)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_uncertainty_propagation(
    maturities: np.ndarray,
    samples: np.ndarray,
    moneyness: float = 1.0,
    T_option: float = 0.5,
    T_futures: float = 1.0,
    chi0: float = 0.0,
    xi0: float = 3.5,
    n_draw: int = 300,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Two-level uncertainty propagation:
      Level 1: p(theta | F_obs) — calibration uncertainty
      Level 2: p(V | F_obs) = integral p(V|theta) p(theta|F_obs) dtheta

    This is the key novel contribution: option prices with calibration-aware
    uncertainty. A desk sees not just a price but a distribution.

    GW analogy: propagating PE uncertainty into derived quantities
    (e.g. luminosity distance or sky area given uncertain parameters).
    """
    from ..models.schwartz_smith import SSParams, SchwartzSmithModel, price_european_call_on_futures

    set_style()
    idx    = np.random.choice(len(samples), n_draw, replace=False)
    F_post = np.array([
        SchwartzSmithModel(SSParams.from_array(samples[i], chi0, xi0))
        .forward_curve(maturities)
        for i in idx
    ])

    # Simplified option prices: use BS approximation for speed in this plot
    from scipy.stats import norm as scipy_norm
    option_prices = []
    for i in idx:
        p      = SSParams.from_array(samples[i], chi0, xi0)
        model  = SchwartzSmithModel(p)
        F_atm  = model.futures_price(np.array([T_futures]))[0]
        K      = moneyness * F_atm
        sigma  = p.sigma_chi * np.sqrt((1 - np.exp(-2*p.kappa*T_option))/(2*p.kappa)) \
                 + p.sigma_xi * np.sqrt(T_option)
        d1 = (np.log(F_atm/K) + 0.5*sigma**2*T_option) / (sigma*np.sqrt(T_option))
        d2 = d1 - sigma*np.sqrt(T_option)
        V  = np.exp(-0.05*T_option) * (F_atm * scipy_norm.cdf(d1) - K * scipy_norm.cdf(d2))
        option_prices.append(V)

    option_prices = np.array(option_prices)
    lo5, hi95 = np.percentile(F_post, [5, 95], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: forward curve uncertainty band
    ax1.fill_between(maturities, lo5, hi95, alpha=0.3, color=GW_BLUE,
                     label="90% forward curve CI")
    ax1.plot(maturities, np.median(F_post, axis=0), color=GW_BLUE, lw=2, label="Median")
    ax1.set_xlabel("Maturity (years)"); ax1.set_ylabel("Futures Price F(T)")
    ax1.set_title("Level 1: Parameter Uncertainty\np(θ | F_obs)")
    ax1.legend()

    # Right: option price distribution
    ax2.hist(option_prices, bins=40, color=STAGE_COLORS[3], alpha=0.7,
             edgecolor="white", density=True)
    ax2.axvline(np.median(option_prices), color="black", lw=2,
                label=f"Median: {np.median(option_prices):.3f}")
    lo_v, hi_v = np.percentile(option_prices, [5, 95])
    ax2.axvspan(lo_v, hi_v, alpha=0.2, color=STAGE_COLORS[3], label="90% CI")
    ax2.set_xlabel(f"Option Price (K/F={moneyness}, T_opt={T_option}y)")
    ax2.set_ylabel("Density")
    ax2.set_title("Level 2: Price Uncertainty\np(V | F_obs) = ∫ p(V|θ) p(θ|F_obs) dθ")
    ax2.legend()

    fig.suptitle("Two-Level Uncertainty Propagation\n"
                 "(novel contribution: calibration-aware option prices)",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_parameter_time_series(
    param_median_series: np.ndarray,
    param_std_series: np.ndarray,
    param_labels: Optional[list[str]] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot calibrated parameter estimates over multiple trading days.
    Shows how κ, σ_χ etc. evolve with honest uncertainty bands.

    GW analogy: equivalent to plotting how source population parameters
    evolve as you accumulate more events over the mission lifetime.
    """
    set_style()
    n_days, n_params = param_median_series.shape
    days = np.arange(n_days)

    if param_labels is None:
        param_labels = [r"$\kappa$", r"$\mu_\xi$", r"$\sigma_\chi$",
                        r"$\sigma_\xi$", r"$\rho$", r"$\lambda_\chi$"]

    n_rows = 2
    n_cols = (n_params + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes[:n_params], param_labels)):
        med = param_median_series[:, i]
        std = param_std_series[:, i]
        ax.fill_between(days, med - std, med + std,
                        alpha=0.3, color=STAGE_COLORS[3], label="±1σ posterior")
        ax.plot(days, med, color=STAGE_COLORS[3], lw=2, label="Posterior median")
        ax.set_xlabel("Trading day"); ax.set_ylabel(label)
        ax.set_title(f"Daily Calibration: {label}")
        if i == 0:
            ax.legend(fontsize=9)

    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.suptitle("Stage 3: Hierarchical Calibration — Parameter Evolution\n"
                 "(analogue of population hyperparameter constraints from EMRI catalog)",
                 fontweight="bold")
    fig.tight_layout()
    return fig
