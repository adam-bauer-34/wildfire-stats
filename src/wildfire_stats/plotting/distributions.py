"""Fitted-distribution figures: parameter space, CCDF/CDF overlays, and Q–Q plots."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from wildfire_stats.config import load_paths, resolve
from wildfire_stats.fitting import powerlaw as wfpl
from wildfire_stats.fitting.stats_helpers.ccdf import ccdf_power_law
from wildfire_stats.fitting.stats_helpers.qq import qq_theoretical, quantile_for_dist
from wildfire_stats.reporting.tables import parse_val_pm_se

FIG_DPI = 600


def plot_distribution_params(results_dicts, counts_dicts, 
                             min_count=None, llhr_cutoff=None, 
                             max_rel_error=None,
                             log_axes=False,
                             datasets=("GFA", "Idaho", "MTBS")):
    """Visualize fitted distribution parameters across datasets and biomes.

    Each subplot corresponds to one distribution. For 1-parameter distributions,
    error bars represent uncertainty. For 2-parameter distributions, ellipses
    show joint uncertainty regions.

    Args:
        results_dicts (dict): Mapping of dataset name → biome → fit results (params, likelihoods).
        counts_dicts (dict): Mapping of dataset name → biome → sample counts.
        min_count (int, optional): Minimum sample count required to include a biome. Defaults to ``min_fires``.
        llhr_cutoff (float, optional): Log-likelihood ratio threshold to exclude poor fits. Defaults to ``llhr_cutoff``.
        max_rel_error (float, optional): Maximum allowed relative error (SE/param). Defaults to ``max_rel_error``.
        log_axes (bool, optional): Whether to log-scale axes. Defaults to False.
        datasets (tuple[str], optional): Dataset names for legend markers. Defaults to ("GFA", "Idaho", "MTBS").

    Displays:
        A faceted matplotlib figure comparing distribution parameter estimates across datasets.
    """
    max_rel_error = resolve(max_rel_error, "max_rel_error")
    llhr_cutoff = resolve(llhr_cutoff, "llhr_cutoff")
    min_count = resolve(min_count, "min_fires")
    markers = {"GFA": "o", "Idaho": "s", "MTBS": "^"}
    colors = plt.cm.tab10.colors
    
    # Collect all unique distribution names
    all_dists = set()
    for dset, res in results_dicts.items():
        for biome, out in res.items():
            all_dists.update(out["params"].index)
    all_dists = sorted(all_dists)

    ncols, nrows = 3, int(np.ceil(len(all_dists)/3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    biome_to_color = {}
    panels_with_points = set()

    # Iterate through datasets and biomes, plotting parameter points or ellipses
    for dset, res in results_dicts.items():
        for biome, out in res.items():
            if biome == "Unknown" or counts_dicts[dset].get(biome, 0) < min_count:
                continue

            likelihoods = out["likelihood_matrix"]

            for dist_name, row in out["params"].iterrows():
                if isinstance(row.get("reduces_to"), str):
                    continue
                if dist_name in likelihoods.index:
                    if likelihoods.loc[dist_name].min() > llhr_cutoff:
                        continue

                # Assign consistent color to biome
                if biome not in biome_to_color:
                    biome_to_color[biome] = colors[len(biome_to_color) % len(colors)]
                ax = axes[all_dists.index(dist_name)]

                p1, p1_se = row.get("p1", np.nan), row.get("p1_se", np.nan)
                p2, p2_se = row.get("p2", np.nan), row.get("p2_se", np.nan)
                if np.isnan(p1):
                    continue

                # Skip fits with excessive relative error
                if not np.isnan(p1_se) and abs(p1) > 0 and p1_se/abs(p1) > max_rel_error:
                    continue
                if not np.isnan(p2) and not np.isnan(p2_se) and abs(p2) > 0 and p2_se/abs(p2) > max_rel_error:
                    continue

                # Validate parameter domains per distribution family
                valid = True
                min_p1, min_p2 = -np.inf, -np.inf
                if dist_name in ("weibull", "weibull_excess", "stretched_exponential"):
                    if p1 <= 0 or (not np.isnan(p2) and p2 <= 0): valid = False
                    min_p1, min_p2 = 0, 0
                elif dist_name in ("lognormal", "lognormal_excess"):
                    if not np.isnan(p2) and p2 <= 0: valid = False
                    min_p2 = 0
                elif dist_name == "generalized_pareto":
                    if p2 <= 0 or p1 < 0: valid = False
                    min_p1, min_p2 = 0, 0
                elif dist_name == "truncated_power_law":
                    if p1 <= 1 or p2 <= 0: valid = False
                    min_p1, min_p2 = 1, 0
                elif dist_name == "exponential":
                    if p1 <= 0: valid = False
                    min_p1 = 0
                elif dist_name == "power_law":
                    if p1 <= 1: valid = False
                    min_p1 = 1

                if not valid:
                    continue

                # Plot parameter confidence regions
                if dist_name in ("exponential", "power_law"):
                    # One-parameter: horizontal error bar
                    p1_low, p1_high = max(p1 - p1_se, min_p1 + 1e-8), p1 + p1_se
                    ax.errorbar(
                        p1, 0,
                        xerr=[[p1 - p1_low], [p1_high - p1]],
                        fmt=markers.get(dset, "o"),
                        color=biome_to_color[biome]
                    )
                else:
                    # Two-parameter: ellipse patch
                    theta = np.linspace(0, 2*np.pi, 200)
                    x = p1 + p1_se * np.cos(theta)
                    y = p2 + p2_se * np.sin(theta) if not np.isnan(p2) else np.zeros_like(theta)

                    # Clip to valid parameter domain
                    x = np.clip(x, min_p1, None)
                    y = np.clip(y, min_p2, None)

                    polygon = Polygon(np.column_stack([x, y]),
                                      closed=True,
                                      facecolor=biome_to_color[biome],
                                      edgecolor="none",
                                      alpha=0.3)
                    ax.add_patch(polygon)

                    ax.plot(p1, p2 if not np.isnan(p2) else 0,
                            markers.get(dset, "o"),
                            color=biome_to_color[biome], markersize=4)

                panels_with_points.add(dist_name)

    # Configure subplot titles and scales
    for i, dist_name in enumerate(all_dists):
        ax = axes[i]
        if dist_name in panels_with_points:
            ax.set_title(dist_name)
            ax.set_xlabel("Parameter 1")
            ax.set_ylabel("Parameter 2")
            if log_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")
        else:
            fig.delaxes(ax)

    # Legends for biome colors and dataset markers
    biome_handles = [
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=color, markersize=10, label=biome)
        for biome, color in biome_to_color.items()
    ]
    dataset_handles = [
        plt.Line2D([0],[0], marker=markers[dset], color="k",
                   linestyle="None", markersize=8, label=dset)
        for dset in datasets
    ]

    fig.legend(handles=biome_handles, title="Biomes", loc="upper right", bbox_to_anchor=(1.15, 1.0))
    fig.legend(handles=dataset_handles, title="Datasets", loc="lower right", bbox_to_anchor=(1.15, 0.0))

    plt.tight_layout()
    plt.show()


def plot_modis_category_ccdf(df, modis_class, xmin=None, which=("power_law",), col_name="modis_class"):
    """Plot the complementary CDF (CCDF) of fire sizes for one land-cover class.

    Args:
        df (pd.DataFrame): Classified fires with ``col_name`` and 'area_km2'.
        modis_class (str): Class name to filter (e.g., 'Croplands').
        xmin (float, optional): Minimum cutoff for fitting (default ``min_fire``).
        which (tuple[str], optional): Distributions to overlay on the CCDF plot.
        col_name (str): Class column.

    Returns:
        matplotlib.axes.Axes | None: The plot axis if successful, else None if data missing.
    """
    xmin = resolve(xmin, "min_fire")
    subset = df[df[col_name] == modis_class]

    if subset.empty:
        print(f"No fires found for MODIS class: {modis_class}")
        return None

    data = subset["area_km2"].dropna().to_numpy()

    if len(data) == 0:
        print(f"No valid area data for MODIS class: {modis_class}")
        return None

    # Plot CCDF with overlayed fits from wildfire_powerlaw
    ax = wfpl.plot_ccdf_with_selected_fits(data, xmin=xmin, which=which)
    ax.set_title(f"CCDF of fire size in {modis_class}")
    plt.show()
    return ax


# ---------------------------------------------------------------- panel helpers

def _ymin_for(data):
    """Lower CCDF y-limit from the empirical plotting positions (floored at 1e-5)."""
    n = len(data)
    empirical_ccdf = 1 - np.arange(1, n + 1) / (n + 1)
    return max(np.min(empirical_ccdf), 1e-5)


def _ccdf_panel(ax, data, xmin, which, label, ymin, title_suffix=""):
    """CCDF with selected wfpl fits overlaid."""
    ax = wfpl.plot_ccdf_with_selected_fits(data, xmin=xmin, which=which, ax=ax)
    ax.set_ylim(ymin, 1)
    ax.set_title(f"{label}: Fire Size CCDF{title_suffix}")
    ax.set_xlabel("Fire size (km²)")
    ax.set_ylabel("CCDF")
    return ax


def _cdf_panel(ax, data, xmin, which, label, extra=None):
    """Empirical CDF plus fitted wfpl CDFs in log-x space; ``extra(ax)`` adds overlays."""
    try:
        fit = wfpl.Fit(data, xmin=xmin, discrete=False)
        fit.plot_cdf(color="k", linewidth=2, ax=ax, label="Empirical data")
        for name in which:
            try:
                dist = getattr(fit, name)
                dist.plot_cdf(
                    ax=ax,
                    linestyle="--",
                    label=f"{name.replace('_',' ').title()} fit",
                )
            except Exception as e:
                print(f"Skipped {name} in CDF: {e}")
        if extra is not None:
            extra(ax)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Fire size (km²)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{label}: Fire Size CDF")
        ax.legend(fontsize=8)
    except Exception as e:
        print(f"Could not plot CDFs: {e}")


def _qq_scatter(ax, theoretical_q, empirical_q, fit_label, title, line=None,
                color="tab:blue", xlabel="Theoretical Quantiles"):
    """Q–Q scatter with a 1:1 reference line on log–log axes."""
    ax.scatter(theoretical_q, empirical_q, s=15, alpha=0.6, color=color, label=fit_label)
    lo, hi = (empirical_q.min(), empirical_q.max()) if line is None else line
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1 line")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _stationary_qq_panel(ax, data, xmin, dist_name, title, err_msg, model=None, fit_label=None):
    """Q–Q of sorted ``data`` against a wfpl fit (``which[0]`` unless ``model`` given)."""
    try:
        if model is None:
            fit = wfpl.Fit(data, xmin=xmin, discrete=False)
            model = getattr(fit, dist_name)
        empirical_q = np.sort(data)
        probs = (np.arange(1, len(empirical_q) + 1) - 0.5) / len(empirical_q)
        theoretical_q = qq_theoretical(model, probs, data.min(), data.max())
        _qq_scatter(ax, theoretical_q, empirical_q,
                    fit_label if fit_label is not None else f"{dist_name.title()} fit", title)
    except Exception as e:
        print(f"{err_msg}: {e}")


def _save_fig(save_dir, fname):
    plt.tight_layout()
    save_dir = os.fspath(load_paths().figs if save_dir is None else save_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", transparent=True)
    print(f"Figure saved to {save_path}")
    plt.show()


def _slug(label):
    return label.replace(" ", "_").lower()


# ---------------------------------------------------------------- figures

def plot_group_ccdf_cdf_qq(
    df,
    group,
    col_name="biome_refined",
    xmin=None,
    which=("power_law",),
    save_dir=None,
    nonstationary=True,
    year_col="year",
    timevary_summary=None,
    timevary_dist=None,
    trend_unit_years=None,
):
    """Diagnostic CCDF / CDF / Q–Q panels for one group of fires.

    Panels:
        1) CCDF with selected fits (via `wfpl`)
        2) CDF with selected fits (via `wfpl`)
        3) Stationary Q–Q plot vs ``which[0]``
        4) (``nonstationary=True``) Q–Q vs the time-varying distribution from
           ``timevary_summary`` (output of ``summarize_fits``)

    Replaces ``plot_modis_category_ccdf_cdf_qq`` (``col_name="modis_class_static"``,
    ``nonstationary=False``) and ``plot_refined_biome_ccdf_cdf_qq`` (also notebook
    7's inline copy).

    Nonstationary QQ logic:
      - Parse p1, p2, Δp1, Δp2 from 'p1 ± se', 'p2 ± se', 'Δp1 ± se', 'Δp2 ± se'
        (blank Δ -> 0).
      - t_scaled = (year - mean(year in this group)) / trend_unit_years;
        p1(t_i) = p1 + Δp1 * t_scaled_i, p2(t_i) = p2 + Δp2 * t_scaled_i.
      - For sorted sizes x_i with p_i = (rank_i - 0.5) / n, plot
        q_i = F^{-1}(p_i; p1(t_i), p2(t_i), xmin) against x_i.

    Supported distributions for the nonstationary QQ:
        'lognormal', 'generalized_pareto', 'power_law',
        'stretched_exponential', 'weibull', 'exponential'.

    Returns:
        Matplotlib Figure, or None if skipped. Saved to ``save_dir``
        (default ``paths.figs``).
    """
    xmin = resolve(xmin, "min_fire")
    trend_unit_years = resolve(trend_unit_years, "trend_unit_years")

    subset = df[df[col_name] == group]
    if subset.empty:
        print(f"No fires found for {col_name}: {group}")
        return None

    years = None
    if nonstationary:
        # Extract years if present; this is required for nonstationary QQ
        if year_col not in subset.columns:
            print(f"year_col='{year_col}' not in df; nonstationary QQ will be skipped.")
        else:
            years = subset[year_col].to_numpy()

    # Extract fire size data and enforce alignment with year values
    # TODO(bug): dropna() on sizes but not years, and the xmin mask below is built
    # from the already-filtered `data`, so years misalign (IndexError) whenever a
    # fire is NaN or below xmin.
    data = subset["area_km2"].dropna().to_numpy()
    if years is not None:
        mask_valid = np.isfinite(data) & np.isfinite(years)
        data = data[mask_valid]
        years = years[mask_valid]

    data = data[data >= xmin]
    if years is not None:
        years = years[data >= xmin]

    if len(data) == 0:
        print(f"No valid area data above xmin={xmin} for {group}")
        return None

    # Sort data (and years, if present) so Q–Q plots are well-defined
    idx = np.argsort(data)
    data = data[idx]
    if years is not None:
        years = years[idx]

    ymin = _ymin_for(data)

    if nonstationary:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        ax_ccdf, ax_cdf, ax_qq_stat, ax_qq_nonstat = axes
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax_ccdf, ax_cdf, ax_qq_stat = axes

    _ccdf_panel(ax_ccdf, data, xmin, which, group, ymin)
    _cdf_panel(ax_cdf, data, xmin, which, group)
    if nonstationary:
        _stationary_qq_panel(ax_qq_stat, data, xmin, which[0], f"{group}: Q–Q Plot (stationary)",
                             "Could not generate stationary Q–Q plot")
    else:
        _stationary_qq_panel(ax_qq_stat, data, xmin, which[0], f"{group}: Q–Q Plot",
                             "Could not generate Q–Q plot")

    if nonstationary:
        try:
            if timevary_summary is None:
                raise ValueError("timevary_summary is None; skipping nonstationary QQ.")
            if years is None:
                raise ValueError("year_col not available; skipping nonstationary QQ.")

            dist_tv = timevary_dist if timevary_dist is not None else which[0]

            mask = (
                (timevary_summary["biome"] == group)
                & (timevary_summary["distribution"] == dist_tv)
            )
            if not mask.any():
                raise ValueError(
                    f"No row in timevary_summary for biome='{group}', "
                    f"distribution='{dist_tv}'."
                )
            row = timevary_summary[mask].iloc[0]

            # Parse p1, p2, and their slopes (Δp1, Δp2) from formatted columns
            p1 = parse_val_pm_se(row.get("p1 ± se", np.nan))
            p2 = parse_val_pm_se(row.get("p2 ± se", np.nan))
            dp1 = parse_val_pm_se(row.get("Δp1 ± se", np.nan))
            dp2 = parse_val_pm_se(row.get("Δp2 ± se", np.nan))

            if not np.isfinite(p1):
                raise ValueError("Non-finite p1 in timevary_summary.")
            if not np.isfinite(p2):
                p2 = 0.0
            if not np.isfinite(dp1):
                dp1 = 0.0
            if not np.isfinite(dp2):
                dp2 = 0.0

            empirical_q_ns = data.copy()
            n_ns = len(empirical_q_ns)
            probs_ns = (np.arange(1, n_ns + 1) - 0.5) / n_ns

            # TODO: recentres on this group's tail years, not the MLE's centring year.
            year_center = float(np.nanmean(years))
            t_scaled = (years - year_center) / trend_unit_years

            p1_t = p1 + dp1 * t_scaled
            p2_t = p2 + dp2 * t_scaled

            theoretical_q_ns = np.empty_like(empirical_q_ns, dtype=float)
            for i in range(n_ns):
                theoretical_q_ns[i] = quantile_for_dist(
                    probs_ns[i], dist_tv, p1_t[i], p2_t[i], xmin
                )

            qmin = min(empirical_q_ns.min(), theoretical_q_ns.min())
            qmax = max(empirical_q_ns.max(), theoretical_q_ns.max())
            _qq_scatter(
                ax_qq_nonstat, theoretical_q_ns, empirical_q_ns,
                f"Nonstationary {dist_tv}", f"{group}: Q–Q Plot (nonstationary {dist_tv})",
                line=(qmin, qmax), color="tab:purple",
                xlabel="Theoretical Quantiles (time-varying)",
            )
        except Exception as e:
            print(f"Could not generate nonstationary Q–Q plot: {e}")

    suffix = "_cdf_ccdf_qq_nonstationary.png" if nonstationary else "_cdf_ccdf_qq.png"
    _save_fig(save_dir, f"{_slug(group)}{suffix}")
    return fig


def plot_tail_scan_ccdf_cdf_qq(
    df,
    group,
    col_name="biome_refined",
    xmin0=None,
    which=("power_law",),
    candidate_percentiles=None,
    min_tail_n: int = 100,
    llhr_cutoff=None,
    R_boot_tail: int = 20,
    save_dir=None,
):
    """Find a robust power-law tail (``fitting.fits.scan_powerlaw_tail``) and plot it.

    Panels: (1) CCDF with selected fits + tail power-law overlay, (2) CDF with
    selected fits + tail overlay, (3) Q–Q of the tail data vs the tail power law.

    Was ``plot_refined_biome_tail_scan_ccdf_cdf_qq``.

    Returns:
        (fig, tail_info); (None, None) on failure.
    """
    from wildfire_stats.fitting.fits import scan_powerlaw_tail

    xmin0 = resolve(xmin0, "min_fire")

    subset = df[df[col_name] == group]
    if subset.empty:
        print(f"No fires found for refined biome: {group}")
        return None, None

    data_all = subset["area_km2"].dropna().to_numpy()
    data_all = data_all[data_all >= xmin0]
    if len(data_all) == 0:
        print(f"No valid area data above xmin0={xmin0} for {group}")
        return None, None

    data_all = np.sort(data_all)
    n_total = len(data_all)

    print(f"\n=== {group} — total fires ≥ {xmin0} km²: n={n_total} ===")
    tail_info = scan_powerlaw_tail(
        data_all, group,
        candidate_percentiles=candidate_percentiles,
        min_tail_n=min_tail_n,
        llhr_cutoff=llhr_cutoff,
        R_boot_tail=R_boot_tail,
    )
    if tail_info is None:
        return None, None

    xmin_tail = tail_info["xmin_tail"]
    perc_tail = tail_info["percentile"]
    n_tail = tail_info["n_tail"]
    alpha_hat = tail_info["alpha_hat"]
    row_min_llhr = tail_info["row_min_llhr"]
    robust_all_above = tail_info["robust_all_above"]

    data = data_all
    ymin = _ymin_for(data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_ccdf, ax_cdf, ax_qq = axes

    # (1) CCDF with wfpl fits plus tail power-law overlay rescaled by tail mass
    ax_ccdf = _ccdf_panel(
        ax_ccdf, data, xmin0, which, group, ymin,
        title_suffix=(
            f"\nTail xmin={xmin_tail:.1f} km² ({perc_tail:.1f}th pct, n_tail={n_tail}, "
            f"row_min_llhr={row_min_llhr:.2f}, robust={'yes' if robust_all_above else 'no'})"
        ),
    )
    if np.isfinite(alpha_hat):
        mass_tail = n_tail / n_total
        xgrid = np.logspace(np.log10(data.min()), np.log10(data.max()), 500)
        mask = xgrid >= xmin_tail
        if np.any(mask):
            y_tail = mass_tail * ccdf_power_law(xgrid[mask], alpha_hat, xmin_tail)
            ax_ccdf.plot(
                xgrid[mask],
                y_tail,
                linestyle="--",
                color="tab:green",
                linewidth=2,
                label="Tail power law",
            )
            ax_ccdf.legend(fontsize=8)

    # (2) CDF with the tail power-law CDF, accounting for mass below xmin_tail
    def tail_cdf_overlay(ax):
        if np.isfinite(alpha_hat):
            xgrid = np.logspace(np.log10(data.min()), np.log10(data.max()), 500)
            mask = xgrid >= xmin_tail
            if np.any(mask):
                ccdf_tail = ccdf_power_law(xgrid[mask], alpha_hat, xmin_tail)
                mass_tail = n_tail / n_total
                p_below = 1.0 - mass_tail
                tail_cdf = np.empty_like(xgrid[mask])
                tail_cdf[:] = p_below + mass_tail * (1.0 - ccdf_tail)
                ax.plot(
                    xgrid[mask],
                    tail_cdf,
                    linestyle="--",
                    color="tab:green",
                    linewidth=2,
                    label="Tail power law",
                )

    _cdf_panel(ax_cdf, data, xmin0, which, group, extra=tail_cdf_overlay)

    # (3) Q–Q: only tail points versus the tail power-law fit
    try:
        tail = data[data >= xmin_tail]
        if len(tail) == 0:
            raise ValueError("No data in tail for QQ plot.")
        tail = np.sort(tail)
        model = wfpl.Fit(tail, xmin=xmin_tail, discrete=False).power_law
    except Exception as e:
        print(f"Could not generate Q–Q plot: {e}")
    else:
        _stationary_qq_panel(ax_qq, tail, xmin_tail, "power_law", f"{group}: Q–Q Plot (tail only)",
                             "Could not generate Q–Q plot", model=model, fit_label="Power law (tail) fit")

    _save_fig(save_dir, f"{_slug(group)}_tail_scan_cdf_ccdf_qq.png")
    return fig, tail_info
