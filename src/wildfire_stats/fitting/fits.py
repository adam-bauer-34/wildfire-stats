"""Static, time-varying, and tail-frequency fits of fire-size distributions."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import differential_evolution

from wildfire_stats.config import load_diff_evo, load_dist_bounds, load_poisson, resolve
from wildfire_stats.fitting import powerlaw as wfpl
from wildfire_stats.fitting.stats_helpers.logpdf import DIST_LOGPDFS

TAIL_SCAN_PERCENTILES = [50, 60, 70, 80, 85, 90, 92, 94, 96, 97, 98]


def fit_static_by_group(
    df,
    group_col,
    value_col="area_km2",
    xmin=None,
    R=None,
    random_state=None,
    min_n=1,
    years=None,
    year_col="year",
    verbose=True,
):
    """Static wfpl fits (bootstrap parameters + likelihood matrix) for each group.

    Args:
        df: Fire table.
        group_col: Column to group by (e.g. ``"modis_class_static"``, ``"EVT_NAME"``).
        value_col: Fire size column (km²).
        xmin, R, random_state: Passed to wfpl; default to ``min_fire``,
            ``static_R_boot`` and ``random_seed`` in the yaml.
        min_n: Skip groups with fewer than ``min_n`` rows (all sizes, not just the tail).
        years: Optional inclusive ``(start, end)`` filter on ``year_col``.

    Returns:
        ``{group: {"params", "likelihood_matrix", "best_fit"}}``.
    """
    xmin = resolve(xmin, "min_fire")
    R = resolve(R, "static_R_boot")
    random_state = resolve(random_state, "random_seed")

    if years is not None:
        df = df[(df[year_col] >= years[0]) & (df[year_col] <= years[1])]

    results = {}
    for group, subset in df.groupby(group_col):
        data = subset[value_col].values
        if len(data) < min_n:
            if verbose and len(data) > 0:
                print(f"⚠️ Skipping {group} (only {len(data)} fires)")
            continue

        if verbose:
            print(f"\n=== {group} (n={len(data)}) ===")
        params = wfpl.summarize_parameters_bootstrap(data, R=R, xmin=xmin, random_state=random_state)
        R_mat, best = wfpl.likelihood_matrix_and_best(data, xmin=xmin)

        results[group] = {
            "params": params,
            "likelihood_matrix": R_mat,
            "best_fit": best,
        }
    return results


def analyze_time_varying_mle(
    df,
    overall_results: dict,
    group_col: str = "modis_class_static",
    year_col: str = "year",
    xmin=None,
    llhr_cutoff=None,
    R_boot=None,
    min_total=None,
    verbose: bool = True,
    prior_weight=None,
    restrict_to_fitted: bool = False,
) -> dict:
    """Time-varying MLE using global Differential Evolution optimization.

    This fits linear-in-time parameterizations (on stable transforms) for each
    candidate distribution per group. For the truncated power law (TPL), it uses:
        α(t) = 1 + exp(a1 + b1 * t)
        λ(t) = exp(a2 + b2 * t)
    which guarantees α>1 and λ>0 while keeping slopes interpretable.

    A weak prior nudges parameters toward the static fit (from `overall_results`)
    to prevent drifting, especially for short series.

    Replaces ``analyze_time_varying_mle`` (MODIS classes),
    ``analyze_time_varying_mle_refined`` (call on ``build_refined_groups`` output
    with ``group_col="biome_refined"``) and notebook 7's
    ``analyze_time_varying_mle_evtname`` (``group_col="EVT_NAME"``,
    ``restrict_to_fitted=True``).

    Args:
        df: Fire records with columns `[year_col, area_km2, group_col]`.
        overall_results: Static fits from ``fit_static_by_group``.
        group_col: Grouping column.
        year_col: Name of year column. Years are centred on the mean of ``df``
            and scaled by ``trend_unit_years``.
        xmin: Minimum size threshold for tail fitting (default ``min_fire``).
        llhr_cutoff: Δlog-likelihood threshold used to filter candidate dists.
        R_boot: Bootstrap reps for SEs (default ``timevary_R_boot``).
        min_total: Minimum number of tail observations per group (default ``min_fires``).
        verbose: Print progress and filtering info.
        prior_weight: Strength of weak prior toward static parameters.
        restrict_to_fitted: Drop groups missing from ``overall_results`` *before*
            centring years (notebook 7 behaviour).

    DE bounds and parameter clips come from ``config/dist_bounds_param_clips.yaml``;
    optimizer settings from ``config/diff_evo.yaml``.

    Returns:
        Nested dict: results[group][dist_name][mode] with keys:
          - "coeffs": transformed coefficients (p1, p1', p2, p2') or subset by mode
          - "ses": bootstrap SEs for the same
          - "loglik": best value of penalized log-likelihood (negated in optimizer)
          - "n": number of tail observations used
    """
    xmin = resolve(xmin, "min_fire")
    llhr_cutoff = resolve(llhr_cutoff, "llhr_cutoff")
    R_boot = resolve(R_boot, "timevary_R_boot")
    min_total = resolve(min_total, "min_fires")
    prior_weight = resolve(prior_weight, "prior_weight")
    trend_unit_years = resolve(None, "trend_unit_years")
    seed = resolve(None, "random_seed")

    dist_cfg = load_dist_bounds()
    de_bounds = dist_cfg["de_bounds"]
    pclip = dist_cfg["param_clips"]
    lclip = dist_cfg["logpdf_clips"]
    de_cfg = load_diff_evo()

    # Suppress runtime warnings inside the optimization loop to keep logs cleaner
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    dist_logpdfs = DIST_LOGPDFS
    timevary_results: dict = {}

    df = df.copy()
    if restrict_to_fitted:
        df = df[df[group_col].isin(set(overall_results.keys()))].copy()

    # Center years by mean and scale to decades to make slopes interpretable and well-scaled
    df["year_c"] = (df[year_col] - df[year_col].mean()) / trend_unit_years

    if verbose and restrict_to_fitted:
        print("\nGroup counts for time-varying MLE (by", group_col, "):")
        print(df[group_col].value_counts())

    # Global random generator used for bootstrapping indices
    rng_global = np.random.default_rng(seed)

    for group, subset in df.groupby(group_col):
        # Extract fire sizes and enforce tail threshold xmin
        data = subset["area_km2"].values
        data = data[data >= xmin]

        if len(data) < min_total:
            if verbose:
                print(f"\n=== {group} skipped: only {len(data)} fires above xmin ({xmin}) ===")
            continue

        # Extract centered years corresponding to tail data
        years = subset.loc[subset["area_km2"] >= xmin, "year_c"].values

        if verbose:
            print(f"\n=== {group} (n={len(data)} fires ≥ {xmin}) ===")

        res = overall_results.get(group, {})
        if not res:
            if verbose:
                print(f"No overall_results found for {group}")
            continue

        params_df: pd.DataFrame = res["params"]
        llhr: pd.DataFrame = res["likelihood_matrix"]

        # Filter candidate distributions by availability, reductions, and Δloglik
        candidates: list[str] = []
        for dist, row in params_df.iterrows():
            if dist not in dist_logpdfs:
                continue
            if isinstance(row.get("reduces_to"), str):
                continue
            if dist in llhr.index and llhr.loc[dist].min() > llhr_cutoff:
                continue
            candidates.append(dist)

        if not candidates:
            if verbose:
                print(f"No viable candidates for {group}")
            continue

        group_res: dict = {}

        for dist_name in candidates:
            fit_modes: dict = {}

            # Which parameters may vary with time in each "mode"
            fit_modes_list = ("both", "p1_only", "p2_only")

            # Extract static parameter estimates to define a weak prior
            static_row = params_df.loc[dist_name]
            p1_static = float(static_row.get("p1", 1.0))
            p2_static = float(static_row.get("p2", 1.0)) if not pd.isna(static_row.get("p2", np.nan)) else 1.0

            for mode in fit_modes_list:
                # -------- Negative log-likelihood with weak prior ----------
                def neg_loglik(params: np.ndarray, data: np.ndarray = data) -> float:
                    try:
                        # unpack (a1,b1,a2,b2) by mode
                        if mode == "both":
                            a1, b1, a2, b2 = params
                        elif mode == "p1_only":
                            a1, b1, a2 = params
                            b2 = 0.0
                        elif mode == "p2_only":
                            a1, a2, b2 = params
                            b1 = 0.0

                        # time-varying parameterization (per distribution)
                        c = pclip.get(dist_name, {})
                        if dist_name == "truncated_power_law":
                            alpha = 1.0 + np.exp(a1 + b1 * years)
                            lambd = np.exp(a2 + b2 * years)
                            ll = dist_logpdfs[dist_name](data, alpha, lambd, xmin, clips=lclip[dist_name])
                        elif dist_name == "lognormal":
                            mu = a1 + b1 * years
                            sigma = np.clip(a2 + b2 * years, *c["sigma"])
                            ll = dist_logpdfs[dist_name](data, mu, sigma, xmin, clips=lclip[dist_name])
                        elif dist_name == "generalized_pareto":
                            xi = np.clip(a1 + b1 * years, *c["xi"])
                            sigma = np.clip(a2 + b2 * years, *c["sigma"])
                            ll = dist_logpdfs[dist_name](data, xi, sigma, xmin, clips=lclip[dist_name])
                        elif dist_name == "power_law":
                            alpha = np.clip(a1 + b1 * years, *c["alpha"])
                            ll = dist_logpdfs[dist_name](data, alpha, xmin, clips=lclip[dist_name])
                        elif dist_name == "weibull":
                            k = np.clip(a1 + b1 * years, *c["k"])
                            lam = np.clip(a2 + b2 * years, *c["lam"])
                            ll = dist_logpdfs[dist_name](data, k, lam, xmin, clips=lclip[dist_name])
                        elif dist_name == "stretched_exponential":
                            lam = np.clip(a1 + b1 * years, *c["lam"])
                            beta = np.clip(a2 + b2 * years, *c["beta"])
                            ll = dist_logpdfs[dist_name](data, lam, beta, xmin, clips=lclip[dist_name])
                        else:
                            return np.inf

                        # Drop any non-finite log-likelihood contributions
                        valid_ll = ll[np.isfinite(ll)]

                        # weak prior → center around static p1/p2 (on stable transforms)
                        # TODO(bug): only the TPL a1 lives on a log scale; for other
                        # families a1/a2 are raw parameters (HANDOFF 4i).
                        if dist_name == "truncated_power_law":
                            prior_center_a1 = np.log(max(p1_static - 1, 1e-3))  # α → log(α-1)
                        else:
                            prior_center_a1 = np.log(max(p1_static, 1e-6))      # p1 → log(p1)
                        prior_center_a2 = np.log(max(p2_static, 1e-6))          # p2 → log(p2)

                        prior_penalty = prior_weight * ((a1 - prior_center_a1) ** 2 + (a2 - prior_center_a2) ** 2)

                        return -np.sum(valid_ll) + prior_penalty
                    except Exception:
                        return np.inf

                # --------- Bounds by distribution and mode ----------
                # TODO(bug): power_law has only 2 bounds, so p1_only/p2_only raise
                # IndexError and "both" always returns inf (HANDOFF 4h).
                bounds = [tuple(b) for b in de_bounds.get(dist_name, de_bounds["default"])]
                if mode == "both":
                    bnds = bounds
                elif mode == "p1_only":
                    bnds = [bounds[0], bounds[1], bounds[2]]
                elif mode == "p2_only":
                    bnds = [bounds[0], bounds[2], bounds[3]]

                # -------- Differential Evolution global search ----------
                opt = differential_evolution(neg_loglik, bounds=bnds, **de_cfg["main"])
                coeffs = opt.x
                ll_max = -opt.fun

                # -------- Bootstrap via DE on resampled data ----------
                # TODO(bug): only sizes are resampled; neg_loglik still pairs them with
                # the original `years`, breaking the size↔year pairing (HANDOFF 4a).
                boot_params: list[np.ndarray] = []
                for _ in range(R_boot):
                    idx = rng_global.choice(len(data), size=len(data), replace=True)
                    boot_data = data[idx]
                    try:
                        opt_b = differential_evolution(
                            lambda p: neg_loglik(p, data=boot_data),
                            bounds=bnds,
                            **de_cfg["bootstrap"],
                        )
                        boot_params.append(opt_b.x)
                    except Exception:
                        # allow some failures (ill-conditioned resamples)
                        continue

                # -------- Transform coefficients to (p1, p1', p2, p2') --------
                if dist_name == "truncated_power_law":
                    if mode == "both":
                        a1, b1, a2, b2 = coeffs
                    elif mode == "p1_only":
                        a1, b1, a2 = coeffs
                        b2 = 0.0
                    elif mode == "p2_only":
                        a1, a2, b2 = coeffs
                        b1 = 0.0

                    # Transform back to α(t) = 1 + exp(a1 + b1*t), λ(t) = exp(a2 + b2*t)
                    p1 = 1 + np.exp(a1)
                    p1_prime = b1 * np.exp(a1)
                    p2 = np.exp(a2)
                    p2_prime = b2 * np.exp(a2)

                    if boot_params:
                        boot_p1, boot_p1p, boot_p2, boot_p2p = [], [], [], []
                        for bp in boot_params:
                            if mode == "both":
                                ba1, bb1, ba2, bb2 = bp
                            elif mode == "p1_only":
                                ba1, bb1, ba2 = bp
                                bb2 = 0.0
                            elif mode == "p2_only":
                                ba1, ba2, bb2 = bp
                                bb1 = 0.0
                            boot_p1.append(1 + np.exp(ba1))
                            boot_p1p.append(bb1 * np.exp(ba1))
                            boot_p2.append(np.exp(ba2))
                            boot_p2p.append(bb2 * np.exp(ba2))
                        # NOTE: ddof=0 here vs ddof=1 in wfpl.summarize_parameters_bootstrap
                        p1_se = np.std(boot_p1)
                        p1p_se = np.std(boot_p1p)
                        p2_se = np.std(boot_p2)
                        p2p_se = np.std(boot_p2p)
                    else:
                        p1_se = p1p_se = p2_se = p2p_se = 0.0

                    coeffs_trans = [p1, p1_prime, p2, p2_prime]
                    ses_trans = [p1_se, p1p_se, p2_se, p2p_se]
                else:
                    # For non-TPL distributions we keep coefficients on the original scale
                    ses = np.std(boot_params, axis=0) if boot_params else np.zeros_like(coeffs)
                    coeffs_trans = coeffs
                    ses_trans = ses

                fit_modes[mode] = {
                    "coeffs": coeffs_trans,
                    "ses": ses_trans,
                    "loglik": ll_max,
                    "n": len(data),
                }

            if fit_modes:
                group_res[dist_name] = fit_modes

        if group_res:
            timevary_results[group] = group_res

    return timevary_results


def fit_poisson_tail_trend(
    df,
    group_col: str = "modis_class_static",
    valid_groups=None,
    year_col: str = "year",
    area_col: str = "area_km2",
    label_col: str = "biome",
    amin_min=None,
    n_thresholds=None,
    min_years=None,
    alpha=None,
    plot: bool = True,
    title: str = "Poisson Trend in Fire Frequency vs. Size Threshold by Biome",
) -> pd.DataFrame:
    """Fit Poisson GLM of fire frequency (≥ Amin) ~ year across a log-spaced Amin grid.

    For each group, uses ``n_thresholds`` log-spaced thresholds between ``amin_min``
    and the global max observed size. Optionally plots β₁ ±1 SE against Amin.

    Replaces ``fit_poisson_tail_trend_by_biome_highres`` (pass
    ``valid_groups=POISSON_BIOMES``) and ``fit_poisson_tail_trend_by_biome_refined``
    (pass the ``build_refined_groups`` output with ``group_col="biome_refined"``,
    ``label_col="biome_refined"``).

    Args:
        df: Input data with `[group_col, year_col, area_col]`.
        valid_groups: Optional list of groups to keep.
        label_col: Name of the group column in the returned table.
        amin_min: Lowest threshold (default ``min_fire``).
        n_thresholds, min_years, alpha: Default to ``config/poisson.yaml``.
        plot: Draw the faceted β₁-vs-Amin figure.
        title: Figure title.

    Returns:
        Long DataFrame of per-group trend estimates over thresholds.
    """
    pcfg = load_poisson()
    amin_min = resolve(amin_min, "min_fire")
    n_thresholds = pcfg["n_thresholds"] if n_thresholds is None else n_thresholds
    min_years = pcfg["min_years"] if min_years is None else min_years
    alpha = pcfg["alpha"] if alpha is None else alpha

    # Suppress warnings produced by the GLM fitting process
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if valid_groups is not None:
        df = df[df[group_col].isin(valid_groups)].copy()

    # Clean up the data: keep core columns, drop missing, enforce positive areas
    df = df[[group_col, year_col, area_col]].dropna()
    df = df[df[area_col] > 0]

    # Create log-spaced thresholds between amin_min and the maximum observed area
    amax = df[area_col].max()
    thresholds = np.logspace(np.log10(amin_min), np.log10(amax), num=n_thresholds)

    groups = sorted(df[group_col].unique())
    trends_by_group: dict = {}
    skipped: list[tuple[str, int]] = []

    for group in groups:
        sub = df[df[group_col] == group]
        n_years = sub[year_col].nunique()
        if n_years < min_years:
            skipped.append((group, n_years))
            continue

        trends: list[dict] = []
        for Amin in thresholds:
            yearly_counts = (
                sub.loc[sub[area_col] >= Amin]
                .groupby(year_col)
                .size()
                .reset_index(name="count")
            )
            if yearly_counts["count"].sum() == 0:
                continue

            # Center year for numerical stability, then build design matrix
            yearly_counts["year_c"] = yearly_counts[year_col] - yearly_counts[year_col].mean()
            X = sm.add_constant(yearly_counts["year_c"])

            # Poisson GLM: log-rate as linear function of centered year
            model = sm.GLM(yearly_counts["count"], X, family=sm.families.Poisson())
            res = model.fit()

            trends.append(
                {
                    label_col: group,
                    "Amin_km2": Amin,
                    "beta1": float(res.params["year_c"]),
                    "beta1_se": float(res.bse["year_c"]),
                    "pval": float(res.pvalues["year_c"]),
                    "n_years": int(len(yearly_counts)),
                }
            )

        if not trends:
            skipped.append((group, n_years))
            continue

        trends_by_group[group] = pd.DataFrame(trends)

    if plot:
        from wildfire_stats.plotting.temporal import plot_poisson_tail_trend

        plot_poisson_tail_trend(trends_by_group, groups, amin_min=amin_min, alpha=alpha, title=title)

    all_results = list(trends_by_group.values())
    results_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=[label_col, "n_unique_years"])
        print("Skipped due to too few unique years:")
        try:
            from IPython.display import display
            display(skipped_df)
        except Exception:
            print(skipped_df.to_string(index=False))

    return results_all


def scan_powerlaw_tail(
    data_all: np.ndarray,
    label: str,
    candidate_percentiles=None,
    min_tail_n: int = 100,
    llhr_cutoff=None,
    R_boot_tail: int = 20,
    random_state=None,
):
    """Search increasing tail thresholds for a robust power-law tail.

    At each candidate percentile p, xmin = quantile(data, p); fit wfpl on the tail
    and call power law "good" if present, not reduced, and its row-min ΔLL is
    ≤ ``llhr_cutoff``. Choose the smallest xmin that passes at that and every
    higher threshold; otherwise the highest passing (or last) candidate.

    Args:
        data_all: Sorted sizes already cut at the main xmin.
        label: Group name (for messages and the returned dict).

    Returns:
        tail_info dict (biome, xmin_tail, percentile, n_tail, n_total, alpha_hat,
        row_min_llhr, robust_all_above), or None if no candidate had enough fires.
    """
    llhr_cutoff = resolve(llhr_cutoff, "llhr_cutoff")
    random_state = resolve(random_state, "random_seed")
    n_total = len(data_all)

    if candidate_percentiles is None:
        candidate_percentiles = TAIL_SCAN_PERCENTILES
    candidate_percentiles = sorted(candidate_percentiles)

    candidates: list[dict] = []

    for perc in candidate_percentiles:
        xmin_tail = float(np.quantile(data_all, perc / 100.0))
        tail = data_all[data_all >= xmin_tail]
        n_tail = len(tail)
        if n_tail < min_tail_n:
            continue

        print(
            f"  Trying tail threshold ≈ {xmin_tail:.2f} km² "
            f"({perc:.1f}th pct, n_tail={n_tail})"
        )

        try:
            params_tail = wfpl.summarize_parameters_bootstrap(
                tail, R=R_boot_tail, xmin=xmin_tail, random_state=random_state
            )
            llhr_tail, best_tail = wfpl.likelihood_matrix_and_best(
                tail, xmin=xmin_tail
            )
        except Exception as e:
            print(f"wfpl failed at this threshold: {e}")
            candidates.append(
                dict(
                    percentile=perc,
                    xmin=xmin_tail,
                    n_tail=n_tail,
                    alpha_hat=np.nan,
                    row_min_llhr=np.nan,
                    best_fit_tail=None,
                    passes=False,
                )
            )
            continue

        if "power_law" not in params_tail.index or "power_law" not in llhr_tail.index:
            print("'power_law' not present in params/llhr, skipping.")
            candidates.append(
                dict(
                    percentile=perc,
                    xmin=xmin_tail,
                    n_tail=n_tail,
                    alpha_hat=np.nan,
                    row_min_llhr=np.nan,
                    best_fit_tail=best_tail,
                    passes=False,
                )
            )
            continue

        row_pl = params_tail.loc["power_law"]
        if isinstance(row_pl.get("reduces_to"), str):
            print(f"'power_law' reduces_to={row_pl['reduces_to']}, skipping.")
            candidates.append(
                dict(
                    percentile=perc,
                    xmin=xmin_tail,
                    n_tail=n_tail,
                    alpha_hat=np.nan,
                    row_min_llhr=np.nan,
                    best_fit_tail=best_tail,
                    passes=False,
                )
            )
            continue

        row_min_llhr = float(llhr_tail.loc["power_law"].min())
        alpha_hat = float(row_pl.get("p1", np.nan))
        passes = row_min_llhr <= llhr_cutoff

        print(
            f"    power_law: alpha_hat={alpha_hat:.3f}, "
            f"row_min_llhr={row_min_llhr:.3f}, passes={passes}"
        )

        candidates.append(
            dict(
                percentile=perc,
                xmin=xmin_tail,
                n_tail=n_tail,
                alpha_hat=alpha_hat,
                row_min_llhr=row_min_llhr,
                best_fit_tail=best_tail,
                passes=passes,
            )
        )

    if not candidates:
        print(f"{label}: no candidate tail with ≥ {min_tail_n} fires.")
        return None

    # For a robust tail, we want all higher thresholds to pass
    passes_array = np.array([c["passes"] for c in candidates], dtype=bool)
    suffix_all_pass = np.zeros_like(passes_array, dtype=bool)
    running = True
    for i in range(len(candidates) - 1, -1, -1):
        running = running and passes_array[i]
        suffix_all_pass[i] = running

    robust_idx = None
    for i in range(len(candidates)):
        if passes_array[i] and suffix_all_pass[i]:
            robust_idx = i
            break

    if robust_idx is not None:
        chosen = candidates[robust_idx]
        robust_all_above = True
        print(
            f"{label}: robust tail at ≈ {chosen['xmin']:.2f} km² "
            f"({chosen['percentile']:.1f}th pct), valid for all higher thresholds."
        )
    else:
        any_pass = [i for i, c in enumerate(candidates) if c["passes"]]
        if any_pass:
            idx = any_pass[-1]
        else:
            idx = len(candidates) - 1
        chosen = candidates[idx]
        robust_all_above = False
        print(
            f"{label}: no percentile where all higher thresholds pass; "
            f"using ≈ {chosen['xmin']:.2f} km² "
            f"({chosen['percentile']:.1f}th pct)."
        )

    return dict(
        biome=label,
        xmin_tail=chosen["xmin"],
        percentile=chosen["percentile"],
        n_tail=chosen["n_tail"],
        n_total=n_total,
        alpha_hat=chosen["alpha_hat"],
        row_min_llhr=chosen["row_min_llhr"],
        robust_all_above=robust_all_above,
    )
