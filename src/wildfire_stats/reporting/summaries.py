"""Summaries of datasets and of static / time-varying fit results."""

import numpy as np
import pandas as pd

from wildfire_stats.config import resolve
from wildfire_stats.data.classification import CORE_BIOMES
from wildfire_stats.reporting.tables import format_pm

TIMEVARY_MODES = ("both", "p1_only", "p2_only")


def print_summary(name, gdf, year_col):
    """Print a summary of key fire statistics for a GeoDataFrame.

    Args:
        name (str): Descriptive name of the dataset (e.g., "MTBS" or "Idaho").
        gdf (GeoDataFrame): Dataset containing fire information, including area and year columns.
        year_col (str): Name of the column in `gdf` that stores fire year information.

    Prints:
        Summary statistics such as number of fires, year range, and basic size metrics.
    """
    print(f"\n{name} dataset:")
    print(f"  Number of fires: {len(gdf)}")
    print(f"  Year range: {gdf[year_col].min()} - {gdf[year_col].max()}")
    print(f"  Mean fire size: {gdf['area_km2'].mean():.2f} km²")
    print(f"  Median fire size: {gdf['area_km2'].median():.2f} km²")
    print(f"  Min fire size: {gdf['area_km2'].min():.2f} km²")
    print(f"  Max fire size: {gdf['area_km2'].max():.2f} km²")
    print(f"  Total burned area: {gdf['area_km2'].sum():.2f} km²")


def summarize_timevary_results_mode(
    timevary_results: dict, mode: str = "both"
) -> pd.DataFrame:
    """Convert one mode ('both', 'p1_only', or 'p2_only') from nested results into a DataFrame.

    The resulting DataFrame contains per-biome regression coefficients and
    slope significance flags for time-varying parameters.

    Args:
        timevary_results: Nested dictionary of results by biome → distribution → mode.
        mode: Which fit mode to extract ("both", "p1_only", or "p2_only").

    Returns:
        DataFrame with columns:
        biome, distribution, n, p1, p1_se, p1', p1'_se, p2, p2_se, p2', p2'_se,
        p1_slope_sig, p2_slope_sig.
    """
    z = resolve(None, "ci_z")

    # Initialize list to hold tidy records for each biome×distribution combination
    rows: list[dict] = []

    # Iterate through biomes and distributions stored in the nested results dict
    for biome, dists in timevary_results.items():
        for dist, modes in dists.items():
            # Skip distributions without the requested mode
            if mode not in modes:
                continue

            # Extract coefficient and SE arrays plus sample size
            res = modes[mode]
            coeffs = np.array(res.get("coeffs", []), dtype=float)
            ses = np.array(res.get("ses", []), dtype=float)
            n = res.get("n", np.nan)

            # Initialize parameters and significance flags with NaN/zero defaults
            p1 = p1_se = p1s = p1s_se = np.nan
            p2 = p2_se = p2s = p2s_se = np.nan
            p1_sig = p2_sig = 0

            # For each mode, interpret the shape of the coeff vector and derive significance
            # based on ±ci_z·SE confidence intervals around the slope terms.
            # interpret coefficients by mode length
            if mode == "both":
                if len(coeffs) == 2:
                    # Only p1 and its slope vary; no second parameter
                    p1, p1s = coeffs
                    p1_se, p1s_se = ses
                    ci_low, ci_high = p1s - z * p1s_se, p1s + z * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 4:
                    # Both p1 and p2 and their slopes are present
                    p1, p1s, p2, p2s = coeffs
                    p1_se, p1s_se, p2_se, p2s_se = ses
                    ci_low1, ci_high1 = p1s - z * p1s_se, p1s + z * p1s_se
                    ci_low2, ci_high2 = p2s - z * p2s_se, p2s + z * p2s_se
                    p1_sig = 1 if (ci_low1 > 0 or ci_high1 < 0) else 0
                    p2_sig = 1 if (ci_low2 > 0 or ci_high2 < 0) else 0

            elif mode == "p1_only":
                if len(coeffs) == 2:
                    # Only p1 varies over time; no p2 parameter
                    p1, p1s = coeffs
                    p1_se, p1s_se = ses
                    ci_low, ci_high = p1s - z * p1s_se, p1s + z * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 3:
                    # p1 varies with time and p2 is static
                    p1, p1s, p2 = coeffs
                    p1_se, p1s_se, p2_se = ses
                    ci_low, ci_high = p1s - z * p1s_se, p1s + z * p1s_se
                    p1_sig = 1 if (ci_low > 0 or ci_high < 0) else 0

            elif mode == "p2_only":
                if len(coeffs) == 2:
                    # p1 is static; p2 varies with time
                    p1, p2s = coeffs
                    p1_se, p2s_se = ses
                    ci_low, ci_high = p2s - z * p2s_se, p2s + z * p2s_se
                    p2_sig = 1 if (ci_low > 0 or ci_high < 0) else 0
                elif len(coeffs) == 3:
                    # p1 and p2 are both present, but only p2 has a slope
                    p1, p2, p2s = coeffs
                    p1_se, p2_se, p2s_se = ses
                    ci_low, ci_high = p2s - z * p2s_se, p2s + z * p2s_se
                    p2_sig = 1 if (ci_low > 0 or ci_high < 0) else 0

            # Append a tidy record summarizing coefficients and slope flags
            rows.append(
                {
                    "biome": biome,
                    "distribution": dist,
                    "n": n,
                    "p1": p1,
                    "p1_se": p1_se,
                    "p1'": p1s,
                    "p1'_se": p1s_se,
                    "p2": p2,
                    "p2_se": p2_se,
                    "p2'": p2s,
                    "p2'_se": p2s_se,
                    "p1_slope_sig": p1_sig,
                    "p2_slope_sig": p2_sig,
                }
            )

    # Convert accumulated records into a DataFrame and enforce column order
    df = pd.DataFrame(rows)
    df = df[
        [
            "biome",
            "distribution",
            "n",
            "p1",
            "p1_se",
            "p1'",
            "p1'_se",
            "p2",
            "p2_se",
            "p2'",
            "p2'_se",
            "p1_slope_sig",
            "p2_slope_sig",
        ]
    ]
    return df


def summarize_timevary_modes(timevary_results: dict):
    """``(df_both, df_p1, df_p2)`` from ``summarize_timevary_results_mode`` for each mode."""
    return tuple(summarize_timevary_results_mode(timevary_results, mode=m) for m in TIMEVARY_MODES)


def _delta_ll(overall_results, biome, dist, how):
    """Static Δloglik for (biome, dist): first column (``"first"``) or row minimum (``"min"``)."""
    if biome not in overall_results:
        return np.nan
    ll_df = overall_results[biome].get("likelihood_matrix")
    if not isinstance(ll_df, pd.DataFrame) or dist not in ll_df.index:
        return np.nan
    return ll_df.loc[dist].iloc[0] if how == "first" else ll_df.loc[dist].min()


def _fit_record(r_b, row_p1, row_p2, biome, dist, delta_ll):
    """One formatted summary row; slopes kept only if significant in both the joint
    and the matching single-parameter fit."""
    n = int(r_b.get("n", np.nan))
    p1, p1_se = r_b.get("p1", np.nan), r_b.get("p1_se", np.nan)
    p2, p2_se = r_b.get("p2", np.nan), r_b.get("p2_se", np.nan)

    p1_trend = p1_trend_se = p2_trend = p2_trend_se = np.nan
    if not row_p1.empty:
        if (r_b.get("p1_slope_sig") == 1) and (row_p1.iloc[0].get("p1_slope_sig") == 1):
            p1_trend = r_b["p1'"]
            p1_trend_se = r_b["p1'_se"]
    if not row_p2.empty:
        if (r_b.get("p2_slope_sig") == 1) and (row_p2.iloc[0].get("p2_slope_sig") == 1):
            p2_trend = r_b["p2'"]
            p2_trend_se = r_b["p2'_se"]

    return {
        "biome": biome,
        "distribution": dist,
        "n": n,
        "Δloglik": delta_ll,
        "p1 ± se": format_pm(p1, p1_se) if pd.notna(p1) else np.nan,
        "p2 ± se": format_pm(p2, p2_se) if pd.notna(p2) else np.nan,
        "Δp1 ± se": format_pm(p1_trend, p1_trend_se) if pd.notna(p1_trend) else "",
        "Δp2 ± se": format_pm(p2_trend, p2_trend_se) if pd.notna(p2_trend) else "",
    }


def _rows(df, biome, dist):
    return df[(df["biome"] == biome) & (df["distribution"] == dist)]


def summarize_fits(
    df_both: pd.DataFrame,
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    overall_results: dict,
    include: str = "all",
    order: str = "delta_ll",
    primary_map: dict | None = None,
) -> pd.DataFrame:
    """Merge the three time-varying modes with static Δloglik into one formatted table.

    Each row is one group × distribution: n, Δloglik, ``p1 ± se``, ``p2 ± se`` and
    ``Δp1 ± se`` / ``Δp2 ± se`` (blank unless the slope is significant in both the
    joint and the single-parameter fit).

    Args:
        df_both, df_p1, df_p2: ``summarize_timevary_modes`` output.
        overall_results: Static fits (for Δloglik).
        include: ``"all"``, ``"primary"`` (only each biome's primary distribution),
            or ``"other"`` (everything except the primaries).
        order: For ``include="all"``: ``"delta_ll"`` (every group in the tables,
            distributions sorted by static Δloglik) or ``"primary_first"`` (only the
            biomes in ``primary_map``, primary distribution first).
        primary_map: biome -> primary distribution (default ``primary_distribution``
            in the yaml). Unused for ``order="delta_ll"``.

    Replaces:
        summarize_ecoregion_fits   -> include="primary"
        summarize_other_fits       -> include="other"
        summarize_all_fits         -> include="all", order="primary_first"
        summarize_all_fits_general -> include="all", order="delta_ll"
    """
    if include not in ("all", "primary", "other"):
        raise ValueError(f"include must be 'all', 'primary', or 'other', got {include!r}")
    if order not in ("delta_ll", "primary_first"):
        raise ValueError(f"order must be 'delta_ll' or 'primary_first', got {order!r}")

    records: list[dict] = []

    if include == "all" and order == "delta_ll":
        biomes = sorted(
            set(df_both["biome"].unique())
            | set(df_p1["biome"].unique())
            | set(df_p2["biome"].unique())
        )
        for biome in biomes:
            biome_dists = sorted(
                set(df_both.loc[df_both["biome"] == biome, "distribution"].unique())
                | set(df_p1.loc[df_p1["biome"] == biome, "distribution"].unique())
                | set(df_p2.loc[df_p2["biome"] == biome, "distribution"].unique())
            )
            if not biome_dists:
                print(f"No distributions found for {biome}")
                continue

            # Order distributions from best to worst based on static Δloglik (ascending)
            delta_lls = {}
            if biome in overall_results:
                ll_df = overall_results[biome].get("likelihood_matrix")
                if isinstance(ll_df, pd.DataFrame):
                    delta_lls = {dist: ll_df.loc[dist].min() for dist in ll_df.index}
            biome_dists = sorted(biome_dists, key=lambda d: delta_lls.get(d, np.inf))

            for dist in biome_dists:
                row_b = _rows(df_both, biome, dist)
                if row_b.empty:
                    continue
                records.append(_fit_record(
                    row_b.iloc[0], _rows(df_p1, biome, dist), _rows(df_p2, biome, dist),
                    biome, dist, _delta_ll(overall_results, biome, dist, "min"),
                ))

        df_out = pd.DataFrame(records)
        if not df_out.empty:
            df_out.sort_values(["biome", "Δloglik"], inplace=True, na_position="last")
        return df_out

    primary_map = resolve(primary_map, "primary_distribution")

    if include == "primary":
        for biome, best_dist in primary_map.items():
            row_both = _rows(df_both, biome, best_dist)
            if row_both.empty:
                print(f"No match for {biome} ({best_dist}) in df_both.")
                continue
            records.append(_fit_record(
                row_both.iloc[0], _rows(df_p1, biome, best_dist), _rows(df_p2, biome, best_dist),
                biome, best_dist, _delta_ll(overall_results, biome, best_dist, "first"),
            ))

    elif include == "other":
        for _, r_b in df_both.iterrows():
            biome = r_b["biome"]
            dist = r_b["distribution"]
            if biome in primary_map and dist == primary_map[biome]:
                continue  # skip primaries
            records.append(_fit_record(
                r_b, _rows(df_p1, biome, dist), _rows(df_p2, biome, dist),
                biome, dist, _delta_ll(overall_results, biome, dist, "first"),
            ))

    else:  # include == "all", order == "primary_first"
        for biome, primary_dist in primary_map.items():
            biome_rows = df_both[df_both["biome"] == biome]
            if biome_rows.empty:
                print(f"No entries for {biome}")
                continue
            dists_in_biome = [primary_dist] + [
                d for d in biome_rows["distribution"].unique() if d != primary_dist
            ]
            for dist in dists_in_biome:
                row_both = biome_rows[biome_rows["distribution"] == dist]
                if row_both.empty:
                    continue
                records.append(_fit_record(
                    row_both.iloc[0], _rows(df_p1, biome, dist), _rows(df_p2, biome, dist),
                    biome, dist, _delta_ll(overall_results, biome, dist, "first"),
                ))

    return pd.DataFrame(records)


def fires_time_series_by_ecoregion(
    df,
    year_col: str = "year",
    class_col: str = "modis_class_static",
    period_years: int = 5,
    valid_classes: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    include_total: bool = False,
    plot: bool = True,
    figsize: tuple[int, int] = (10, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate fire counts per ecoregion over fixed-length periods (drops final partial bin).

    Args:
        df: Input with `year_col` and `class_col`.
        year_col: Column with integer years.
        class_col: Column with class labels.
        period_years: Width of each time bin.
        valid_classes: Optional list of classes to include (default ``CORE_BIOMES``).
        start_year: Optional start bound for years.
        end_year: Optional end bound for years.
        include_total: Add a "Total" series across classes.
        plot: If True, draw a multi-series line chart
            (``plotting.temporal.plot_fire_counts_by_period``).
        figsize: Figure size.

    Returns:
        (counts_long, counts_wide)
            counts_long: tidy with ["period_start", "period_end", "period_label", class_col, "count"]
            counts_wide: wide matrix (#fires per period × class)
    """
    if valid_classes is None:
        valid_classes = CORE_BIOMES

    data = df.copy()

    # Coerce year column to numeric and drop rows with invalid years
    data = data[pd.to_numeric(data[year_col], errors="coerce").notna()]
    data[year_col] = data[year_col].astype(int)

    # Drop rows with missing class labels and restrict to valid_classes
    data = data[data[class_col].notna()]
    data = data[data[class_col].isin(valid_classes)]

    if data.empty:
        raise ValueError("No valid rows after filtering by years and classes.")

    ymin = data[year_col].min() if start_year is None else int(start_year)
    ymax = data[year_col].max() if end_year is None else int(end_year)

    # Drop incomplete final bin to ensure all bins have equal length
    max_complete_year = (ymax // period_years) * period_years + (period_years - 1)
    if max_complete_year > ymax:
        max_complete_year -= period_years
    data = data[(data[year_col] >= ymin) & (data[year_col] <= max_complete_year)]

    def period_start(y: int) -> int:
        offset = (y - ymin) // period_years
        return ymin + offset * period_years

    data["_period_start"] = data[year_col].apply(period_start)
    data["_period_end"] = data["_period_start"] + period_years - 1
    data["_period_label"] = data["_period_start"].astype(str) + "–" + data["_period_end"].astype(str)

    counts = (
        data.groupby(["_period_start", "_period_end", "_period_label", class_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["_period_start", class_col])
    )

    counts_wide = counts.pivot_table(
        index="_period_label", columns=class_col, values="count", aggfunc="sum", fill_value=0
    ).reindex(columns=sorted(valid_classes), fill_value=0)

    if include_total:
        counts_wide["Total"] = counts_wide.sum(axis=1)

    counts_long = counts.rename(
        columns={"_period_start": "period_start", "_period_end": "period_end", "_period_label": "period_label"}
    ).reset_index(drop=True)

    if plot:
        from wildfire_stats.plotting.temporal import plot_fire_counts_by_period

        plot_fire_counts_by_period(counts_wide, period_years, figsize=figsize)

    return counts_long, counts_wide


def classification_agreement(df, col_a="modis_class_timevary", col_b="modis_class_static",
                             years=None, year_col="year", exclude="Unknown"):
    """Share of fires where two classifications agree (notebook 5: time-varying vs static MODIS).

    Uses fires within ``years`` (default ``modis_years``) whose ``col_a`` is not ``exclude``.

    Returns:
        (agreement_rate, comparison_df)
    """
    years = resolve(years, "modis_years")
    overlap_mask = (df[year_col].between(years[0], years[1])) & (df[col_a] != exclude)

    comparison_df = df.loc[overlap_mask].copy()
    comparison_df["match"] = comparison_df[col_a] == comparison_df[col_b]

    agreement_rate = comparison_df["match"].mean()
    print(f"Agreement between {col_a} and {col_b} ({years[0]}–{years[1]}): {agreement_rate:.2%}")
    return agreement_rate, comparison_df
