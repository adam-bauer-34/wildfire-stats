"""Time-series and time-evolution figures."""

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
from matplotlib import cm

from wildfire_stats.config import resolve
from wildfire_stats.fitting.stats_helpers.ccdf import ccdf_dispatch


def plot_fire_counts_faceted(
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame,
    year_col: str = "year",
    static_col: str = "modis_class_static",
    ncols: int = 3,
    panel_width: float = 6,
    panel_height: float = 4,
) -> None:
    """Faceted barplots of fire counts per year by MODIS class + overall.

    Args:
        mtbs_classified: DataFrame/GeoDataFrame with `[year_col, static_col]`.
        year_col: Column containing the year.
        static_col: Column with static MODIS class labels.
        ncols: Number of columns in the facet grid.
        panel_width: Width of each panel (inches).
        panel_height: Height of each panel (inches).
    """
    # Aggregate total yearly fire counts across all MODIS classes
    yearly_counts = mtbs_classified.groupby(year_col).size()

    # Aggregate yearly counts by MODIS class and pivot to wide format
    category_counts = mtbs_classified.groupby([year_col, static_col]).size().unstack(fill_value=0)

    # Determine categories and layout of facet grid
    categories = category_counts.columns.tolist()
    n_categories = len(categories)
    n_plots = n_categories + 1  # one panel per category + overall panel
    nrows = int(np.ceil(n_plots / ncols))

    # Set up figure with shared axes across facets
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    # per-category panels
    # For each MODIS class, draw a bar plot of yearly fire counts
    for i, cat in enumerate(categories):
        ax = axes[i]
        category_counts[cat].plot(kind="bar", ax=ax, color="tab:blue", alpha=0.7)
        ax.set_title(cat)
        ax.set_xlabel("")
        ax.set_ylabel("Fires")

    # overall panel
    # In the next panel, plot the total yearly fire count across all classes
    ax = axes[n_categories]
    yearly_counts.plot(kind="bar", ax=ax, color="black", alpha=0.8)
    ax.set_title("Overall Fires")
    ax.set_xlabel("")
    ax.set_ylabel("Fires")

    # remove any extra subplot axes
    # If grid has more panels than needed, remove the unused axes
    for j in range(n_categories + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a common title and tidy layout
    fig.suptitle("MTBS Fires by Year and Static MODIS Classification", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


def fire_threshold_analysis(
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame,
    year_col: str = "year",
    area_col: str = "area_km2",
    thresholds: tuple[float, ...] = (0, 4, 10, 20, 50),
) -> None:
    """Plot yearly fire counts at multiple minimum-size thresholds.

    Args:
        mtbs_classified: Input with `[year_col, area_col]`.
        year_col: Name of year column.
        area_col: Name of size column in km².
        thresholds: List of minimum size thresholds to consider.
    """
    # For each minimum threshold, compute yearly fire counts of fires above that size
    yearly_counts_by_thresh: dict[float, pd.Series] = {}
    for t in thresholds:
        filtered = mtbs_classified[mtbs_classified[area_col] >= t]
        yearly_counts = filtered.groupby(year_col).size()
        yearly_counts_by_thresh[t] = yearly_counts

    # Combine threshold-specific time series into a single DataFrame
    counts_df = pd.DataFrame(yearly_counts_by_thresh).fillna(0).astype(int)

    # Plot multiseries line chart comparing thresholds
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in thresholds:
        ax.plot(counts_df.index, counts_df[t], marker="o", label=f"≥ {t} km²")

    # Add informative labels and legend
    ax.set_title("Yearly Fire Counts by Threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Fires")
    ax.legend(title="Min fire size")
    plt.tight_layout()
    plt.show()


def plot_fire_counts_by_period(counts_wide: pd.DataFrame, period_years: int, figsize=(10, 5)) -> None:
    """Multi-series line chart of fires per period (from ``fires_time_series_by_ecoregion``)."""
    plt.figure(figsize=figsize)
    for cls in counts_wide.columns:
        plt.plot(counts_wide.index, counts_wide[cls], marker="o", label=cls)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"Number of fires per {period_years}-year period")
    plt.title(f"MTBS Fires per Land Cover Type — {period_years}-year periods" if period_years else
              "MTBS Fires per Land Cover Type")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_poisson_tail_trend(trends_by_group: dict, groups: list, amin_min=None, alpha: float = 0.05,
                            title: str = "Poisson Trend in Fire Frequency vs. Size Threshold by Biome") -> None:
    """Facet per group of Poisson β₁ ±1 SE against Amin (from ``fit_poisson_tail_trend``).

    ``groups`` fixes the panel order; groups missing from ``trends_by_group``
    (skipped fits) leave an empty panel.
    """
    amin_min = resolve(amin_min, "min_fire")
    ncols = 3
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    for i, group in enumerate(groups):
        if group not in trends_by_group:
            continue
        df_trend = trends_by_group[group]

        # Plot β1 vs Amin for this group, with error bars and significant points marked
        ax = axes[i]
        sig = df_trend["pval"] < alpha

        ax.errorbar(
            df_trend["Amin_km2"],
            df_trend["beta1"],
            yerr=df_trend["beta1_se"],
            fmt="o-",
            capsize=3,
            color="C0",
            alpha=0.8,
        )
        ax.scatter(
            df_trend.loc[sig, "Amin_km2"],
            df_trend.loc[sig, "beta1"],
            color="red",
            s=25,
            label=f"p<{alpha:g}",
        )
        ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_xscale("log")

        # Let each facet autoscale its y-range
        ax.relim()
        ax.autoscale(axis="y")

        ax.set_xlim(amin_min, df_trend["Amin_km2"].max())
        ax.set_title(group, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Trend β₁ (log-rate per year)")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Minimum fire size (km²)")

    # Remove unused axes when the number of groups is not a multiple of ncols
    for j in range(len(groups), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()


def plot_distribution_evolution_ccdf_pdf(
    df_both: pd.DataFrame,
    fires: pd.DataFrame,
    df_p1: pd.DataFrame | None = None,
    df_p2: pd.DataFrame | None = None,
    xmin=None,
    group_col: str = "modis_class_static",
    year_col: str = "year",
    trend_unit_years=None,
    x_max_log10: int = 3,
    ccdf_ymin: float = 1e-5,
) -> None:
    """For each (group, distribution) row in `df_both`, draw CCDF (log–log, normalized at xmin)
    and corresponding PDF (linear y, log x). Time evolution only included when the parameter
    slope is significant both in `df_both` and the corresponding single-parameter table.

    Replaces ``plot_distribution_evolution_ccdf_pdf`` and
    ``plot_refined_distribution_evolution_ccdf_pdf`` (``group_col="biome_refined"``).

    Args:
        df_both: Summary table with columns ["biome","distribution","p1","p2","p1'","p2'","p1_slope_sig","p2_slope_sig"].
        fires: Input data for deriving per-group year ranges.
        df_p1: Single-parameter (p1-only) significance table (optional).
        df_p2: Single-parameter (p2-only) significance table (optional).
        xmin: Tail cutoff for distributions (default ``min_fire``).
        group_col: Column with group labels in `fires`.
        year_col: Year column in `fires`.
        trend_unit_years: Years corresponding to “+1” in slope multipliers.
        x_max_log10: Max exponent for logspace grid (10^x_max_log10).
        ccdf_ymin: Lower limit for CCDF plots (default 1e-5).
    """
    xmin = resolve(xmin, "min_fire")
    trend_unit_years = resolve(trend_unit_years, "trend_unit_years")

    # Build lookup tables for significance of p1 and p2 slopes in single-parameter fits
    sig_lookup_p1: dict[tuple[str, str], bool] = {}
    sig_lookup_p2: dict[tuple[str, str], bool] = {}
    if df_p1 is not None:
        for _, r in df_p1.iterrows():
            sig_lookup_p1[(r["biome"], r["distribution"])] = int(r.get("p1_slope_sig", 0)) == 1
    if df_p2 is not None:
        for _, r in df_p2.iterrows():
            sig_lookup_p2[(r["biome"], r["distribution"])] = int(r.get("p2_slope_sig", 0)) == 1

    # Precompute min/max year per group for scaling and selecting a small set of time slices
    year_summary = (
        fires.dropna(subset=[group_col, year_col]).groupby(group_col)[year_col].agg(["min", "max"]).to_dict("index")
    )

    # Common x-grid used for all distributions and time slices
    x = np.logspace(np.log10(xmin), x_max_log10, 500)
    cmap = cm.viridis

    for _, row in df_both.iterrows():
        biome = row["biome"]
        dist = row["distribution"]

        # Skip groups without year info or with degenerate time ranges
        if biome not in year_summary:
            print(f"Skipping {biome} — no year data.")
            continue
        y0, y1 = int(year_summary[biome]["min"]), int(year_summary[biome]["max"])
        if y0 >= y1:
            print(f"Skipping {biome} — only one year.")
            continue

        # Choose a small set of evenly spaced years across the observed range
        # TODO: centres on this group's year range, not the MLE's global mean year.
        years = np.linspace(y0, y1, 5)
        year_center = np.mean(years)
        norm = plt.Normalize(vmin=y0, vmax=y1)

        key = (biome, dist)
        sig_p1_both = int(row.get("p1_slope_sig", 0)) == 1 and sig_lookup_p1.get(key, False)
        sig_p2_both = int(row.get("p2_slope_sig", 0)) == 1 and sig_lookup_p2.get(key, False)

        def p1(year: float) -> float:
            base = float(row["p1"])
            slope = float(row.get("p1'", 0) or 0)
            if not sig_p1_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        def p2(year: float) -> float | None:
            if not np.isfinite(row.get("p2", np.nan)):
                return None
            base = float(row["p2"])
            slope = float(row.get("p2'", 0) or 0)
            if not sig_p2_both or not np.isfinite(slope):
                return base
            return base + slope * ((year - year_center) / trend_unit_years)

        # Set up two-panel figure: CCDF (log–log) and approximate PDF (from CCDF derivative)
        fig, (ax_ccdf, ax_pdf) = plt.subplots(1, 2, figsize=(11, 4))
        good = False

        for yr in years:
            try:
                y_ccdf = ccdf_dispatch(dist, x, p1(yr), p2(yr), xmin)
                if y_ccdf is None:
                    continue
                y_pdf = np.gradient(1 - y_ccdf, x)

                ax_ccdf.plot(x, y_ccdf, color=cmap(norm(yr)), lw=1.5, label=f"{int(yr)}")
                ax_pdf.plot(x, y_pdf, color=cmap(norm(yr)), lw=1.5)
                good = True
            except Exception:
                continue

        if not good:
            plt.close()
            print(f"Skipped {biome} ({dist}) — invalid CCDF/PDF.")
            continue

        ax_ccdf.set_xscale("log")
        ax_ccdf.set_yscale("log")
        ax_ccdf.set_ylim(ccdf_ymin, 1.0)
        ax_ccdf.set_xlabel("Fire size (km²)")
        ax_ccdf.set_ylabel("CCDF")
        ax_ccdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (CCDF)")
        ax_ccdf.grid(True, alpha=0.3)

        ax_pdf.set_xscale("log")
        ax_pdf.set_xlabel("Fire size (km²)")
        ax_pdf.set_ylabel("PDF")
        ax_pdf.set_title(f"{biome}\n{dist.replace('_',' ').title()} (PDF)")
        ax_pdf.grid(True, alpha=0.3)

        # Add a colorbar indicating which curve corresponds to which year
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_label("Year", rotation=270, labelpad=15)
        plt.show()
