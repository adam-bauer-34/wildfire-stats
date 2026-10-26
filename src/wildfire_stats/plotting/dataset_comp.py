"""Figures comparing fire-size distributions across datasets (notebook 1)."""

import matplotlib.pyplot as plt
import numpy as np

from wildfire_stats.fitting.stats_helpers.kuiper import kuiper_scan


def _plot_ecdf_pair(df1, df2, labels, area_col, title):
    plt.figure(figsize=(8, 6))

    # Compute and plot empirical CDFs
    for data, label in [(df1, labels[0]), (df2, labels[1])]:
        sizes = np.sort(data[area_col].values)
        cdf = np.arange(1, len(sizes) + 1) / len(sizes)
        plt.plot(sizes, cdf, label=label)

    plt.xscale("log")
    plt.xlabel("Fire size (km², log scale)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


def plot_cdf_vs_area(df1, df2, area_col="area_km2", labels=("MTBS", "Idaho")):
    """Plot empirical CDFs of fire sizes for two datasets on a log-x axis.

    Args:
        df1, df2 (DataFrame): Datasets containing the fire size column.
        area_col (str, optional): Fire area column (km²).
        labels (tuple[str, str]): Legend labels for ``df1`` and ``df2``.
    """
    _plot_ecdf_pair(df1, df2, labels, area_col, "CDF of Fire Sizes")


def plot_alpha_vs_kuiper_with_min_cdf(df1, df2, alphas, area_col="area_km2", labels=("MTBS", "Idaho")):
    """Analyze sensitivity of Kuiper statistic across fire size thresholds.

    For a range of `alpha` thresholds, computes the Kuiper statistic, plots it
    against alpha, highlights the minimizing alpha, and re-plots both CDFs
    filtered at that alpha. This is where ``min_fire`` comes from.

    Args:
        df1, df2 (DataFrame): Datasets containing the fire size column.
        alphas (Iterable[float]): Threshold values (km²) to evaluate.
        area_col (str, optional): Fire size column.
        labels (tuple[str, str]): Legend labels for ``df1`` and ``df2``.

    Returns:
        tuple: (alpha minimizing the Kuiper statistic, minimum Kuiper value).
    """
    kuipers, min_alpha, min_kuiper = kuiper_scan(df1, df2, alphas, area_col)

    # Plot alpha vs. Kuiper statistic
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, kuipers, marker="o")
    plt.axvline(min_alpha, color="red", linestyle="--", label=f"Min Kuiper at alpha={min_alpha:.2f}")
    plt.xlabel("Alpha (km² threshold)")
    plt.ylabel("Kuiper Statistic")
    plt.title("Alpha vs. Kuiper Statistic")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.show()

    print(f"Minimum Kuiper statistic {min_kuiper:.4f} occurs at alpha = {min_alpha:.4f} km²")

    # Plot CDFs for datasets filtered at the minimizing alpha
    _plot_ecdf_pair(
        df1[df1[area_col] > min_alpha],
        df2[df2[area_col] > min_alpha],
        labels,
        area_col,
        f"CDF of Fire Sizes for alpha > {min_alpha:.4f} km² (min Kuiper)",
    )

    return min_alpha, min_kuiper
