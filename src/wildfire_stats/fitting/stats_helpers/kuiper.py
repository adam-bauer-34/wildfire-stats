"""Kuiper two-sample statistic, used to choose the min_fire cutoff (notebook 1)."""

import numpy as np


def kuiper_statistic(data1, data2):
    """Compute the Kuiper statistic between two one-dimensional samples.

    The Kuiper statistic is similar to the Kolmogorov–Smirnov statistic but
    is equally sensitive at the tails and the center of the distributions.

    Args:
        data1 (np.ndarray): First dataset (1D array-like).
        data2 (np.ndarray): Second dataset (1D array-like).

    Returns:
        float: Kuiper statistic value measuring distributional difference.
    """
    # Sort both datasets
    data1 = np.sort(data1)
    data2 = np.sort(data2)

    # Compute empirical CDFs across combined range
    n1, n2 = len(data1), len(data2)
    all_data = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, all_data, side='right') / n1
    cdf2 = np.searchsorted(data2, all_data, side='right') / n2

    # Compute maximum positive and negative differences
    d_plus = np.max(cdf1 - cdf2)
    d_minus = np.max(cdf2 - cdf1)

    # Kuiper statistic is the sum of both
    return d_plus + d_minus


def kuiper_for_alpha(mtbs_df, idaho_df, alpha, area_col="area_km2"):
    """Compute the Kuiper statistic after filtering datasets by a minimum fire size.

    Args:
        mtbs_df (DataFrame): MTBS dataset containing fire size column.
        idaho_df (DataFrame): Idaho dataset containing fire size column.
        alpha (float): Fire size threshold (km²) used to filter both datasets.
        area_col (str, optional): Name of the fire size column. Defaults to "area_km2".

    Returns:
        float: Kuiper statistic computed for the filtered datasets.
    """
    # Filter both datasets for fires larger than alpha
    mtbs_filt = mtbs_df[mtbs_df[area_col] > alpha]
    idaho_filt = idaho_df[idaho_df[area_col] > alpha]

    # Compute and return the Kuiper statistic
    return kuiper_statistic(mtbs_filt[area_col].values, idaho_filt[area_col].values)


def kuiper_scan(df1, df2, alphas, area_col="area_km2"):
    """Kuiper statistic at each threshold in ``alphas``.

    Returns:
        (kuipers, min_alpha, min_kuiper)
    """
    kuipers = np.array([kuiper_for_alpha(df1, df2, a, area_col) for a in alphas])
    min_idx = np.argmin(kuipers)
    return kuipers, alphas[min_idx], kuipers[min_idx]
