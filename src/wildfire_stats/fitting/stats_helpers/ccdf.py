"""Analytic CCDFs conditioned at xmin, used to draw time-varying fits."""

import numpy as np
from scipy.special import gamma, gammaincc
from scipy.stats import lognorm

from wildfire_stats.config import resolve


def ccdf_exponential(x: np.ndarray, lam: float, xmin=None) -> np.ndarray:
    """Compute the complementary CDF (CCDF) of an exponential distribution.

    Args:
        x: Input array of fire sizes.
        lam: Exponential rate parameter λ.
        xmin: Lower cutoff value.

    Returns:
        CCDF values normalized to 1 at xmin.
    """
    xmin = resolve(xmin, "min_fire")
    # Restrict support to values at or above xmin so the CCDF is well-defined on the tail
    mask = x >= xmin

    # Compute exponential CCDF shifted to start at xmin
    ccdf = np.exp(-lam * (x - xmin))

    # Normalize so that CCDF(xmin) = 1 (on the valid mask)
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip to avoid exact zeros or ones, which helps with log-scale plotting
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_power_law(x: np.ndarray, alpha: float, xmin=None) -> np.ndarray:
    """Compute the CCDF of a pure power-law distribution.

    Args:
        x: Fire sizes or other continuous data.
        alpha: Power-law exponent α (>1 typical).
        xmin: Minimum cutoff for valid values.

    Returns:
        CCDF normalized at xmin.
    """
    xmin = resolve(xmin, "min_fire")
    # Restrict to the domain x >= xmin where the power-law model is assumed to hold
    mask = x >= xmin

    # Compute CCDF in closed form relative to xmin
    ccdf = (x / xmin) ** (1 - alpha)

    # Normalize to 1 at xmin for comparability across distributions
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip to guard against numerical underflow/overflow
    return np.clip(ccdf, 1e-12, 1.0)


# TODO(bug): scipy's gammaincc(1 - alpha, .) is NaN for alpha > 1, so this
# always falls back to ccdf_power_law (HANDOFF 4g).
def ccdf_truncated_power_law(
    x: np.ndarray, alpha: float, lambd: float, xmin=None
) -> np.ndarray:
    """Stable truncated power-law CCDF.

    Automatically reduces to a pure power law when λ ≈ 0 or α is large.

    Args:
        x: Input array of data (e.g., fire sizes).
        alpha: Power-law exponent.
        lambd: Truncation rate λ.
        xmin: Minimum cutoff for validity.

    Returns:
        CCDF array (same shape as x), numerically stable.
    """
    xmin = resolve(xmin, "min_fire")
    # Ensure floating-point array for downstream operations
    x = np.asarray(x, dtype=float)

    # Identify values above the lower cutoff and precompute λx
    mask = x >= xmin
    z = lambd * x[mask]

    # If λ is very small or α is large, fall back to pure power law (truncation negligible)
    if lambd < 1e-3 or np.all(z < 1e-2) or alpha > 5:
        return ccdf_power_law(x, alpha, xmin)

    try:
        # Compute normalized incomplete gamma expression for the CCDF
        val = gammaincc(1 - alpha, z) / gamma(1 - alpha)

        # If the incomplete gamma evaluation is numerically unstable, fall back to power law
        if not np.any(np.isfinite(val)) or np.all(val < 1e-12):
            # fallback to power law if gamma underflows
            return ccdf_power_law(x, alpha, xmin)

        # Initialize CCDF and assign values only where x >= xmin
        ccdf = np.zeros_like(x)
        ccdf[mask] = np.clip(val, 1e-12, 1.0)

        # Normalize at xmin for consistency across distributions
        ccdf /= max(ccdf[mask][0], 1e-12)
    except Exception:
        # On any unexpected numerical error, revert to a pure power-law CCDF
        return ccdf_power_law(x, alpha, xmin)

    # Final clipping to ensure numerical robustness on log plots
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_stretched_exponential(
    x: np.ndarray, lam: float, beta: float, xmin=None
) -> np.ndarray:
    """Compute the CCDF of a stretched exponential (Weibull form).

    Args:
        x: Input array of values ≥ xmin.
        lam: Rate parameter λ.
        beta: Stretching exponent β.
        xmin: Lower cutoff for normalization.

    Returns:
        Normalized CCDF array.
    """
    xmin = resolve(xmin, "min_fire")
    # Restrict CCDF definition to tail values x >= xmin
    mask = x >= xmin

    # Compute stretched exponential CCDF shifted to normalize at xmin
    ccdf = np.exp(-(lam * x) ** beta + (lam * xmin) ** beta)

    # Normalize to CCDF(xmin) = 1
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip extremes for numerical stability
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_weibull(x: np.ndarray, k: float, lam: float, xmin=None) -> np.ndarray:
    """Compute the CCDF of a Weibull distribution.

    Args:
        x: Input data array.
        k: Shape parameter (β in some notation).
        lam: Scale parameter λ.
        xmin: Lower truncation cutoff.

    Returns:
        CCDF normalized to 1 at xmin.
    """
    xmin = resolve(xmin, "min_fire")
    # Restrict CCDF to the domain x >= xmin
    mask = x >= xmin

    # Compute Weibull CCDF with a shift to normalize at xmin
    ccdf = np.exp(-((x / lam) ** k - (xmin / lam) ** k))

    # Normalize to CCDF(xmin) = 1
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip to avoid exactly 0 or 1 on log-scales
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_lognormal(x: np.ndarray, mu: float, sigma: float, xmin=None) -> np.ndarray:
    """Compute the CCDF of a lognormal distribution.

    Args:
        x: Input values.
        mu: Log-mean of the distribution.
        sigma: Log-standard deviation.
        xmin: Minimum value for normalization.

    Returns:
        Normalized CCDF array.
    """
    xmin = resolve(xmin, "min_fire")
    # Restrict CCDF to x >= xmin for the tail behavior
    mask = x >= xmin

    # Use SciPy's lognormal survival function to compute CCDF
    ccdf = lognorm.sf(x, sigma, scale=np.exp(mu))

    # Normalize at xmin so CCDF(xmin) = 1
    ccdf /= max(ccdf[mask][0], 1e-12)

    # Clip for numerical safety
    return np.clip(ccdf, 1e-12, 1.0)


def ccdf_genpareto(x: np.ndarray, xi: float, sigma: float, xmin=None) -> np.ndarray:
    """Compute CCDF for the Generalized Pareto Distribution (GPD).

    Args:
        x: Input data (e.g., fire sizes).
        xi: Shape parameter ξ.
        sigma: Scale parameter σ.
        xmin: Lower truncation bound.

    Returns:
        CCDF array normalized at xmin.
    """
    xmin = resolve(xmin, "min_fire")
    # Restrict to tail values and shift domain to start at xmin
    mask = x >= xmin
    y = x[mask] - xmin

    # Use exponential limit when shape parameter is very close to zero
    if abs(xi) < 1e-6:
        ccdf = np.exp(-y / sigma)
    else:
        # Standard GPD CCDF expression
        ccdf = (1 + xi * y / sigma) ** (-1 / xi)

    # Initialize full output and assign computed CCDF only to valid entries
    full_ccdf = np.zeros_like(x)
    full_ccdf[mask] = ccdf

    # Normalize to CCDF(xmin) = 1
    full_ccdf /= max(full_ccdf[mask][0], 1e-12)

    # Final clipping for numerical stability
    return np.clip(full_ccdf, 1e-12, 1.0)


def ccdf_dispatch(dist: str, x: np.ndarray, p1: float, p2: float | None, xmin: float) -> np.ndarray | None:
    """CCDF of ``dist`` with parameters (p1, p2) on grid ``x``; None for unsupported names."""
    if dist == "exponential":
        return ccdf_exponential(x, p1, xmin)
    elif dist == "power_law":
        return ccdf_power_law(x, p1, xmin)
    elif dist == "truncated_power_law":
        return ccdf_truncated_power_law(x, p1, max(p2, 1e-8), xmin)
    elif dist == "stretched_exponential":
        return ccdf_stretched_exponential(x, p1, p2, xmin)
    elif dist == "weibull":
        return ccdf_weibull(x, p1, p2, xmin)
    elif dist == "lognormal":
        return ccdf_lognormal(x, p1, p2, xmin)
    elif dist == "generalized_pareto":
        return ccdf_genpareto(x, p1, p2, xmin)
    return None
