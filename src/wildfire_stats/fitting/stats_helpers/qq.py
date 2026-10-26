"""Quantile functions for Q–Q plots (closed forms conditioned on X >= xmin)."""

import numpy as np
from scipy.stats import norm

QQ_GRID_POINTS = 4000


def q_trunc_lognormal(p, mu, sigma, xmin_local):
    """Quantile for lognormal truncated below xmin."""
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    sigma = float(max(sigma, 1e-8))
    z_min = (np.log(xmin_local) - mu) / sigma
    c0 = norm.cdf(z_min)
    target = c0 + p * (1.0 - c0)
    z = norm.ppf(target)
    return float(np.exp(mu + sigma * z))


def q_gpd_tail(p, xi, sigma, xmin_local):
    """Quantile for GPD tail above xmin: X = xmin + Y, Y~GPD(xi,sigma)."""
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    sigma = float(max(sigma, 1e-10))
    xi = float(xi)
    u = -np.log(1.0 - p)  # >= 0

    if abs(xi) < 1e-6:
        y = sigma * u
    else:
        z = np.clip(xi * u, -50.0, 50.0)
        y = sigma / xi * (np.exp(z) - 1.0)

    y = max(y, 0.0)
    return float(xmin_local + y)


def q_power_law(p, alpha, xmin_local):
    """Quantile for X>=xmin: CCDF(x) = (x/xmin)^(1-alpha)."""
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    if alpha == 1:
        return float(xmin_local)
    x = xmin_local * (1.0 - p) ** (1.0 / (1.0 - alpha))
    return float(max(x, xmin_local))


def q_stretched_exp(p, lam, beta, xmin_local):
    """
    Quantile for stretched exponential tail:
        CCDF(x) = exp(-(lam*x)^beta + (lam*xmin)^beta), x>=xmin.
    """
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    lam = float(max(lam, 1e-12))
    beta = float(max(beta, 1e-6))
    term0 = (lam * xmin_local) ** beta
    term1 = -np.log(1.0 - p)
    base = term0 + term1
    base = max(base, 1e-12)
    x = (base ** (1.0 / beta)) / lam
    return float(max(x, xmin_local))


def q_weibull(p, k, lam, xmin_local):
    """
    Quantile for Weibull tail conditional on X>=xmin:
        CCDF(x) = exp(-(x/lam)^k + (xmin/lam)^k).
    """
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    lam = float(max(lam, 1e-12))
    k = float(max(k, 1e-6))
    term0 = (xmin_local / lam) ** k
    term1 = -np.log(1.0 - p)
    base = term0 + term1
    base = max(base, 1e-12)
    x = lam * (base ** (1.0 / k))
    return float(max(x, xmin_local))


def q_exponential(p, lam, xmin_local):
    """Quantile for exponential tail above xmin: X = xmin + Exp(lam)."""
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    lam = float(max(lam, 1e-12))
    y = -np.log(1.0 - p) / lam
    return float(xmin_local + max(y, 0.0))


def quantile_for_dist(p, dist_name, p1_t, p2_t, xmin_local):
    """Dispatch to the correct quantile formula."""
    if dist_name == "lognormal":
        return q_trunc_lognormal(p, p1_t, p2_t, xmin_local)
    elif dist_name == "generalized_pareto":
        return q_gpd_tail(p, p1_t, p2_t, xmin_local)
    elif dist_name == "power_law":
        return q_power_law(p, p1_t, xmin_local)
    elif dist_name == "stretched_exponential":
        return q_stretched_exp(p, p1_t, p2_t, xmin_local)
    elif dist_name == "weibull":
        return q_weibull(p, p1_t, p2_t, xmin_local)
    elif dist_name == "exponential":
        return q_exponential(p, p1_t, xmin_local)
    elif dist_name == "truncated_power_law":
        raise ValueError("Nonstationary QQ for 'truncated_power_law' not implemented.")
    else:
        raise ValueError(f"Unsupported distribution '{dist_name}' for nonstationary QQ.")


def qq_theoretical(model, probs: np.ndarray, lo: float, hi: float, ngrid: int = QQ_GRID_POINTS) -> np.ndarray:
    """Theoretical quantiles of a wfpl distribution at plotting positions ``probs``.

    Uses ``model.ppf`` if present, otherwise numerically inverts ``model.cdf`` on a
    log grid over [lo, hi].
    """
    # If the model exposes a ppf, use it directly; otherwise invert its CDF numerically
    if hasattr(model, "ppf") and callable(model.ppf):
        return model.ppf(probs)
    xgrid = np.logspace(np.log10(lo), np.log10(hi), ngrid)
    try:
        cdf_vals = model.cdf(xgrid)
        if np.ndim(cdf_vals) > 1:
            cdf_vals = np.asarray(cdf_vals).squeeze()
    except Exception:
        cdf_vals = np.array([float(model.cdf(float(x))) for x in xgrid])

    # Enforce well-behaved CDF and invert to get quantiles
    cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)
    cdf_vals = np.maximum.accumulate(cdf_vals)
    return np.interp(probs, cdf_vals, xgrid)
