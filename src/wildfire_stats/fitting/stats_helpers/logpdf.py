"""Vectorized log-densities used by the time-varying MLE.

These were nested inside ``analyze_time_varying_mle`` (and its ``_refined`` and
notebook-7 ``_evtname`` copies). Parameters may be arrays (one value per fire).
``clips`` maps parameter name -> (lo, hi); the defaults are the original
hard-coded ranges, mirrored in ``config/dist_bounds_param_clips.yaml``
(``logpdf_clips``).
"""

import numpy as np
from scipy.special import gamma, gammaincc

DEFAULT_CLIPS = {
    "lognormal": {"sigma": (1e-6, 50)},
    "power_law": {"alpha": (0.0, 5)},
    "truncated_power_law": {"alpha": (0, 5), "lambd": (0, 5), "Z": (1e-300, 1e300)},
    "generalized_pareto": {"xi": (-1, 2), "sigma": (1e-6, 10)},
    "weibull": {"k": (1e-6, 50), "lam": (1e-6, 50), "z": (1e-12, 1e6)},
    "stretched_exponential": {"lam": (1e-6, 50), "beta": (0.1, 5)},
}


# TODO(bug): not conditioned on x >= xmin (missing -log SF(xmin)), unlike the
# static wfpl fit (HANDOFF 4j).
def logpdf_lognormal(x: np.ndarray, mu: float, sigma: float, xmin: float = 0, clips: dict | None = None) -> np.ndarray:
    c = DEFAULT_CLIPS["lognormal"] if clips is None else clips
    x = np.asarray(x)
    valid = x > xmin
    pdf = -np.inf * np.ones_like(x, dtype=float)

    # Avoid degenerate sigma by clipping to a reasonable range
    sigma = np.clip(sigma, *c["sigma"])

    # Standard lognormal log-density for valid support
    pdf[valid] = (
        -np.log(x[valid]) - np.log(sigma)
        - 0.5 * ((np.log(x[valid]) - mu) / sigma) ** 2
        - np.log(np.sqrt(2 * np.pi))
    )
    return pdf


# TODO(bug): `alpha != 1` on an array raises (caught upstream -> inf), and
# log(abs(C)) lets alpha < 1 score as valid (HANDOFF 4l).
def logpdf_powerlaw(x: np.ndarray, alpha: float, xmin: float = 1, clips: dict | None = None) -> np.ndarray:
    c = DEFAULT_CLIPS["power_law"] if clips is None else clips
    x = np.asarray(x)
    valid = x >= xmin
    pdf = -np.inf * np.ones_like(x, dtype=float)

    # Clip exponent to a plausible range for numerical stability
    alpha = np.clip(alpha, *c["alpha"])

    # Normalization constant for power-law density (with special-case α=1)
    C = (alpha - 1) / xmin if alpha != 1 else 1 / xmin

    # Log-density for valid x
    pdf[valid] = np.log(np.abs(C)) - alpha * np.log(x[valid] / xmin)
    return pdf


# TODO(bug): gammaincc(1 - alpha, .) is NaN for alpha > 1, so the whole
# log-likelihood is NaN and the fit sees only the prior; the normaliser also
# uses lambda**(1 - alpha) instead of lambda**(alpha - 1) (HANDOFF 4g).
def logpdf_trunc_powerlaw(x: np.ndarray, alpha: float, lambd: float, xmin: float = 1, clips: dict | None = None) -> np.ndarray:
    """Properly normalized truncated power law; stable for small/large α, λ."""
    c = DEFAULT_CLIPS["truncated_power_law"] if clips is None else clips
    x = np.asarray(x)
    valid = x >= xmin
    pdf = -np.inf * np.ones_like(x, dtype=float)

    # Clip α and λ to avoid excessively extreme values
    alpha = np.clip(alpha, *c["alpha"])
    lambd = np.clip(lambd, *c["lambd"])

    try:
        # Compute normalization constant using incomplete gamma for truncation at xmin
        Z = (
            (lambd ** (1 - alpha))
            * np.exp(lambd * xmin)
            * gammaincc(1 - alpha, lambd * xmin)
            * gamma(1 - alpha)
        )
        # Clip Z to avoid log underflow/overflow
        Z = np.clip(Z, *c["Z"])
        logZ = np.log(Z)
    except Exception:
        # On failure, fall back to unnormalized log-density (still usable for optimization)
        logZ = 0.0

    # Log-density over valid range
    pdf[valid] = -alpha * np.log(x[valid]) - lambd * (x[valid] - xmin) - logZ
    return pdf


# TODO(bug): no xi -> 0 branch; 1/xi blows up at the exponential limit (HANDOFF 4k).
# TODO(bug): with array (time-varying) xi/sigma, `xi * y[valid]` fails to broadcast
# as soon as any point is outside the support (xi < 0), so neg_loglik returns inf.
def logpdf_genpareto(x: np.ndarray, xi: float, sigma: float, xmin: float = 0, clips: dict | None = None) -> np.ndarray:
    c = DEFAULT_CLIPS["generalized_pareto"] if clips is None else clips
    x = np.asarray(x)
    y = x - xmin
    pdf = -np.inf * np.ones_like(x, dtype=float)

    # Clip shape and scale into a stable, valid range
    xi = np.clip(xi, *c["xi"])
    sigma = np.clip(sigma, *c["sigma"])

    # Guard against invalid combinations that would yield negative support
    valid = (sigma > 0) & (1 + xi * y / sigma > 0)

    # Standard GPD log-density
    pdf[valid] = -np.log(sigma) - (1 / xi + 1) * np.log(1 + xi * y[valid] / sigma)
    return pdf


# TODO(bug): not conditioned on x >= xmin (missing +(xmin/lam)^k) (HANDOFF 4j).
def logpdf_weibull(x: np.ndarray, k: float, lam: float, xmin: float = 0, clips: dict | None = None) -> np.ndarray:
    c = DEFAULT_CLIPS["weibull"] if clips is None else clips
    x = np.asarray(x)
    valid = x >= xmin
    pdf = -np.inf * np.ones_like(x, dtype=float)

    # Clip shape and scale to avoid degeneracy
    k = np.clip(k, *c["k"])
    lam = np.clip(lam, *c["lam"])

    # Compute dimensionless ratio and ensure it is bounded
    z = np.clip(x[valid] / lam, *c["z"])

    # Weibull log-density expression
    pdf[valid] = np.log(k) - np.log(lam) + (k - 1) * np.log(z) - z ** k
    return pdf


def logpdf_stretched_exp(x: np.ndarray, lam: float, beta: float, xmin: float = 0, clips: dict | None = None) -> np.ndarray:
    c = DEFAULT_CLIPS["stretched_exponential"] if clips is None else clips
    x = np.asarray(x)
    valid = x >= xmin
    pdf = -np.inf * np.ones_like(x, dtype=float)

    # Constrain parameters to plausible ranges
    lam = np.clip(lam, *c["lam"])
    beta = np.clip(beta, *c["beta"])

    # Stretched exponential (Weibull-form) log-density with shift to xmin
    pdf[valid] = (
        np.log(beta)
        + np.log(lam)
        + (beta - 1) * (np.log(x[valid]) + np.log(lam))
        + (lam * xmin) ** beta
        - (lam * x[valid]) ** beta
    )
    return pdf


# Distribution name -> log-PDF function
DIST_LOGPDFS = {
    "lognormal": logpdf_lognormal,
    "power_law": logpdf_powerlaw,
    "truncated_power_law": logpdf_trunc_powerlaw,
    "generalized_pareto": logpdf_genpareto,
    "weibull": logpdf_weibull,
    "stretched_exponential": logpdf_stretched_exp,
}
