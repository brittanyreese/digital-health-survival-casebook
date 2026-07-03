"""
CFA-based ordinal item generation via Gaussian copula.

For each set of items, the model is:
    x*_ij = λ_j · ξ_i + δ_ij
where ξ_i ~ N(μ_stage, Σ_factor) is the latent factor score (or vector of
factor scores) and δ_ij ~ N(0, ψ_j) is unique variance with ψ_j = 1 - λ_j².

Ordinal responses are obtained by applying uniform-spacing probit thresholds
to the standardised latent score x*_ij.  This is a close approximation to
Muthén's (1984) polychoric parameterisation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _probit_thresholds(n_points: int) -> np.ndarray:
    """Equally-spaced probit thresholds for n_points-category Likert items."""
    return norm.ppf(np.linspace(0, 1, n_points + 1)[1:-1])


def generate_one_factor_items(
    n: int,
    n_items: int,
    loading: float,
    factor_mean: float,
    factor_sd: float = 1.0,
    n_points: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate item responses for a single-factor block with uniform loadings.

    Parameters
    ----------
    n          : sample size
    n_items    : number of items
    loading    : common factor loading (λ) for all items
    factor_mean: stage-conditional factor mean (in z-score units)
    factor_sd  : factor SD (default 1; keeps scale standard)
    n_points   : number of Likert response categories

    Returns
    -------
    responses  : (n, n_items) int array, values in 1..n_points
    """
    if rng is None:
        rng = np.random.default_rng()

    # Factor scores
    xi = rng.normal(factor_mean, factor_sd, size=(n, 1))  # (n,1)

    # Latent item scores
    loadings_vec = np.full(n_items, loading)  # (n_items,)
    x_star = xi * loadings_vec  # (n, n_items)

    # Residual noise: ψ = 1 - λ²
    psi = np.sqrt(np.clip(1 - loading ** 2, 0.01, None))
    x_star += rng.normal(0, psi, size=(n, n_items))

    # Discretize
    thresholds = _probit_thresholds(n_points)
    return np.digitize(x_star, thresholds) + 1


def generate_two_factor_items(
    n: int,
    loadings_f1: np.ndarray,
    loadings_f2: np.ndarray,
    factor_means: np.ndarray,   # shape (2,)
    factor_cov: np.ndarray,     # shape (2,2)
    n_points: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate item responses for a two-factor block (items load on one factor only).

    Items from factor 1 come first, then factor 2.

    Returns
    -------
    responses : (n, n1+n2) int array
    """
    if rng is None:
        rng = np.random.default_rng()

    n1, n2 = len(loadings_f1), len(loadings_f2)
    n_items = n1 + n2

    # Joint factor scores
    xi = rng.multivariate_normal(factor_means, factor_cov, size=n)  # (n,2)

    # Build latent item scores: each item loads on exactly one factor
    x_star = np.zeros((n, n_items))
    for j, lam in enumerate(loadings_f1):
        psi = np.sqrt(np.clip(1 - lam ** 2, 0.01, None))
        x_star[:, j] = xi[:, 0] * lam + rng.normal(0, psi, size=n)
    for j, lam in enumerate(loadings_f2):
        psi = np.sqrt(np.clip(1 - lam ** 2, 0.01, None))
        x_star[:, n1 + j] = xi[:, 1] * lam + rng.normal(0, psi, size=n)

    thresholds = _probit_thresholds(n_points)
    return np.digitize(x_star, thresholds) + 1
