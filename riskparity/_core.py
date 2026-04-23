"""
Improved CCD algorithm for risk parity portfolio.

Reference:
    Choi, J., & Chen, R. (2022). Improved iterative methods for solving
    risk parity portfolio. Journal of Derivatives and Quantitative Studies,
    30(2). https://doi.org/10.1108/JDQS-12-2021-0031
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RiskParityResult:
    """Result from :func:`risk_parity`."""

    weights: np.ndarray
    """Normalized portfolio weights, shape (N,), summing to 1."""

    risk_contributions: np.ndarray
    """Fractional risk contributions, shape (N,), summing to 1."""

    iterations: int
    """Number of outer iterations performed."""

    converged: bool
    """Whether the tolerance was met before ``max_iter``."""


def risk_parity(
    cov: np.ndarray,
    b: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> RiskParityResult:
    """Compute risk parity portfolio weights via the improved CCD method.

    Solves the equal (or budgeted) risk contribution condition::

        w_i * (C w)_i = V²(w) * b_i   for all i,

    using the improved cyclical coordinate descent algorithm of Choi & Chen
    (2022), which works on the correlation matrix and adds a rescaling step
    after each sweep for faster convergence.

    Parameters
    ----------
    cov:
        Symmetric positive (semi-)definite covariance matrix, shape (N, N).
    b:
        Risk-budget vector, shape (N,).  Must be strictly positive and sum
        to 1.  Defaults to equal risk parity: ``b_i = 1/N``.
    tol:
        Convergence tolerance on ``max_i |w_i (R w)_i - b_i|``.
        Default ``1e-6`` matches the paper.
    max_iter:
        Maximum number of outer iterations.  Default 1000.

    Returns
    -------
    RiskParityResult
        Dataclass with fields ``weights``, ``risk_contributions``,
        ``iterations``, and ``converged``.

    Examples
    --------
    >>> import numpy as np
    >>> from riskparity import risk_parity
    >>> cov = np.array([[0.04, 0.02], [0.02, 0.09]])
    >>> result = risk_parity(cov)
    >>> result.weights          # sums to 1
    >>> result.risk_contributions  # should be close to [0.5, 0.5]
    """
    cov = np.asarray(cov, dtype=float)
    N = cov.shape[0]
    if cov.shape != (N, N):
        raise ValueError("cov must be a square 2-D array")

    if b is None:
        b = np.full(N, 1.0 / N)
    else:
        b = np.asarray(b, dtype=float)
        if b.shape != (N,):
            raise ValueError("b must have shape (N,)")
        if not np.all(b > 0):
            raise ValueError("all entries of b must be strictly positive")
        if not np.isclose(b.sum(), 1.0):
            raise ValueError("b must sum to 1")

    # Step 1: correlation matrix R and standard deviations σ
    sigma = np.sqrt(np.diag(cov))
    R = cov / np.outer(sigma, sigma)

    # Step 2: initial guess  w_i = 1 / sqrt(sum_{j,k} R_{jk})   [Eq. 8]
    ones = np.ones(N)
    w = ones / np.sqrt(ones @ R @ ones)

    # Step 3: improved CCD iterations
    converged = False
    n_iter = 0
    Rw = R @ w

    for n_iter in range(1, max_iter + 1):
        # Inner CCD sweep over all assets
        for i in range(N):
            # a_i = (1/2) sum_{j != i} R_{ij} w_j = (Rw_i - w_i) / 2   [Eq. 11]
            a_i = (Rw[i] - w[i]) / 2.0
            w_new_i = np.sqrt(a_i * a_i + b[i]) - a_i
            # Incrementally update Rw to keep it in sync
            delta = w_new_i - w[i]
            Rw += R[:, i] * delta
            w[i] = w_new_i

        # Rescaling step: w <- w / sqrt(w^T R w)   [Eq. 12]
        scale = np.sqrt(w @ Rw)
        w /= scale
        Rw /= scale

        # Convergence check: max_i |w_i (Rw)_i - b_i|
        err = np.max(np.abs(w * Rw - b))
        if err <= tol:
            converged = True
            break

    # Convert back to original-space weights: w_bar_i = (w_i/σ_i) / sum_k(w_k/σ_k)
    w_over_sigma = w / sigma
    weights = w_over_sigma / w_over_sigma.sum()

    return RiskParityResult(
        weights=weights,
        risk_contributions=risk_contributions(weights, cov),
        iterations=n_iter,
        converged=converged,
    )


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute fractional risk contributions for a portfolio.

    The marginal risk contribution of asset *i* is::

        rc_i = w_i * (C w)_i / V(w)

    where ``V(w) = sqrt(w^T C w)`` is portfolio volatility.  This function
    returns ``rc_i`` normalised to sum to 1.

    Parameters
    ----------
    weights:
        Portfolio weight vector, shape (N,).
    cov:
        Covariance matrix, shape (N, N).

    Returns
    -------
    np.ndarray
        Fractional risk contributions, shape (N,), summing to 1.
    """
    w = np.asarray(weights, dtype=float)
    C = np.asarray(cov, dtype=float)
    rc = w * (C @ w)
    return rc / rc.sum()
