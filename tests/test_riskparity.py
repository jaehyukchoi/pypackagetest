"""Tests for the improved CCD risk parity solver."""

import numpy as np
import pytest
from scipy.stats import random_correlation

from riskparity import RiskParityResult, risk_contributions, risk_parity


# ── helpers ──────────────────────────────────────────────────────────────────

def make_cov_from_corr(corr: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Build a covariance matrix from a correlation matrix and vol vector."""
    return corr * np.outer(vols, vols)


def assert_risk_parity(weights, cov, b, tol=1e-5):
    """Assert that weights achieve the target risk budgets b."""
    rc = risk_contributions(weights, cov)
    np.testing.assert_allclose(rc, b, atol=tol)


# ── basic correctness ─────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = risk_parity(cov)
    assert np.isclose(result.weights.sum(), 1.0)


def test_equal_risk_contributions_2d():
    cov = np.array([[0.04, 0.02], [0.02, 0.09]])
    result = risk_parity(cov)
    assert result.converged
    assert_risk_parity(result.weights, cov, np.array([0.5, 0.5]))


def test_equal_risk_contributions_3d():
    rng = np.random.default_rng(0)
    vols = rng.uniform(0.05, 0.30, 3)
    eigs = rng.uniform(0, 1, 3)
    eigs /= eigs.sum() / 3
    R = random_correlation.rvs(eigs, random_state=rng)
    cov = make_cov_from_corr(R, vols)
    result = risk_parity(cov)
    assert result.converged
    N = 3
    assert_risk_parity(result.weights, cov, np.full(N, 1.0 / N))


def test_risk_budgets():
    """Non-equal risk budgets should yield those exact fractional contributions."""
    cov = np.array([[0.04, 0.01, 0.005],
                    [0.01, 0.09, 0.02],
                    [0.005, 0.02, 0.01]])
    b = np.array([0.5, 0.3, 0.2])
    result = risk_parity(cov, b=b)
    assert result.converged
    assert_risk_parity(result.weights, cov, b)


def test_uniform_correlation_analytical():
    """
    When all pairwise correlations are equal, the risk parity weight is
    proportional to 1/σ_i (Eq. 7 of the paper).
    """
    N = 5
    rho = 0.4
    vols = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    R = np.full((N, N), rho)
    np.fill_diagonal(R, 1.0)
    cov = make_cov_from_corr(R, vols)

    result = risk_parity(cov)
    expected = (1.0 / vols) / (1.0 / vols).sum()
    np.testing.assert_allclose(result.weights, expected, atol=1e-5)


def test_diagonal_covariance():
    """For a diagonal covariance, weights are 1/σ_i (no correlations)."""
    vols = np.array([0.10, 0.20, 0.40])
    cov = np.diag(vols ** 2)
    result = risk_parity(cov)
    expected = (1.0 / vols) / (1.0 / vols).sum()
    np.testing.assert_allclose(result.weights, expected, atol=1e-6)
    assert result.converged


# ── positive semi-definite covariance ────────────────────────────────────────

def test_positive_semidefinite():
    """Rank-deficient covariance (Test 2 setting from the paper)."""
    rng = np.random.default_rng(42)
    N = 10
    eigs = rng.uniform(0, 1, N)
    eigs[-2:] = 0.0           # 20% zero eigenvalues
    eigs /= eigs.sum() / N
    R = random_correlation.rvs(eigs, random_state=rng)
    vols = rng.uniform(0.05, 0.30, N)
    cov = make_cov_from_corr(R, vols)

    result = risk_parity(cov)
    assert result.converged
    np.testing.assert_allclose(result.weights.sum(), 1.0)


# ── large portfolio ───────────────────────────────────────────────────────────

def test_large_portfolio():
    """Smoke-test convergence for N=100."""
    rng = np.random.default_rng(7)
    N = 100
    eigs = rng.uniform(0, 1, N)
    eigs /= eigs.sum() / N
    R = random_correlation.rvs(eigs, random_state=rng)
    vols = rng.uniform(0.05, 0.30, N)
    cov = make_cov_from_corr(R, vols)

    result = risk_parity(cov)
    assert result.converged
    assert result.iterations < 50        # should converge quickly
    assert_risk_parity(result.weights, cov, np.full(N, 1.0 / N), tol=1e-4)


# ── iteration / result metadata ──────────────────────────────────────────────

def test_result_type():
    cov = np.eye(3) * 0.01
    result = risk_parity(cov)
    assert isinstance(result, RiskParityResult)
    assert result.iterations >= 1
    assert result.converged


def test_risk_contributions_utility():
    w = np.array([0.5, 0.5])
    cov = np.eye(2) * 0.01
    rc = risk_contributions(w, cov)
    np.testing.assert_allclose(rc, [0.5, 0.5])


# ── input validation ──────────────────────────────────────────────────────────

def test_bad_budget_not_sum_to_one():
    cov = np.eye(3)
    with pytest.raises(ValueError, match="sum to 1"):
        risk_parity(cov, b=np.array([0.4, 0.4, 0.4]))


def test_bad_budget_negative():
    cov = np.eye(3)
    with pytest.raises(ValueError, match="strictly positive"):
        risk_parity(cov, b=np.array([0.5, 0.6, -0.1]))


def test_non_square_cov():
    with pytest.raises(ValueError):
        risk_parity(np.ones((3, 4)))
