# riskparity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaehyukchoi/pypackagetest/blob/main/notebooks/demo.ipynb)

Risk parity portfolio optimization via the **improved CCD method** of Choi & Chen (2022).

> Choi, J., & Chen, R. (2022). Improved iterative methods for solving risk parity portfolio.
> *Journal of Derivatives and Quantitative Studies*, 30(2).
> https://doi.org/10.1108/JDQS-12-2021-0031

## What is risk parity?

A risk parity (equal risk contribution) portfolio chooses weights **w** so that every
asset contributes the same fraction of total portfolio volatility:

$$\frac{w_i (\mathbf{C}\mathbf{w})_i}{\mathbf{w}^\top \mathbf{C}\mathbf{w}} = b_i \quad \forall i$$

where **C** is the return covariance matrix and **b** is the risk-budget vector
(defaults to `1/N` for equal risk parity).

## Algorithm

The improved CCD method (Algorithm 1 in the paper) works on the **correlation** matrix
and adds a **rescaling step** after each sweep, making it ~3× faster than the original
CCD method and saving ~40% of iterations.

## Installation

```bash
pip install mafn-pypackagetest
```

## Quick start

```python
import numpy as np
from riskparity import risk_parity

# 3-asset covariance matrix
cov = np.array([
    [0.04,  0.01,  0.002],
    [0.01,  0.09,  0.015],
    [0.002, 0.015, 0.0025],
])

result = risk_parity(cov)
print(result.weights)            # portfolio weights, sum to 1
print(result.risk_contributions) # fractional risk contributions, sum to 1
print(result.iterations)         # number of CCD iterations
print(result.converged)          # True if tolerance was met
```

### Custom risk budgets

```python
b = np.array([0.5, 0.3, 0.2])   # 50 / 30 / 20 % risk allocation
result = risk_parity(cov, b=b)
```

### Standalone risk contribution utility

```python
from riskparity import risk_contributions
rc = risk_contributions(weights, cov)  # fractional RC for any portfolio
```

## API reference

### `risk_parity(cov, b=None, tol=1e-6, max_iter=1000)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `cov` | `(N, N)` array | Covariance matrix (positive semi-definite) |
| `b` | `(N,)` array or `None` | Risk budgets, must be positive and sum to 1. Default: `1/N` |
| `tol` | `float` | Convergence tolerance (default `1e-6`) |
| `max_iter` | `int` | Maximum iterations (default `1000`) |

Returns a `RiskParityResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `(N,)` array | Portfolio weights, sum to 1 |
| `risk_contributions` | `(N,)` array | Fractional risk contributions, sum to 1 |
| `iterations` | `int` | Outer iterations used |
| `converged` | `bool` | Whether tolerance was met |

### `risk_contributions(weights, cov)`

Compute fractional risk contributions for any portfolio.

## License

MIT
