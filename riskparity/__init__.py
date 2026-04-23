"""
riskparity — Risk parity portfolio via the improved CCD method.

Reference:
    Choi, J., & Chen, R. (2022). Improved iterative methods for solving
    risk parity portfolio. Journal of Derivatives and Quantitative Studies,
    30(2). https://doi.org/10.1108/JDQS-12-2021-0031
"""

from ._core import RiskParityResult, risk_contributions, risk_parity

__all__ = ["risk_parity", "risk_contributions", "RiskParityResult"]
