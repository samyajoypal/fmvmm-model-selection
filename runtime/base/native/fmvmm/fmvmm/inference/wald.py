from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from fmvmm.inference.adapters import as_adapter
from fmvmm.inference.constraints import LinearConstraint, NonlinearConstraint
from fmvmm.inference.parameters import ParameterVector, covariance_from_info
from fmvmm.inference.results import WaldTestResult


def _constraint_matrices(constraint, pv: ParameterVector):
    if isinstance(constraint, (LinearConstraint, NonlinearConstraint)):
        h = constraint.evaluate(pv.theta)
        J = constraint.jacobian(pv.theta)
        df = constraint.df
        return h, J, df
    raise TypeError("constraint must be a LinearConstraint or NonlinearConstraint.")


def wald_test(model_or_adapter, constraint, *, parameterization="internal",
              info_method="auto", candidate="best", use_pinv=True) -> WaldTestResult:
    """
    Generic Wald test for linear or nonlinear constraints.

    The constraint must be expressed on the same parameterization requested here.
    For FMVMM, ``parameterization='user'`` exposes all pi values followed by
    component parameters; ``parameterization='internal'`` uses eta logits.
    """
    adapter = model_or_adapter
    if not hasattr(adapter, "parameter_vector"):
        adapter = as_adapter(model_or_adapter, candidate=candidate)

    pv = adapter.parameter_vector(parameterization=parameterization, info_method=info_method)
    cov = pv.covariance
    if cov is None:
        if pv.information is None:
            raise ValueError("No covariance or information matrix available.")
        cov = covariance_from_info(pv.information, use_pinv=use_pinv)

    h, J, df = _constraint_matrices(constraint, pv)
    mid = J @ cov @ J.T
    mid_inv = covariance_from_info(mid, use_pinv=use_pinv)
    stat = float(h.T @ mid_inv @ h)
    return WaldTestResult(
        statistic=stat,
        df=int(df),
        pvalue=float(chi2.sf(stat, df)),
        warning=None,
    )
