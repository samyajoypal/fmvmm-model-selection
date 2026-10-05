from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from fmvmm.inference.adapters import as_adapter
from fmvmm.inference.results import LRTResult


NONREGULAR_WARNING = (
    "Mixture-model LRTs can be non-regular for hypotheses involving component "
    "number, zero mixing weights, boundary parameters, or non-identifiability. "
    "Use parametric bootstrap when in doubt."
)


def lrt(full_model, null_model, *, df=None, candidate_full="best",
        candidate_null="best", reference="chi2") -> LRTResult:
    """
    Likelihood-ratio test for nested fitted models.

    This function computes the statistic generically. The user remains
    responsible for ensuring nesting and for choosing a valid reference law.
    """
    full_adapter = as_adapter(full_model, candidate=candidate_full)
    null_adapter = as_adapter(null_model, candidate=candidate_null)

    ll_full = full_adapter.log_likelihood()
    ll_null = null_adapter.log_likelihood()
    stat = 2.0 * (ll_full - ll_null)

    if df is None:
        try:
            p_full = full_adapter.parameter_vector(parameterization="internal").theta.size
            p_null = null_adapter.parameter_vector(parameterization="internal").theta.size
            df = int(p_full - p_null)
        except Exception:
            df = None

    if df is None or reference != "chi2":
        return LRTResult(
            statistic=float(stat),
            df=-1 if df is None else int(df),
            pvalue=float("nan"),
            ll_full=float(ll_full),
            ll_null=float(ll_null),
            reference=reference,
            warning=NONREGULAR_WARNING,
        )

    return LRTResult(
        statistic=float(stat),
        df=int(df),
        pvalue=float(chi2.sf(max(stat, 0.0), df)),
        ll_full=float(ll_full),
        ll_null=float(ll_null),
        warning=NONREGULAR_WARNING,
    )
