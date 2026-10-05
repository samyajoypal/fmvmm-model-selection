from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from fmvmm.inference.adapters import as_adapter
from fmvmm.inference.parameters import covariance_from_info
from fmvmm.inference.results import ScoreTestResult


def score_test(model_null, tested, *, parameterization="internal",
               info_method="auto", candidate="best", nuisance=None,
               use_pinv=True) -> ScoreTestResult:
    """
    Generic efficient score test for fixed coordinates under a fitted null.

    Parameters
    ----------
    model_null : fitted null model
        Must expose a score matrix through its adapter. FMVMM does; other model
        classes can opt in later by exposing score details.
    tested : sequence[int] or sequence[str]
        Coordinates tested under the null. Names are resolved through the
        parameter vector.
    nuisance : optional sequence[int] or sequence[str]
        Nuisance coordinates. If omitted, all non-tested coordinates are used.
    """
    adapter = as_adapter(model_null, candidate=candidate)
    pv = adapter.parameter_vector(parameterization=parameterization, info_method=info_method)
    U = adapter.score(parameterization=parameterization, method=info_method)
    I = pv.information
    if I is None:
        raise ValueError("Information matrix is required for score_test.")

    def _resolve(indices):
        out = []
        for item in indices:
            out.append(pv.index(item) if isinstance(item, str) else int(item))
        return np.asarray(out, dtype=int)

    q_idx = _resolve(tested)
    if nuisance is None:
        mask = np.ones(pv.theta.size, dtype=bool)
        mask[q_idx] = False
        n_idx = np.where(mask)[0]
    else:
        n_idx = _resolve(nuisance)

    Uq = U[q_idx]
    Un = U[n_idx]
    Iqq = I[np.ix_(q_idx, q_idx)]
    Iqn = I[np.ix_(q_idx, n_idx)]
    Inq = I[np.ix_(n_idx, q_idx)]
    Inn = I[np.ix_(n_idx, n_idx)]

    Inn_inv = covariance_from_info(Inn, use_pinv=use_pinv)
    efficient_score = Uq - Iqn @ Inn_inv @ Un
    efficient_info = Iqq - Iqn @ Inn_inv @ Inq
    efficient_info_inv = covariance_from_info(efficient_info, use_pinv=use_pinv)

    stat = float(efficient_score.T @ efficient_info_inv @ efficient_score)
    df = int(q_idx.size)
    return ScoreTestResult(
        statistic=stat,
        df=df,
        pvalue=float(chi2.sf(max(stat, 0.0), df)),
    )
