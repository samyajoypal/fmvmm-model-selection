r"""fmvmm.inference.inference_dmm

Inference utilities for Dirichlet Mixture Models (DMMs).

This module is intentionally lightweight and model-agnostic: it assumes you
have already fitted one (or two) DMM objects from ``fmvmm.mixtures``.

Key convention (matches your updated ``fmvmm.utils.utils_dmm``)
-------------------------------------------------------------
All information matrices and score vectors are computed on the *unconstrained*
parameterization

    \tilde\theta = (\eta_1,...,\eta_{K-1}, \alpha_{11},...,\alpha_{Kp}),

where \eta is the ALR transform of \pi with reference \pi_K (the last component):

    \eta_j = log(\pi_j / \pi_K),  j=1,...,K-1.

This avoids singularity from the simplex constraint \sum_j \pi_j = 1.

What this module provides (now)
-------------------------------
- Wald tests for linear hypotheses in unconstrained coordinates.
- Score tests for hypotheses that *fix* a subset of parameters under the null,
  using the fitted-null model's score and info.
- Likelihood Ratio Tests (LRT) for nested models (regular cases).

Important limitations
---------------------
Mixture models are non-regular for many hypotheses (e.g., testing a component
weight \pi_j=0 or testing k vs k-1). In those cases, standard chi-square
reference distributions can be wrong. This module exposes helpers to compute the
*test statistic* cleanly, while leaving you free to choose the reference law
(parametric bootstrap, chi-bar-square, etc.).

Author: (your extension based on Samyajoy Pal's fmvmm framework)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import chi2

from fmvmm.utils import utils_dmm


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_EPS = 1e-15


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _check_fitted_dmm(model: Any) -> None:
    """Validate that the object looks like a fitted fmvmm DMM."""
    required = [
        "pi_new",
        "alpha_new",
        "gamma_temp_ar",
        "log_likelihood_new",
        "data",
        "get_info_mat",
    ]
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(
            "Model does not appear to be a fitted DMM object. Missing attributes: "
            + ", ".join(missing)
        )


def theta_tilde_from_model(model: Any) -> np.ndarray:
    """Build unconstrained parameter vector (eta, alpha_flat) from a fitted model."""
    _check_fitted_dmm(model)
    pi = _as_array(model.pi_new)
    pi = np.clip(pi, _EPS, 1.0)
    pi = pi / pi.sum()
    alpha = _as_array(model.alpha_new)
    if alpha.ndim != 2:
        raise ValueError("alpha_new must be 2D with shape (K, p)")

    eta = utils_dmm.alr_transform(pi)  # (K-1,)
    return np.concatenate([eta, alpha.reshape(-1)])


def dims_from_model(model: Any) -> Tuple[int, int]:
    """Return (K, p) from a fitted DMM."""
    _check_fitted_dmm(model)
    pi = _as_array(model.pi_new)
    alpha = _as_array(model.alpha_new)
    if alpha.ndim != 2:
        raise ValueError("alpha_new must be 2D with shape (K, p)")
    K = len(pi)
    p = alpha.shape[1]
    if alpha.shape[0] != K:
        raise ValueError("alpha_new shape inconsistent with pi_new")
    return K, p


def idx_eta(K: int) -> np.ndarray:
    """Indices of eta block in theta_tilde."""
    return np.arange(0, K - 1, dtype=int)


def idx_alpha(K: int, p: int, j: int, m: Optional[int] = None) -> Union[np.ndarray, int]:
    """Indices for alpha parameters in theta_tilde.

    Parameters
    ----------
    K : int
        Number of mixture components.
    p : int
        Dimension of composition.
    j : int
        Component index (0-based).
    m : Optional[int]
        If None: returns indices for the entire alpha_j block (length p).
        Else: returns the scalar index for alpha_{j,m}.
    """
    if not (0 <= j < K):
        raise ValueError("j out of range")
    base = (K - 1) + j * p
    if m is None:
        return np.arange(base, base + p, dtype=int)
    if not (0 <= m < p):
        raise ValueError("m out of range")
    return int(base + m)


def cov_from_info(info: np.ndarray) -> np.ndarray:
    """Robust inverse of an information matrix."""
    try:
        return np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(info)


# -----------------------------------------------------------------------------
# Wald tests
# -----------------------------------------------------------------------------

@dataclass
class WaldTestResult:
    stat: float
    df: int
    pvalue: float


def wald_test(theta_hat: np.ndarray, cov_hat: np.ndarray, R: np.ndarray, r: np.ndarray) -> WaldTestResult:
    """General Wald test for linear constraints R theta = r.

    Parameters
    ----------
    theta_hat : (q,) array
        Unconstrained estimate.
    cov_hat : (q,q) array
        Covariance of theta_hat.
    R : (h,q) array
        Constraint matrix.
    r : (h,) array
        Constraint vector.

    Returns
    -------
    WaldTestResult
        stat = (R theta_hat - r)^T (R Cov R^T)^{-1} (R theta_hat - r)
        df = h
        pvalue = 1 - chi2_cdf(stat, h)
    """
    theta_hat = _as_array(theta_hat)
    cov_hat = _as_array(cov_hat)
    R = _as_array(R)
    r = _as_array(r)

    diff = R @ theta_hat - r
    mid = R @ cov_hat @ R.T
    mid_inv = cov_from_info(mid)
    stat = float(diff.T @ mid_inv @ diff)
    df = int(R.shape[0])
    pvalue = float(chi2.sf(stat, df))
    return WaldTestResult(stat=stat, df=df, pvalue=pvalue)


def wald_test_alpha_equal(model: Any, j: int, m: int, value: float, info_method: str = "louis") -> WaldTestResult:
    """Convenience Wald test: H0: alpha_{j,m} = value (on raw scale).

    Notes
    -----
    This test uses the delta method implicitly via the covariance of theta_tilde.
    Since alpha lives directly in theta_tilde (no transform), the constraint is
    linear: pick the corresponding coordinate.
    """
    _check_fitted_dmm(model)
    K, p = dims_from_model(model)
    I, _ = model.get_info_mat(method=info_method)
    cov = cov_from_info(I)
    theta = theta_tilde_from_model(model)

    q = len(theta)
    idx = idx_alpha(K, p, j, m)
    R = np.zeros((1, q))
    R[0, idx] = 1.0
    r = np.array([value], dtype=float)
    return wald_test(theta, cov, R, r)


def wald_test_eta_equal(model: Any, j: int, value: float, info_method: str = "louis") -> WaldTestResult:
    """Convenience Wald test: H0: eta_j = value, eta_j = log(pi_j/pi_K).

    Useful for hypotheses on mixture proportions once mapped to ALR space.
    """
    _check_fitted_dmm(model)
    K, _p = dims_from_model(model)
    if not (0 <= j < K - 1):
        raise ValueError("j must be in {0,...,K-2} for eta indices")

    I, _ = model.get_info_mat(method=info_method)
    cov = cov_from_info(I)
    theta = theta_tilde_from_model(model)

    q = len(theta)
    R = np.zeros((1, q))
    R[0, j] = 1.0
    r = np.array([value], dtype=float)
    return wald_test(theta, cov, R, r)


# -----------------------------------------------------------------------------
# Score tests (fixed-parameter version)
# -----------------------------------------------------------------------------

@dataclass
class ScoreTestResult:
    stat: float
    df: int
    pvalue: float


def score_test_fixed(model_null: Any, test_indices: Sequence[int], mode: Optional[str] = None) -> ScoreTestResult:
    """Score (LM) test for parameters fixed under the null.

    This implements the classical score test for *fixed coordinates* in
    theta_tilde.

    Workflow:
      1) Fit the null model (with the parameters fixed).
      2) Compute the score vector U(theta0) at the null.
      3) Compute the observed information I(theta0) at the null.
      4) Take the tested subvector U_q and block I_qq, and form

            LM = U_q^T I_qq^{-1} U_q  ~  chi^2_{q}

    Parameters
    ----------
    model_null : fitted DMM under H0
    test_indices : sequence of int
        Indices of parameters (in theta_tilde) that are fixed under H0 and
        being tested.
    mode : {'soft','hard', None}
        If None, inferred from model_null.EM_type ('Soft' -> soft, 'Hard' -> hard)

    Returns
    -------
    ScoreTestResult

    Notes
    -----
    - This uses the *per-observation* score implemented in utils_dmm for
      theta_tilde = (eta, alpha).
    - For non-regular hypotheses in mixtures, the chi-square reference can fail.
    """
    _check_fitted_dmm(model_null)

    if mode is None:
        mode = "soft" if getattr(model_null, "EM_type", "Soft") == "Soft" else "hard"

    pi = _as_array(model_null.pi_new)
    alpha = _as_array(model_null.alpha_new)
    gamma = _as_array(model_null.gamma_temp_ar)
    x = _as_array(model_null.data)

    # Score summed across observations
    if mode == "hard":
        gamma_use = utils_dmm.hard_assignments(gamma)
    else:
        gamma_use = gamma

    N = x.shape[0]
    # Sum score vectors
    U = np.zeros(((len(pi) - 1) + alpha.size,), dtype=float)
    for i in range(N):
        U += utils_dmm.score_vector_observation(pi, alpha, x[i], gamma_use[i])

    # Information at the null (Louis is recommended; score outer product is also possible)
    I0, _ = model_null.get_info_mat(method="louis")

    test_idx = np.asarray(list(test_indices), dtype=int)
    Uq = U[test_idx]
    Iqq = I0[np.ix_(test_idx, test_idx)]

    Iqq_inv = cov_from_info(Iqq)
    stat = float(Uq.T @ Iqq_inv @ Uq)
    df = int(len(test_idx))
    pvalue = float(chi2.sf(stat, df))
    return ScoreTestResult(stat=stat, df=df, pvalue=pvalue)


# -----------------------------------------------------------------------------
# Likelihood Ratio Tests
# -----------------------------------------------------------------------------

@dataclass
class LRTResult:
    stat: float
    df: int
    pvalue: float
    ll_full: float
    ll_null: float


def lrt(full_model: Any, null_model: Any, df: Optional[int] = None) -> LRTResult:
    """Likelihood Ratio Test between two fitted models.

    Parameters
    ----------
    full_model : fitted model under H1
    null_model : fitted model under H0 (nested in full)
    df : Optional[int]
        Degrees of freedom for chi-square reference. If None, tries to infer
        from ``total_parameters`` attributes if present.

    Returns
    -------
    LRTResult

    Notes
    -----
    For many mixture-model hypotheses (e.g., k vs k-1, pi_j=0), the asymptotic
    chi-square reference is not valid. This helper still returns the statistic
    2(ll_full-ll_null) and a naive p-value if df is provided/inferred.
    """
    _check_fitted_dmm(full_model)
    _check_fitted_dmm(null_model)

    ll_full = float(full_model.log_likelihood_new)
    ll_null = float(null_model.log_likelihood_new)
    stat = 2.0 * (ll_full - ll_null)

    if df is None:
        p_full = getattr(full_model, "total_parameters", None)
        p_null = getattr(null_model, "total_parameters", None)
        if (p_full is not None) and (p_null is not None):
            df = int(p_full - p_null)

    if df is None:
        # still return without p-value meaning; set pvalue nan
        return LRTResult(stat=float(stat), df=-1, pvalue=float("nan"), ll_full=ll_full, ll_null=ll_null)

    pvalue = float(chi2.sf(stat, df))
    return LRTResult(stat=float(stat), df=int(df), pvalue=pvalue, ll_full=ll_full, ll_null=ll_null)


# -----------------------------------------------------------------------------
# Public convenience: index builders for users
# -----------------------------------------------------------------------------

def build_test_indices_alpha(model: Any, j: int, m_list: Optional[Sequence[int]] = None) -> List[int]:
    """Return indices (in theta_tilde) for alpha_{j,m}.

    If m_list is None, returns the full block for component j.
    """
    K, p = dims_from_model(model)
    if m_list is None:
        return idx_alpha(K, p, j).tolist()
    return [idx_alpha(K, p, j, m) for m in m_list]


def build_test_indices_eta(model: Any, j_list: Optional[Sequence[int]] = None) -> List[int]:
    """Return indices (in theta_tilde) for eta_j.

    If j_list is None, returns all eta indices.
    """
    K, _p = dims_from_model(model)
    if j_list is None:
        return idx_eta(K).tolist()
    for j in j_list:
        if not (0 <= j < K - 1):
            raise ValueError("eta indices must be in {0,...,K-2}")
    return list(map(int, j_list))
