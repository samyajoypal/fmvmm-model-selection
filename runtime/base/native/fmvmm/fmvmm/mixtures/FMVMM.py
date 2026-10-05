from scipy.optimize import approx_fprime, minimize
import numpy as np
import os
import copy
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from fmvmm.utils.utils_fmm import (fmm_kmeans_init,fmm_gmm_init, fmm_loglikelihood, fmm_responsibilities,
                             fmm_pi_estimate, fmm_estimate_alphas, fmm_aic, fmm_bic, fmm_icl)


from fmvmm.utils.utils_mixture import (mixture_clusters)
import math

import itertools

from fmvmm.distributions import multivariate_skewnorm as mvsn
from fmvmm.distributions import multivariate_genhyperbolic as mghp
from fmvmm.distributions import multivariate_genskewt as mgst
from fmvmm.distributions import multivariate_hyperbolic as mvhb
from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.distributions import multivariate_norminvgauss as mnig
from fmvmm.distributions import multivariate_t as mvt
from fmvmm.distributions import multivariate_vargamma as mvvg
from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.distributions import multivariate_skew_t_smsn as mvst
from fmvmm.distributions import multivariate_skewnorm_cont as msnc
from fmvmm.distributions import multivariate_skewslash as mssl
from fmvmm.distributions import multivariate_slash as msl
import traceback
from scipy.special import logsumexp
from fmvmm.mixtures._base import BaseMixture
from fmvmm.utils.utils_dmm import hard_assignments, mixture_counts, mixture_proportions_info
from fmvmm.mixtures.mixmgh import approx_hessian_scipy
from fmvmm.mixtures.skewnormmix_smsn import (
    estimate_alphas_skewnormal, dmvSN, d_mixedmvSN
)
from fmvmm.mixtures.skewtmix_smsn import estimate_alphas_skewt
from fmvmm.mixtures.tmix_smsn import estimate_alphas_t
from fmvmm.mixtures.skewcontmix_smsn import (
    estimate_alphas_skewcn, dmvSNC, d_mixedmvSNC
)
from fmvmm.mixtures.skewslashmix_smsn import (
    estimate_alphas_skewslash,
    dmvSS as dmvSS_skewslash,
    d_mixedmvSS as d_mixedmvSS_skewslash,
)
from fmvmm.mixtures.slashmix_smsn import (
    estimate_alphas_slash,
    dmvSS as dmvSS_slash,
    d_mixedmvSS as d_mixedmvSS_slash,
)
from fmvmm.mixtures.skewlaplacemix import estimate_alphas_skewlaplace
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST
from fmvmm.utils.utils_dist import (
    pack_mvn_unconstrained,
    pack_smsn_unconstrained,
    unpack_mvn_unconstrained,
    unpack_smsn_unconstrained,
)
import warnings
warnings.filterwarnings('ignore')

dist_map = {"mvsn": mvsn, "mghp": mghp, "mgst": mgst, "mvhb": mvhb,
            "mvn": mvn, "mnig": mnig, "mvt": mvt, "mvvg": mvvg, "mvsl": mvsl,
            "mvst": mvst, "msnc": msnc, "mssl": mssl, "msl": msl}

all_dist = ["mvsn", "mghp", "mgst", "mvhb", "mvn", "mnig", "mvt", "mvvg",
            "mvsl", "mvst", "msnc","mssl","msl"]

soft_dist = ["mvsn", "mghp", "mgst", "mvhb", "mvn", "mnig", "mvt", "mvvg",
             "mvsl", "mvst", "msnc", "mssl", "msl"]
dist_name_by_module = {module: name for name, module in dist_map.items()}


def _regularize_cov(cov, eps=1e-6):
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = np.min(eigvals)
        if min_eig < eps:
            cov = cov + (eps - min_eig) * np.eye(cov.shape[0])
    except np.linalg.LinAlgError:
        cov = cov + eps * np.eye(cov.shape[0])
    return cov


def _weighted_mvn_update(X, weights, alpha_prev):
    weights = np.asarray(weights, dtype=float)
    n_eff = np.sum(weights)
    if n_eff <= 1e-12:
        return alpha_prev

    mu = np.sum(weights[:, None] * X, axis=0) / n_eff
    diff = X - mu
    cov = (diff.T @ (weights[:, None] * diff)) / n_eff
    return mu, _regularize_cov(cov)


def _as_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return float(arr.reshape(-1)[0])


def _gh_alpha_bar_from_chi_psi(chi, psi):
    return float(np.sqrt(max(_as_scalar(chi) * _as_scalar(psi), 0.0)))


def _soft_update_gh_family(dist_name, X, weights, alpha_prev, gh_mstep_kwargs):
    if dist_name == "mghp":
        lmbda, chi, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=_as_scalar(lmbda),
            alpha_bar=_gh_alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": True, "alpha_bar": True,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        if fit["alpha_bar"] != 0:
            chi_new, psi_new = mghp._alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
        elif fit["lmbda"] > 0:
            chi_new, psi_new = 0, 2 * fit["lmbda"]
        else:
            chi_new, psi_new = -2 * (fit["lmbda"] + 1), 0
        return fit["lmbda"], chi_new, psi_new, fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mgst":
        lmbda, chi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=_as_scalar(lmbda),
            alpha_bar=0,
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": True, "alpha_bar": False,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        return fit["lmbda"], -2 * (fit["lmbda"] + 1), fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mvhb":
        chi, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=(X.shape[1] + 1) / 2.0,
            alpha_bar=_gh_alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": False, "alpha_bar": True,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        chi_new, psi_new = mghp._alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
        return chi_new, psi_new, fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mnig":
        chi, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=-0.5,
            alpha_bar=_gh_alpha_bar_from_chi_psi(chi, psi),
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": False, "alpha_bar": True,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        return fit["alpha_bar"], fit["alpha_bar"], fit["mu"], fit["sigma"], fit["gamma"]

    if dist_name == "mvvg":
        lmbda, psi, mu, sigma, gamma = alpha_prev
        fit = mghp.fitghypmv(
            X, weights=weights,
            lmbda=_as_scalar(lmbda),
            alpha_bar=0,
            mu=mu, sigma=sigma, gamma=gamma,
            opt_pars={"lmbda": True, "alpha_bar": False,
                      "mu": True, "sigma": True, "gamma": True},
            **gh_mstep_kwargs,
        )
        return fit["lmbda"], 2 * fit["lmbda"], fit["mu"], fit["sigma"], fit["gamma"]

    raise NotImplementedError(f"GH-family soft M-step is not available for {dist_name}.")


def _smsn_unconstrained_spec(dist_name, alpha):
    if dist_name in {"mvsn", "mvst", "mssl", "msnc"}:
        mu, sigma, shape, *tail_parts = alpha
        tail = np.concatenate([
            np.asarray(part, dtype=float).reshape(-1) for part in tail_parts
        ]) if tail_parts else np.array([], dtype=float)
        transforms = {
            "mvsn": (),
            "mvst": ("positive",),
            "mssl": ("positive",),
            "msnc": ("unit", "unit"),
        }[dist_name]
        has_shape = True
    elif dist_name in {"mvt", "msl"}:
        mu, sigma, tail_value = alpha
        shape = None
        tail = np.asarray(tail_value, dtype=float).reshape(-1)
        transforms = ("positive",)
        has_shape = False
    else:
        raise NotImplementedError(f"No unconstrained SMSN specification for {dist_name}.")

    p = len(mu)
    u = pack_smsn_unconstrained(
        p=p,
        mu=mu,
        sigma=sigma,
        shape=shape,
        tail=tail,
        tail_transforms=transforms,
    )

    def unpack(value):
        return unpack_smsn_unconstrained(
            value,
            p=p,
            has_shape=has_shape,
            tail_transforms=transforms,
        )

    return u, unpack


def _soft_update_smsn_family(dist_name, X, weights, alpha_prev,
                             smsn_mstep_kwargs=None):
    """Numerically maximize the weighted component log-likelihood."""
    smsn_mstep_kwargs = {} if smsn_mstep_kwargs is None else dict(smsn_mstep_kwargs)
    module = dist_map[dist_name]
    u0, unpack = _smsn_unconstrained_spec(dist_name, alpha_prev)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape != (len(X),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Component weights must be finite and nonnegative.")
    positive = weights > 0
    weighted_X, positive_weights = X[positive], weights[positive]

    def objective(u):
        try:
            log_density = np.asarray(module.logpdf(weighted_X, *unpack(u)), dtype=float).reshape(-1)
        except Exception:
            return 1e100
        if log_density.size != weighted_X.shape[0] or not np.all(np.isfinite(log_density)):
            return 1e100
        value = -float(np.dot(positive_weights, log_density))
        return value if np.isfinite(value) else 1e100

    q0 = objective(u0)
    options = {
        "maxiter": int(smsn_mstep_kwargs.pop("maxiter", 80)),
        "maxfun": int(smsn_mstep_kwargs.pop("maxfun", 1000)),
        "ftol": float(smsn_mstep_kwargs.pop("ftol", 1e-9)),
        "gtol": float(smsn_mstep_kwargs.pop("gtol", 1e-5)),
        "maxls": int(smsn_mstep_kwargs.pop("maxls", 30)),
    }
    options.update(smsn_mstep_kwargs.pop("options", {}))
    result = minimize(
        objective,
        u0,
        method=smsn_mstep_kwargs.pop("method", "L-BFGS-B"),
        options=options,
        **smsn_mstep_kwargs,
    )
    if np.isfinite(result.fun) and result.fun <= q0 + 1e-8:
        return unpack(result.x)
    return alpha_prev


def _weighted_component_loglik(dist_name, X, weights, alpha):
    """Weighted observed component log-likelihood for GEM acceptance."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape != (len(X),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Component weights must be finite and nonnegative.")
    positive = weights > 0
    try:
        log_density = np.asarray(
            dist_map[dist_name].logpdf(X[positive], *alpha), dtype=float
        ).reshape(-1)
    except Exception:
        return -np.inf
    if log_density.size != np.sum(positive) or not np.all(np.isfinite(log_density)):
        return -np.inf
    value = float(np.dot(weights[positive], log_density))
    return value if np.isfinite(value) else -np.inf


def _gem_accept_component_update(dist_name, X, weights, alpha_prev, alpha_candidate,
                                 accept_tol=1e-8):
    """
    Accept a numerical M-step update only when it satisfies the GEM condition.

    The candidate may be non-converged because an inner optimizer hit maxiter; it is
    still a valid generalized EM update if the weighted component log-likelihood
    does not decrease.
    """
    old_value = _weighted_component_loglik(dist_name, X, weights, alpha_prev)
    new_value = _weighted_component_loglik(dist_name, X, weights, alpha_candidate)
    if np.isfinite(new_value) and (
        not np.isfinite(old_value) or new_value >= old_value - accept_tol
    ):
        return alpha_candidate
    return alpha_prev


def _soft_update_one_component(dist_name, X, weights, alpha_prev, gh_mstep_kwargs=None,
                               smsn_mstep_kwargs=None, mstep_accept_tol=1e-8):
    gamma = np.asarray(weights, dtype=float).reshape(-1, 1)
    if np.sum(gamma) <= 1e-12:
        return alpha_prev

    gh_mstep_kwargs = {} if gh_mstep_kwargs is None else gh_mstep_kwargs
    pi_prev = np.array([1.0])
    alpha_list = [alpha_prev]
    p = X.shape[1]

    if dist_name in {"mghp", "mgst", "mvhb", "mnig", "mvvg"}:
        alpha_candidate = _soft_update_gh_family(
            dist_name, X, gamma[:, 0], alpha_prev, gh_mstep_kwargs
        )
        return _gem_accept_component_update(
            dist_name, X, gamma[:, 0], alpha_prev, alpha_candidate, mstep_accept_tol
        )

    if dist_name == "mvn":
        return _weighted_mvn_update(X, gamma[:, 0], alpha_prev)

    if dist_name in {"mvsn", "mvst", "msnc", "mssl", "mvt", "msl"}:
        alpha_candidate = _soft_update_smsn_family(
            dist_name,
            X,
            gamma[:, 0],
            alpha_prev,
            smsn_mstep_kwargs=smsn_mstep_kwargs,
        )
        return _gem_accept_component_update(
            dist_name, X, gamma[:, 0], alpha_prev, alpha_candidate, mstep_accept_tol
        )

    if dist_name == "mvsn":
        return estimate_alphas_skewnormal(
            X, gamma, alpha_list, pi_prev,
            dmvSN_func=dmvSN,
            d_mixedmvSN_func=d_mixedmvSN,
        )[0]

    if dist_name == "mvst":
        mu, sigma, lmbda, nu = alpha_prev
        alpha_st = [(mu, sigma, lmbda, _as_scalar(nu))]
        mu_new, sigma_new, lmbda_new, nu_new = estimate_alphas_skewt(
            X, gamma, alpha_st, pi_prev
        )[0]
        return mu_new, sigma_new, lmbda_new, np.array([nu_new])

    if dist_name == "msnc":
        return estimate_alphas_skewcn(
            X, gamma, alpha_list, pi_prev,
            dmvSNC_func=dmvSNC,
            d_mixedmvSNC_func=d_mixedmvSNC,
        )[0]

    if dist_name == "mssl":
        mu, sigma, lmbda, nu = alpha_prev
        alpha_list = [(mu, sigma, lmbda, _as_scalar(nu))]
        mu_new, sigma_new, lmbda_new, nu_new = estimate_alphas_skewslash(
            X, gamma, alpha_list, pi_prev,
            dmvSS_func=dmvSS_skewslash,
            d_mixedmvSS_func=d_mixedmvSS_skewslash,
        )[0]
        return mu_new, sigma_new, lmbda_new, np.array([nu_new])

    if dist_name == "mvsl":
        alpha_candidate = estimate_alphas_skewlaplace(X, gamma, alpha_list)[0]
        return _gem_accept_component_update(
            dist_name, X, gamma[:, 0], alpha_prev, alpha_candidate, mstep_accept_tol
        )

    if dist_name == "mvt":
        loc, shape, df = alpha_prev
        alpha_t = [(loc, shape, np.zeros(p), _as_scalar(df))]
        mu_new, sigma_new, _, nu_new = estimate_alphas_t(
            X, gamma, alpha_t, pi_prev,
            dmvt_ls_func=dmvt_ls,
            d_mixedmvST_func=d_mixedmvST,
        )[0]
        return mu_new, sigma_new, np.array([nu_new])

    if dist_name == "msl":
        mu, sigma, nu = alpha_prev
        alpha_slash = [(mu, sigma, np.zeros(p), _as_scalar(nu))]
        mu_new, sigma_new, _, nu_new = estimate_alphas_slash(
            X, gamma, alpha_slash, pi_prev,
            dmvSS_func=dmvSS_slash,
            d_mixedmvSS_func=d_mixedmvSS_slash,
        )[0]
        return mu_new, sigma_new, np.array([nu_new])

    raise NotImplementedError(f"Soft M-step is not available for {dist_name}.")


def fmm_estimate_alphas_soft(X, gamma_matrix, alpha_prev, soft_dist_comb=None, **kwargs):
    if soft_dist_comb is None:
        raise ValueError("soft_dist_comb must be provided for non-identical soft EM.")
    gh_mstep_kwargs = kwargs.get("gh_mstep_kwargs", {})
    smsn_mstep_kwargs = kwargs.get("smsn_mstep_kwargs", {})
    mstep_accept_tol = kwargs.get("mstep_accept_tol", 1e-8)
    alpha_new = []
    for j, dist_module in enumerate(soft_dist_comb):
        dist_name = dist_name_by_module.get(dist_module)
        if dist_name not in soft_dist:
            raise NotImplementedError(f"Soft M-step is not available for {dist_name}.")
        try:
            alpha_new.append(
                _soft_update_one_component(
                    dist_name, X, gamma_matrix[:, j], alpha_prev[j],
                    gh_mstep_kwargs=gh_mstep_kwargs,
                    smsn_mstep_kwargs=smsn_mstep_kwargs,
                    mstep_accept_tol=mstep_accept_tol,
                )
            )
        except Exception:
            alpha_new.append(alpha_prev[j])
    return alpha_new

def convert_to_numpy(tuples_list):
    """
    Given a list of tuples, convert any integer or float inside the tuples
    into a NumPy array of shape (1,).
    """
    def convert_item(item):
        if isinstance(item, (int, float)):
            return np.array([item])
        elif isinstance(item, tuple):
            return tuple(convert_item(sub_item) for sub_item in item)
        elif isinstance(item, list):  # If lists are inside tuples, handle them too
            return [convert_item(sub_item) for sub_item in item]
        else:
            return item  # Keep other types unchanged

    return [tuple(convert_item(item) for item in tpl) for tpl in tuples_list]


# def compute_info_individual_fmvmm(dist_comp, params_init, X, epsilon=1e-6):
#     """
#     Use SciPy to approximate the Hessian of the negative log-likelihood
#     for the GH mixture, then invert to get the empirical info matrix.
#
#     Returns: I_e, Cov, SE
#     """
#     IM = dist_comp.info_mat(X, *params_init)
#
#     # Flatten final parameters
#     n, p = X.shape
#     param_shapes = [np.array(p).shape for p in params_init]
#     theta_hat = flatten_params(params_init)
#
#     d = len(theta_hat)
#
#
#
#     return IM, len(theta_hat)

def compute_info_individual_fmvmm(dist_comp, params_init, X, epsilon=1e-6,
                                  prefer_scores=True, ridge=1e-8):
    """
    Returns
    -------
    IM : (d,d)
    param_len : int
    se : (d,)  (optional, convenient)
    cov : (d,d)
    meta : dict (debug info)
    """
    X = np.asarray(X)
    n, p = X.shape

    # ---------- 1) Try score-based OPG if available ----------
    if prefer_scores and hasattr(dist_comp, "score_mat") and callable(getattr(dist_comp, "score_mat")):
        S = dist_comp.score_mat(X, *params_init)  # (n,d)
        S = np.asarray(S, float)
        if S.ndim != 2 or S.shape[0] != n:
            raise ValueError(f"score_mat must return shape (n,d). Got {S.shape}")
        S -= S.mean(axis=0, keepdims=True)   # << important

        IM = S.T @ S
        IM = 0.5 * (IM + IM.T) + ridge * np.eye(IM.shape[0])

        try:
            cov = np.linalg.inv(IM)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(IM)

        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return IM, IM.shape[0], se, cov, {"source": "scores_opg", "n": n, "p": p}

    # ---------- 2) Fallback: info_mat ----------
    IM = dist_comp.info_mat(X, *params_init)
    IM = np.asarray(IM, float)
    IM = 0.5 * (IM + IM.T) + ridge * np.eye(IM.shape[0])

    try:
        cov = np.linalg.inv(IM)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(IM)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return IM, IM.shape[0], se, cov, {"source": "info_mat", "n": n, "p": p}

def organize_data_by_clusters(data, cluster_predictions):
    # Get the unique cluster labels
    unique_clusters = np.unique(cluster_predictions)

    # Initialize a list to store data points for each cluster
    data_by_clusters = [np.empty((0, data.shape[1])) for _ in unique_clusters]

    # Iterate over each data point and its cluster prediction
    for point, cluster in zip(data, cluster_predictions):
        # Append the data point to the corresponding cluster
        data_by_clusters[cluster] = np.vstack((data_by_clusters[cluster], point))

    return data_by_clusters


def flatten_params(params):
    """
    Flatten parameters into a single vector.

    Parameters:
        params (tuple): The parameters.

    Returns:
        flat_params (ndarray): The flattened parameters.
    """
    return np.array(np.concatenate([p.flatten() if isinstance(p, np.ndarray) else np.array([p]) for p in params]), dtype=float)


def reshape_params(flat_params, param_shapes):
    """
    Reshape a flattened parameter vector into original shapes.

    Parameters:
        flat_params (ndarray): The flattened parameters.
        param_shapes (list): The shapes of the original parameters.

    Returns:
        params (tuple): The reshaped parameters.
    """
    params = []
    start = 0
    for shape in param_shapes:
        end = start + np.prod(shape)
        params.append(np.reshape(flat_params[start:end], shape))
        start = end
    return tuple(params)

def ensure_positive_diagonal(sum_outer):
    """
    Ensure that the matrix has positive diagonal elements.

    Parameters:
        sum_outer (ndarray): The matrix. Shape (num_params, num_params).

    Returns:
        sum_outer_pos (ndarray): The matrix with positive diagonal elements.
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(sum_outer)

    # Adjust eigenvalues to ensure positivity
    eigenvalues[eigenvalues < 0] = 1e-6  # Set negative eigenvalues to a small positive value

    # Reconstruct the matrix
    sum_outer_pos = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.linalg.inv(eigenvectors)))

    return sum_outer_pos

# def project_to_psd(A, eps=1e-8):
#     A = 0.5 * (A + A.T)
#     w, V = np.linalg.eigh(A)
#     w = np.maximum(w, eps)
#     return (V * w) @ V.T

def project_to_psd(A, eps=1e-8, max_tries=3):
    """
    Robust PSD projection with ridge fallback.
    """

    A = 0.5 * (A + A.T)

    for _ in range(max_tries):
        try:
            w, V = np.linalg.eigh(A)
            w = np.maximum(w, eps)
            return (V * w) @ V.T
        except np.linalg.LinAlgError:
            # inflate diagonal and retry
            A = A + 10 * eps * np.eye(A.shape[0])

    # ultimate fallback: diagonal approximation
    diag = np.maximum(np.diag(A), eps)
    return np.diag(diag)

def pi_info_constrained(pi, N, eps=1e-12):
    """
    Constrained (simplex-aware) information for mixture weights
    using logits eta_j = log(pi_j/pi_k), j=1..k-1.

    Returns:
      I_eta      : (k-1)x(k-1) information in eta coords
      Cov_eta    : inverse (pseudo-inverse if needed)
      Cov_pi     : approximate covariance for pi (k x k) via delta method
    """
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    pi = np.clip(pi, eps, 1 - eps)
    pi = pi / pi.sum()

    p = pi[:k-1]                 # length k-1
    I_eta = N * (np.diag(p) - np.outer(p, p))  # (k-1)x(k-1)

    # invert in eta space
    I_eta = 0.5 * (I_eta + I_eta.T)
    try:
        Cov_eta = np.linalg.inv(I_eta)
    except np.linalg.LinAlgError:
        Cov_eta = np.linalg.pinv(I_eta)

    # Jacobian J = d pi / d eta  (k x (k-1))
    # For j<k: d pi_j / d eta_m = pi_j*(1{j=m} - pi_m)
    # For k:   d pi_k / d eta_m = -pi_k*pi_m
    J = np.zeros((k, k-1))
    for m in range(k-1):
        for j in range(k-1):
            J[j, m] = pi[j] * ((1.0 if j == m else 0.0) - pi[m])
        J[k-1, m] = -pi[k-1] * pi[m]

    Cov_pi = J @ Cov_eta @ J.T
    return I_eta, Cov_eta, Cov_pi


def pi_jacobian_from_eta(pi, eps=1e-12):
    """
    Jacobian d pi / d eta for eta_j = log(pi_j / pi_k), j=1,...,k-1.
    """
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    pi = np.clip(pi, eps, 1 - eps)
    pi = pi / pi.sum()

    J = np.zeros((k, k - 1))
    for m in range(k - 1):
        for j in range(k - 1):
            J[j, m] = pi[j] * ((1.0 if j == m else 0.0) - pi[m])
        J[k - 1, m] = -pi[k - 1] * pi[m]
    return J


def eta_score_matrix(tau, pi):
    """
    Per-observation scores for mixture logits eta_j = log(pi_j/pi_k).
    """
    tau = np.asarray(tau, dtype=float)
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    if tau.shape[1] != k:
        raise ValueError("tau and pi have incompatible numbers of components.")
    if k == 1:
        return np.zeros((tau.shape[0], 0))
    return tau[:, :k - 1] - pi[:k - 1]


def safe_component_score_matrix(dist_module, X, params):
    """
    Component score matrix with numerical sanitation.

    The returned scores are on the parameterization used by each distribution's
    score_mat implementation, typically unconstrained Cholesky/log parameters
    for GH/MVN and the package's existing SMSN score coordinates.
    """
    if not hasattr(dist_module, "score_mat") or not callable(getattr(dist_module, "score_mat")):
        raise NotImplementedError(
            f"{dist_module.__name__} does not provide score_mat; cannot compute FMVMM information."
        )

    S = dist_module.score_mat(X, *params)
    S = np.real_if_close(S, tol=1000)
    S = np.asarray(np.real(S), dtype=float)
    if S.ndim != 2 or S.shape[0] != X.shape[0]:
        raise ValueError(
            f"{dist_module.__name__}.score_mat must return shape (n,d); got {S.shape}."
        )

    return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)


def _component_unconstrained_spec(dist_module, params, p):
    """Return fitted internal coordinates and their inverse map."""
    dist_name = dist_name_by_module[dist_module]

    if dist_name == "mvn":
        mu, sigma = params
        u = pack_mvn_unconstrained(p=p, mu=mu, sigma=sigma)
        return u, lambda value: unpack_mvn_unconstrained(value, p=p)

    if dist_name == "mghp":
        lmbda, chi, psi, mu, sigma, gamma = params
        u = mghp.pack_gh_alpha_bar_unconstrained(
            p=p,
            lmbda=lmbda,
            alpha_bar=mghp.alpha_bar_from_chi_psi(chi, psi),
            mu=mu,
            sigma=sigma,
            gamma=gamma,
            free=("lmbda", "alpha_bar"),
        )
        return u, lambda value: mghp.unpack_gh_alpha_bar_unconstrained(
            value, p=p, fixed={}, free=("lmbda", "alpha_bar")
        )

    if dist_name in {"mnig", "mvhb"}:
        chi, psi, mu, sigma, gamma = params
        fixed_lambda = -0.5 if dist_name == "mnig" else (p + 1.0) / 2.0
        u = mghp.pack_gh_alpha_bar_unconstrained(
            p=p,
            lmbda=fixed_lambda,
            alpha_bar=mghp.alpha_bar_from_chi_psi(chi, psi),
            mu=mu,
            sigma=sigma,
            gamma=gamma,
            free=("alpha_bar",),
        )
        return u, lambda value: mghp.unpack_gh_alpha_bar_unconstrained(
            value,
            p=p,
            fixed={"lmbda": fixed_lambda},
            free=("alpha_bar",),
        )[1:]

    if dist_name == "mvvg":
        lmbda, _psi, mu, sigma, gamma = params
        u = mghp.pack_vg_unconstrained(
            p=p, lmbda=lmbda, mu=mu, sigma=sigma, gamma=gamma
        )

        def unpack_vg(value):
            lmbda_new, _chi, psi_new, mu_new, sigma_new, gamma_new = (
                mghp.unpack_vg_unconstrained(value, p=p)
            )
            return lmbda_new, psi_new, mu_new, sigma_new, gamma_new

        return u, unpack_vg

    if dist_name == "mgst":
        lmbda, _chi, mu, sigma, gamma = params
        u = mghp.pack_gst_unconstrained(
            p=p, lmbda=lmbda, mu=mu, sigma=sigma, gamma=gamma
        )

        def unpack_gst(value):
            lmbda_new, chi_new, _psi, mu_new, sigma_new, gamma_new = (
                mghp.unpack_gst_unconstrained(value, p=p)
            )
            return lmbda_new, chi_new, mu_new, sigma_new, gamma_new

        return u, unpack_gst

    if dist_name in {"mvsn", "mvst", "msnc", "mssl", "mvt", "msl"}:
        return _smsn_unconstrained_spec(dist_name, params)

    if dist_name == "mvsl":
        mu, sigma, gamma = params
        u = pack_smsn_unconstrained(
            p=p, mu=mu, sigma=sigma, shape=gamma
        )
        return u, lambda value: unpack_smsn_unconstrained(
            value, p=p, has_shape=True
        )

    raise NotImplementedError(
        f"No fitted internal parameterization for {dist_name}."
    )


def _negative_weighted_loglik_hessian(dist_module, X, params, weights,
                                       step=1e-4):
    """Numerical negative Hessian of a weighted component log-likelihood."""
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    p = X.shape[1]
    u0, unpack = _component_unconstrained_spec(dist_module, params, p)
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    d = u0.size
    h = float(step) * np.maximum(1.0, np.abs(u0))

    def loglike(value):
        density = np.asarray(
            dist_module.logpdf(X, *unpack(value)), dtype=float
        ).reshape(-1)
        if density.size != X.shape[0] or not np.all(np.isfinite(density)):
            raise FloatingPointError(
                f"Non-finite log density while differentiating {dist_module.__name__}."
            )
        return float(np.dot(weights, density))

    f0 = loglike(u0)
    hessian = np.zeros((d, d), dtype=float)
    for a in range(d):
        plus = u0.copy()
        minus = u0.copy()
        plus[a] += h[a]
        minus[a] -= h[a]
        hessian[a, a] = (loglike(plus) - 2.0 * f0 + loglike(minus)) / (h[a] ** 2)
        for b in range(a):
            pp = u0.copy()
            pm = u0.copy()
            mp = u0.copy()
            mm = u0.copy()
            pp[a] += h[a]; pp[b] += h[b]
            pm[a] += h[a]; pm[b] -= h[b]
            mp[a] -= h[a]; mp[b] += h[b]
            mm[a] -= h[a]; mm[b] -= h[b]
            value = (
                loglike(pp) - loglike(pm) - loglike(mp) + loglike(mm)
            ) / (4.0 * h[a] * h[b])
            hessian[a, b] = value
            hessian[b, a] = value
    return -0.5 * (hessian + hessian.T)


def complete_information_fmvmm(X, pi, alpha, dist_comb, tau,
                               hessian_step=1e-4):
    """Conditional expectation of negative complete-data Hessians."""
    X = np.asarray(X, dtype=float)
    pi = np.asarray(pi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n, k = tau.shape
    eta_dim = max(k - 1, 0)
    param_dims = [
        safe_component_score_matrix(module, X[:1], params).shape[1]
        for module, params in zip(dist_comb, alpha)
    ]
    total_dim = eta_dim + sum(param_dims)
    information = np.zeros((total_dim, total_dim), dtype=float)

    if eta_dim:
        p_eta = pi[:eta_dim]
        information[:eta_dim, :eta_dim] = n * (
            np.diag(p_eta) - np.outer(p_eta, p_eta)
        )

    cursor = eta_dim
    for component, (module, params, dim) in enumerate(
        zip(dist_comb, alpha, param_dims)
    ):
        block = _negative_weighted_loglik_hessian(
            module,
            X,
            params,
            tau[:, component],
            step=hessian_step,
        )
        if block.shape != (dim, dim):
            raise ValueError(
                f"Hessian/score dimension mismatch for {module.__name__}: "
                f"{block.shape} versus {(dim, dim)}."
            )
        information[cursor:cursor + dim, cursor:cursor + dim] = block
        cursor += dim
    return 0.5 * (information + information.T), param_dims


def observed_score_matrix_fmvmm(X, pi, alpha, dist_comb, tau):
    """
    Full observed-data score matrix for a non-identical finite mixture.

    Parameter order:
      eta_1,...,eta_{k-1}, theta_1, ..., theta_k
    where eta_j = log(pi_j/pi_k), and theta_j follows the score_mat
    parameterization of component j.
    """
    X = np.asarray(X, dtype=float)
    pi = np.asarray(pi, dtype=float)
    tau = np.asarray(tau, dtype=float)
    n = X.shape[0]
    k = len(dist_comb)

    if len(alpha) != k or pi.size != k or tau.shape != (n, k):
        raise ValueError("Incompatible X, pi, alpha, dist_comb, and tau dimensions.")

    eta_scores = eta_score_matrix(tau, pi)
    component_scores = []
    raw_component_scores = []
    param_dims = []

    for j, dist_module in enumerate(dist_comb):
        S_j = safe_component_score_matrix(dist_module, X, alpha[j])
        raw_component_scores.append(S_j)
        component_scores.append(tau[:, j:j + 1] * S_j)
        param_dims.append(S_j.shape[1])

    if component_scores:
        S_obs = np.hstack([eta_scores] + component_scores)
    else:
        S_obs = eta_scores

    return S_obs, raw_component_scores, param_dims


def louis_score_decomposition_fmvmm(X, pi, alpha, dist_comb, tau):
    """
    Exact Louis decomposition with numerically differentiated component Hessians.

    I_obs = E[-H_c | X] - Var[S_c | X]. The second term includes all
    membership-uncertainty cross-blocks.
    """
    S_obs, raw_component_scores, param_dims = observed_score_matrix_fmvmm(
        X, pi, alpha, dist_comb, tau
    )
    tau = np.asarray(tau, dtype=float)
    pi = np.asarray(pi, dtype=float)
    n, k = tau.shape
    eta_dim = max(k - 1, 0)
    total_dim = eta_dim + sum(param_dims)

    offsets = []
    cursor = eta_dim
    for d in param_dims:
        offsets.append(cursor)
        cursor += d

    complete_opg = np.zeros((total_dim, total_dim), dtype=float)

    for i in range(n):
        for h in range(k):
            sc = np.zeros(total_dim, dtype=float)
            if eta_dim:
                sc[:eta_dim] = -pi[:eta_dim]
                if h < eta_dim:
                    sc[h] += 1.0
            start = offsets[h]
            end = start + param_dims[h]
            sc[start:end] = raw_component_scores[h][i]
            complete_opg += tau[i, h] * np.outer(sc, sc)

    observed_opg = S_obs.T @ S_obs
    observed_opg = 0.5 * (observed_opg + observed_opg.T)
    complete_opg = 0.5 * (complete_opg + complete_opg.T)
    missing = complete_opg - observed_opg
    missing = 0.5 * (missing + missing.T)

    complete_information, hessian_dims = complete_information_fmvmm(
        X, pi, alpha, dist_comb, tau
    )
    if hessian_dims != param_dims:
        raise ValueError("Louis Hessian and score parameterizations do not match.")
    observed_information = complete_information - missing
    observed_information = 0.5 * (observed_information + observed_information.T)

    return observed_information, complete_information, missing, S_obs, param_dims


def classification_information_fmvmm(X, pi, alpha, dist_comb, labels):
    """
    Modified-Louis/classification information for hard EM.

    This treats component labels as known. The missing-information term from
    uncertain membership is therefore zero. The eta block is the multinomial
    complete-data information. Component blocks use numerical negative
    Hessians on hard-assigned observations.

    Parameter order is the same internal identifiable order used by the soft
    method:
      eta_1,...,eta_{k-1}, theta_1, ..., theta_k
    """
    X = np.asarray(X, dtype=float)
    pi = np.asarray(pi, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = X.shape[0]
    k = len(dist_comb)
    eta_dim = max(k - 1, 0)

    if labels.shape[0] != n:
        raise ValueError("labels must have one entry per observation.")

    tau_hard = np.zeros((n, k), dtype=float)
    tau_hard[np.arange(n), labels] = 1.0
    I, param_dims = complete_information_fmvmm(
        X, pi, alpha, dist_comb, tau_hard
    )
    S_complete, _, _ = observed_score_matrix_fmvmm(X, pi, alpha, dist_comb, tau_hard)
    missing = np.zeros_like(I)

    return I, I.copy(), missing, S_complete, param_dims


def invert_information_matrix(info, ridge=1e-8, use_pinv=True):
    info = np.asarray(info, dtype=float)
    info = 0.5 * (info + info.T)
    if ridge and ridge > 0:
        info = info + ridge * np.eye(info.shape[0])
    try:
        cov = np.linalg.pinv(info) if use_pinv else np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return info, cov, se


def pi_cov_from_eta_cov(pi, cov_eta):
    if len(pi) <= 1:
        return np.zeros((len(pi), len(pi)))
    J = pi_jacobian_from_eta(pi)
    return J @ cov_eta @ J.T


def user_scale_transform(pi, param_dims):
    """
    Linear delta-method map from internal [eta, theta] to user [pi, theta].
    """
    pi = np.asarray(pi, dtype=float)
    k = pi.size
    eta_dim = max(k - 1, 0)
    theta_dim = int(np.sum(param_dims))
    T = np.zeros((k + theta_dim, eta_dim + theta_dim), dtype=float)
    if eta_dim:
        T[:k, :eta_dim] = pi_jacobian_from_eta(pi)
    if theta_dim:
        T[k:, eta_dim:] = np.eye(theta_dim)
    return T


def transform_information_to_user_scale(pi, param_dims, cov_internal,
                                        ridge=1e-8, use_pinv=True):
    """
    Return covariance/SE for [all pi, theta] by delta method.

    The all-pi covariance is singular because sum(pi)=1. We therefore report a
    generalized information matrix as the Moore-Penrose inverse of the user-scale
    covariance. The SEs are still the natural marginal SEs for all pi values.
    """
    T = user_scale_transform(pi, param_dims)
    cov_user = T @ cov_internal @ T.T
    cov_user = 0.5 * (cov_user + cov_user.T)
    info_user = np.linalg.pinv(cov_user)
    info_user = 0.5 * (info_user + info_user.T)
    se_user = np.sqrt(np.maximum(np.diag(cov_user), 0.0))
    return info_user, cov_user, se_user, T


def ecdf_from_log_density(log_density):
    """
    Compute the empirical cumulative distribution function (ECDF) from log-density values.

    Parameters:
        log_density (numpy.ndarray): The array of log-density values.

    Returns:
        numpy.ndarray: The empirical cumulative distribution function values.
    """
    # Convert log-density to density by exponentiation
    density = np.exp(log_density)

    # Sort the density values in ascending order
    sorted_density = np.sort(density)

    # Compute the cumulative sum of the sorted density values
    cumulative_sum = np.cumsum(sorted_density)

    # Normalize the cumulative sum to get ECDF values
    ecdf_values = cumulative_sum / np.sum(density)

    return ecdf_values


def fmm_sorted_lpdf_cdf(pi_temp,alpha_temp,data_lol,list_of_dist):
    n = len(data_lol)
    k = len(alpha_temp)
    dist_variables = [dist_map[list_of_dist[j]]
                           for j in range(len(list_of_dist))]
    dist_comb = list(
        itertools.combinations(dist_variables, k))[0]
    log_likelihood_values_temp = []
    for c in range(n):
        try:
            log_likelihood_old_temp = math.log(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        except:
            log_likelihood_old_temp = math.log1p(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        log_likelihood_values_temp.append(log_likelihood_old_temp)

    sorted_lists = sorted(zip(log_likelihood_values_temp, data_lol),reverse=False)
    sorted_lpdf, sorted_data = zip(*sorted_lists)


    cdfs=ecdf_from_log_density(log_likelihood_values_temp)
    return sorted_lpdf,cdfs

def add_elements_at_indices(my_list, indices_to_add, new_elements):
    for i, index in enumerate(indices_to_add):
        my_list.insert(index+i, new_elements[i])
    return my_list

def flatten(seq):
    flat_list = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):  # Check if item is a list, tuple or ndarray
            flat_list.extend(flatten(item))  # Recursively call flatten on the nested list, tuple or ndarray
        else:
            flat_list.append(item)  # Append non-list/tuple/ndarray items to the flat list
    return flat_list


def remove_nan_tuples(list_of_lists, list_of_list_of_tuples):
    # Initialize a list to store indices of sub-lists or tuples to remove
    indices_to_remove = []

    for i in range(len(list_of_list_of_tuples)):
        temp=flatten(list_of_list_of_tuples[i])
        if any(np.isnan(temp)) or any(np.isinf(temp)):
            indices_to_remove.append(i)

    for i, inner_list in enumerate(list_of_lists):
        if any(val == 0 for val in inner_list):
            indices_to_remove.append(i)


    # Remove sub-lists and tuples using the stored indices
    if len(indices_to_remove)>0:
        list_of_lists = [list_of_lists[i] for i in range(len(list_of_lists)) if i not in indices_to_remove]
        list_of_list_of_tuples = [list_of_list_of_tuples[i] for i in range(len(list_of_list_of_tuples)) if i not in indices_to_remove]

    return list_of_lists, list_of_list_of_tuples, indices_to_remove

def remove_elements_by_index(my_list, index_list):
    if len(index_list)>0:
        return [item for i, item in enumerate(my_list) if i not in index_list]
    else:
        return my_list

def get_dist_names(dist_comb):
    return [str(c.__name__) for c in dist_comb]


def _dist_names_from_comb(dist_comb):
    return [dist_name_by_module.get(module, str(module.__name__).split(".")[-1])
            for module in dist_comb]


def _dist_comb_from_names(dist_names):
    return tuple(dist_map[name] for name in dist_names)


def _dist_combs_from_candidate_combinations(candidate_combinations, k):
    dist_combs = []
    for candidate in candidate_combinations:
        if len(candidate) != k:
            raise ValueError(
                f"Each candidate combination must have length {k}; got {candidate!r}."
            )
        dist_combs.append(tuple(dist_map[name] if isinstance(name, str) else name
                               for name in candidate))
    return dist_combs


def _expand_assignment_permutations(dist_combs):
    """Return every distinct family-to-component assignment once."""
    expanded = []
    seen = set()
    for dist_comb in dist_combs:
        for assignment in itertools.permutations(dist_comb):
            if assignment in seen:
                continue
            seen.add(assignment)
            expanded.append(assignment)
    return expanded


def _assignment_equivalence_key(dist_comb):
    """Identify ordered candidates representing the same unlabeled mixture."""
    return tuple(sorted(_dist_names_from_comb(dist_comb)))


def _mu_index_for_dist(dist_name):
    if dist_name in {"mvn", "mvt", "mvsl", "mvsn", "mvst", "msnc", "mssl", "msl"}:
        return 0
    if dist_name == "mghp":
        return 3
    if dist_name in {"mnig", "mvhb"}:
        return 2
    if dist_name == "mvvg":
        return 2
    if dist_name == "mgst":
        return 2
    return None


def _apply_fixed_mu(alpha, dist_comb, fixed_mu):
    if not fixed_mu:
        return alpha
    alpha_new = list(alpha)
    for item in fixed_mu:
        component = int(item["component"])
        coordinate = int(item.get("coordinate", 0))
        value = float(item["value"])
        if component < 0 or component >= len(alpha_new):
            raise ValueError(f"fixed_mu component {component} is out of range.")
        dist_name = dist_name_by_module.get(dist_comb[component])
        mu_idx = _mu_index_for_dist(dist_name)
        if mu_idx is None:
            raise NotImplementedError(f"Cannot fix mu for distribution {dist_name}.")
        params = list(alpha_new[component])
        mu = np.asarray(params[mu_idx], dtype=float).copy()
        mu[coordinate] = value
        params[mu_idx] = mu
        alpha_new[component] = tuple(params)
    return alpha_new


def _fit_fmvmm_candidate_worker(args):
    """
    Fit one FMVMM candidate in a separate process.

    The worker constructs a one-candidate FMVMM object. This avoids sharing the
    mutable EM state stored on the main object while preserving the exact fitting
    path used by sequential candidate fits.
    """
    (
        data, n_clusters, dist_names, tol, initialization, print_log_likelihood,
        max_iter, verbose, debug, em_type, gh_mstep_kwargs, smsn_mstep_kwargs, init_fit_kwargs,
        fixed_pi, fixed_mu, mstep_max_iter, mstep_maxfun, mstep_accept_tol,
        numerical_policy, candidate_initializations, strict_density,
    ) = args

    candidate = fmvmm(
        n_clusters=n_clusters,
        tol=tol,
        list_of_dist=list(dist_names),
        specific_comb=True,
        initialization=initialization,
        print_log_likelihood=print_log_likelihood,
        max_iter=max_iter,
        verbose=verbose,
        debug=debug,
        em_type=em_type,
        gh_mstep_kwargs=gh_mstep_kwargs,
        smsn_mstep_kwargs=smsn_mstep_kwargs,
        init_fit_kwargs=init_fit_kwargs,
        n_jobs=1,
        fixed_pi=fixed_pi,
        fixed_mu=fixed_mu,
        mstep_max_iter=mstep_max_iter,
        mstep_maxfun=mstep_maxfun,
        mstep_accept_tol=mstep_accept_tol,
        numerical_policy=numerical_policy,
        candidate_initializations=candidate_initializations,
        strict_density=strict_density,
    )
    candidate.fit(data)
    if len(candidate.worked_dist) == 0:
        return {
            "worked": False,
            "dist_names": list(dist_names),
            "error": candidate.candidate_failures[0]["error"] if candidate.candidate_failures else "No candidate fit completed successfully.",
        }
    return {
        "worked": True,
        "dist_names": list(dist_names),
        "aic": candidate.list_aic[0],
        "bic": candidate.list_bic[0],
        "icl": candidate.list_icl[0],
        "pi": candidate.list_pi[0],
        "alpha": candidate.list_alpha[0],
        "cluster": candidate.list_cluster[0],
        "log_likelihood": candidate.list_log_likelihood[0],
        "gamma_matrix": candidate.list_gamma_matrix[0],
        "all_log_likelihood": candidate.list_all_log_likelihood[0],
        "em_diagnostics": candidate.em_diagnostics[0],
    }




class fmvmm(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001, list_of_dist=all_dist,specific_comb=False,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True, debug = False, em_type="soft", gh_mstep_kwargs=None, smsn_mstep_kwargs=None, init_fit_kwargs=None, n_jobs=1, candidate_combinations=None, fixed_pi=None, fixed_mu=None, assignment_permutations=False, mstep_max_iter=None, mstep_maxfun=None, mstep_accept_tol=1e-8, numerical_policy=None, candidate_initializations=None, strict_density=None):
        em_type_normalized = em_type.lower()
        if em_type_normalized not in ("hard", "soft"):
            raise ValueError("em_type must be either 'hard' or 'soft'.")
        EM_type = "Soft" if em_type_normalized == "soft" else "Hard"
        super().__init__(n_clusters=n_clusters, EM_type=EM_type, mixture_type="nonidentical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.numerical_policy = None
        if numerical_policy is not None:
            from fmvmm.numerics.em_policy import make_policy
            self.numerical_policy = make_policy(numerical_policy)
            if em_type_normalized != "soft":
                raise ValueError("numerical_policy currently supports soft EM only.")
            if fixed_mu:
                raise ValueError("Constrained EM does not yet support fixed_mu constraints.")
            if self.k*self.numerical_policy.minimum_weight >= 1:
                raise ValueError("Infeasible minimum mixture weight.")
            if int(max_iter) != max_iter or max_iter < 1:
                raise ValueError("Constrained EM max_iter must be a positive integer.")
        self.strict_density = numerical_policy is not None if strict_density is None else bool(strict_density)
        if self.numerical_policy is not None and not self.strict_density:
            raise ValueError("Constrained EM requires strict_density=True.")
        self.candidate_initializations = {} if candidate_initializations is None else dict(candidate_initializations)
        self.list_of_dist = list_of_dist
        self.specific_comb=specific_comb
        self.dist_variables = [dist_map[list_of_dist[j]]
                               for j in range(len(list_of_dist))]
        self.specific_comb=specific_comb
        if candidate_combinations is not None:
            self.dist_combs = _dist_combs_from_candidate_combinations(
                candidate_combinations, self.k)
        elif self.specific_comb==True:
            self.dist_combs = list(
                itertools.combinations(self.dist_variables, self.k))
        else:
            self.dist_combs = list(
                itertools.combinations_with_replacement(self.dist_variables, self.k))
        self.assignment_permutations = bool(assignment_permutations)
        if self.assignment_permutations:
            if fixed_mu:
                raise ValueError(
                    "assignment_permutations cannot be combined with component-indexed "
                    "fixed_mu constraints. Fit the selected ordered candidate instead."
                )
            if fixed_pi is not None:
                fixed_pi_array = np.asarray(fixed_pi, dtype=float).reshape(-1)
                if not np.allclose(fixed_pi_array, fixed_pi_array[0]):
                    raise ValueError(
                        "assignment_permutations requires exchangeable fixed_pi values."
                    )
            self.dist_combs = _expand_assignment_permutations(self.dist_combs)
        self.n_assignment_starts = len(self.dist_combs)
        self.initialization=initialization
        self.debug = debug
        self.em_type = em_type_normalized
        self.mstep_max_iter = mstep_max_iter
        self.mstep_maxfun = mstep_maxfun
        self.mstep_accept_tol = float(mstep_accept_tol)
        self.gh_mstep_kwargs = {} if gh_mstep_kwargs is None else dict(gh_mstep_kwargs)
        self.smsn_mstep_kwargs = {} if smsn_mstep_kwargs is None else dict(smsn_mstep_kwargs)
        if mstep_max_iter is not None:
            self.gh_mstep_kwargs.setdefault("nit", int(mstep_max_iter))
            self.gh_mstep_kwargs.setdefault("opt_maxiter", int(mstep_max_iter))
            self.smsn_mstep_kwargs.setdefault("maxiter", int(mstep_max_iter))
        if mstep_maxfun is not None:
            self.gh_mstep_kwargs.setdefault("opt_maxfun", int(mstep_maxfun))
            self.smsn_mstep_kwargs.setdefault("maxfun", int(mstep_maxfun))
        self.init_fit_kwargs = {} if init_fit_kwargs is None else init_fit_kwargs
        self.fixed_pi = None
        if fixed_pi is not None:
            fixed_pi = np.asarray(fixed_pi, dtype=float).reshape(-1)
            if fixed_pi.size != self.k:
                raise ValueError(f"fixed_pi must have length {self.k}.")
            if np.any(fixed_pi <= 0):
                raise ValueError("fixed_pi entries must be strictly positive.")
            self.fixed_pi = fixed_pi / fixed_pi.sum()
        self.fixed_mu = [] if fixed_mu is None else list(fixed_mu)
        self.n_jobs = int(n_jobs) if n_jobs is not None else 1
        if self.n_jobs == 0:
            self.n_jobs = os.cpu_count() or 1
        if self.n_jobs < 0:
            self.n_jobs = max((os.cpu_count() or 1) + 1 + self.n_jobs, 1)
        self.init_fit_kwargs_by_module = {
            dist_map[name]: kwargs
            for name, kwargs in self.init_fit_kwargs.items()
            if name in dist_map
        }

    def _log_pdf_non_identical(self,X,alphas,dist_comb):
        N,p=X.shape
        k=len(alphas)
        probs=np.empty((N, k))
        for j in range(k):
            alpha=alphas[j]
            log_prob = np.asarray(dist_comb[j].logpdf(X,*alpha), dtype=float)
            if self.strict_density:
                if log_prob.size != N or not np.isfinite(log_prob).all():
                    raise FloatingPointError(f"Invalid log density for {_dist_names_from_comb(dist_comb)[j]}.")
                probs[:, j] = log_prob.reshape(-1)
            else:
                probs[:, j]=np.where(np.isfinite(log_prob), log_prob, -1e300)
        return probs

    def _estimate_weighted_log_prob_nonidentical(self, X, alpha, pi, dist_comb):
        return self._log_pdf_non_identical(X,alpha,dist_comb) + np.log(pi)

    def _reset_fit_storage(self):
        self.list_aic = []
        self.list_bic = []
        self.list_icl = []
        self.list_pi = []
        self.list_alpha = []
        self.list_cluster = []
        self.list_log_likelihood = []
        self.list_all_log_likelihood = []
        self.list_gamma_matrix = []
        self.not_worked_dist = []
        self.worked_dist = []
        self.em_diagnostics = []
        self.candidate_failures = []

    def _fit_single_candidate(self, dist_comb):
        try:
            self._component_diagnostics = []
            self._initialization_diagnostics = []
            self._em_stop_reason = None
            names = tuple(_dist_names_from_comb(dist_comb))
            if names in self.candidate_initializations:
                self.pi_not, self.alpha_not = copy.deepcopy(self.candidate_initializations[names])
                self.pi_not = np.asarray(self.pi_not, dtype=float).reshape(-1)
                if self.pi_not.shape != (self.k,) or not np.isfinite(self.pi_not).all() or np.any(self.pi_not <= 0):
                    raise ValueError("Candidate initial weights must be finite, strictly positive and of length k.")
                if len(self.alpha_not) != self.k:
                    raise ValueError("Candidate initialization requires one parameter tuple per component.")
                self.pi_not /= self.pi_not.sum()
                self.pi_temp, self.alpha_temp = self.pi_not, self.alpha_not
            elif self.initialization=="kmeans":
                self.pi_not, self.alpha_not = fmm_kmeans_init(
                    self.data, self.k, dist_comb,
                    fit_kwargs_by_module=self.init_fit_kwargs_by_module)
                self.alpha_temp = self.alpha_not
                self.pi_temp = self.pi_not
            else:
                self.pi_not, self.alpha_not = fmm_gmm_init(
                    self.data, self.k, dist_comb,
                    fit_kwargs_by_module=self.init_fit_kwargs_by_module)
                self.alpha_temp = self.alpha_not
                self.pi_temp = self.pi_not
            if self.fixed_pi is not None:
                self.pi_not = self.fixed_pi.copy()
                self.pi_temp = self.fixed_pi.copy()
            if self.fixed_mu:
                self.alpha_not = _apply_fixed_mu(self.alpha_not, dist_comb, self.fixed_mu)
                self.alpha_temp = self.alpha_not

            if self.numerical_policy is not None:
                from fmvmm.numerics.em_policy import bounded_weights
                from fmvmm.numerics.constrained_mstep import prepare_component
                prepared = [prepare_component(name, alpha, self.numerical_policy)
                            for name, alpha in zip(names, self.alpha_temp)]
                self.alpha_temp = [item[0] for item in prepared]
                self.alpha_not = self.alpha_temp
                self._initialization_diagnostics = [item[1] for item in prepared]
                self._accepted_component_coordinates = [np.asarray(item[1]['optimizer_coordinates_vector'],float) for item in prepared]
                if self.fixed_pi is not None and np.min(self.fixed_pi) < self.numerical_policy.minimum_weight:
                    raise ValueError("fixed_pi is outside the declared weight domain.")
                self.pi_temp = bounded_weights(self.pi_temp, self.numerical_policy.minimum_weight)
                self.pi_not = self.pi_temp
            if self.EM_type == "Soft":
                pi_new, alpha_new, log_likelihood_new, log_gamma_new = self._fit(
                    self.data,
                    self.pi_temp,
                    self.alpha_temp,
                    fmm_estimate_alphas_soft,
                    dist_comb=dist_comb,
                    soft_dist_comb=dist_comb,
                    gh_mstep_kwargs=self.gh_mstep_kwargs,
                    smsn_mstep_kwargs=self.smsn_mstep_kwargs,
                    mstep_accept_tol=self.mstep_accept_tol,
                    post_iteration=self._post_em_iteration if self.numerical_policy is not None else None,
                    reject_nonmonotone_proposal=self.numerical_policy is not None,
                    nonmonotone_tolerance=self.numerical_policy.accept_tolerance if self.numerical_policy is not None else 1e-8,
                    post_m_step=self._post_m_step_constraints if (self.fixed_pi is not None or self.fixed_mu) else None,
                )
            else:
                pi_new, alpha_new, log_likelihood_new, log_gamma_new = self._fit(
                    self.data,
                    self.pi_temp,
                    self.alpha_temp,
                    fmm_estimate_alphas,
                    dist_comb=dist_comb,
                    fit_kwargs_by_module=self.init_fit_kwargs_by_module,
                    post_m_step=self._post_m_step_constraints if (self.fixed_pi is not None or self.fixed_mu) else None,
                )

            return {
                "worked": True,
                "dist_comb": dist_comb,
                "aic": fmm_aic(alpha_new, log_likelihood_new, dist_comb),
                "bic": fmm_bic(alpha_new, log_likelihood_new, dist_comb, self.n),
                "icl": fmm_icl(alpha_new, log_likelihood_new, dist_comb, self.n, np.exp(log_gamma_new)),
                "pi": pi_new,
                "alpha": alpha_new,
                "cluster": log_gamma_new.argmax(axis=1),
                "log_likelihood": log_likelihood_new,
                "gamma_matrix": np.exp(log_gamma_new),
                "all_log_likelihood": list(self.log_likelihoods),
                "em_diagnostics": self._collect_em_diagnostics(dist_comb, pi_new, alpha_new, np.exp(log_gamma_new)),
            }
        except Exception as exc:
            if self.debug:
                traceback.print_exc()
                print("Error received while running,", dist_comb)
            return {
                "worked": False,
                "dist_comb": dist_comb,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _m_step(self, gamma_matrix, X, estimate_alphas_function, dist_comb, **kwargs):
        if self.numerical_policy is None:
            return super()._m_step(gamma_matrix, X, estimate_alphas_function, dist_comb, **kwargs)
        from fmvmm.numerics.em_policy import bounded_weights
        from fmvmm.numerics.constrained_mstep import fit_component
        pi_new = bounded_weights(gamma_matrix.sum(axis=0), self.numerical_policy.minimum_weight)
        alpha_new, records = [], []
        for j, name in enumerate(_dist_names_from_comb(dist_comb)):
            alpha, record = fit_component(name, X, gamma_matrix[:, j], self.alpha_temp[j], self.numerical_policy,
                coordinates_prev=self._accepted_component_coordinates[j])
            alpha_new.append(alpha)
            records.append(record)
        self._component_diagnostics.append(records)
        self._proposed_component_coordinates = [np.asarray(r['optimizer_coordinates_vector'],float) for r in records]
        self._proposed_component_parameters = copy.deepcopy(alpha_new)
        return pi_new, alpha_new

    def _stopping_criteria(self, log_likelihoods, tol, n_iter_max):
        if self.numerical_policy is None:
            return super()._stopping_criteria(log_likelihoods, tol, n_iter_max)
        values = np.asarray(log_likelihoods, float)
        if not np.isfinite(values[-2:]).all():
            raise FloatingPointError("Nonfinite EM likelihood trace.")
        if values[-1] < values[-2] - self.numerical_policy.accept_tolerance:
            raise FloatingPointError("Constrained EM likelihood decreased.")
        relative = abs(values[-1]-values[-2])/max(1.,abs(values[-2]))
        last = self._component_diagnostics[-1]
        resolved = all(record['gradient_acceptable'] and record['accepted'] for record in last)
        if self._nonmonotone_proposal_rejections:
            self._em_stop_reason = 'rejected_nonmonotone_proposal'
            return True
        if relative <= tol:
            final_scores, weight_gap = self._final_em_stationarity()
            stationary = all(record['gradient_acceptable'] for record in final_scores) and weight_gap <= self.numerical_policy.gradient_tolerance
            if resolved and stationary:
                self._em_stop_reason = 'relative_likelihood_and_stationarity_tolerances'
                return True
            # A small likelihood increment alone is not a stationarity test.
            # Continue the same EM loop until its budget is reached.
        if len(values)-1 >= n_iter_max:
            self._em_stop_reason = 'max_iter'
            return True
        return False

    def _post_em_iteration(self, pi, alpha, log_gamma):
        if not self._nonmonotone_proposal_rejections:
            # Keep exact optimizer coordinates if the outer loop accepted that
            # native tuple. A post-M-step constraint can alter it independently.
            self._accepted_component_coordinates = [u if all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(actual,proposed)) else None
                for u,actual,proposed in zip(self._proposed_component_coordinates,alpha,self._proposed_component_parameters)]
        self._last_em_state = (pi, alpha, log_gamma)
        self._final_stationarity_cache = None

    def _final_em_stationarity(self):
        if self._final_stationarity_cache is None:
            from fmvmm.numerics.constrained_mstep import component_stationarity
            from fmvmm.numerics.em_policy import bounded_weights
            pi, alpha, log_gamma = self._last_em_state
            gamma = np.exp(log_gamma)
            scores = [component_stationarity(name, self.data, gamma[:, j], alpha[j], self.numerical_policy,
                        coordinates=self._accepted_component_coordinates[j])
                      for j, name in enumerate(_dist_names_from_comb(self._current_dist_comb))]
            weight_gap = 0. if self.fixed_pi is not None else float(np.max(np.abs(pi-bounded_weights(gamma.sum(axis=0), self.numerical_policy.minimum_weight))))
            self._final_stationarity_cache = (scores, weight_gap)
        return self._final_stationarity_cache

    def _collect_em_diagnostics(self, dist_comb, pi, alpha, gamma):
        if self.numerical_policy is None:
            return {'backend': 'existing_em', 'iterations': len(self.log_likelihoods)-1,
                    'trace_kind': 'actual EM likelihood iterates', 'global_optimum_certified': False}
        records = [record for iteration in self._component_diagnostics for record in iteration]
        final_scores, final_weight_gap = self._final_em_stationarity()
        return {'backend': 'constrained_component_em', 'outer_loop': 'BaseMixture._fit',
                'policy': asdict(self.numerical_policy), 'coordinate_system': 'caller supplied; no internal whitening',
                'iterations': len(self.log_likelihoods)-1, 'stop_reason': self._em_stop_reason,
                'converged': self._em_stop_reason == 'relative_likelihood_and_stationarity_tolerances',
                'convergence_meaning': 'relative likelihood tolerance, resolved last weighted updates, and final-membership ordinary or explicitly classified cusp stationarity/weight tolerances',
                'final_component_stationarity': final_scores, 'final_weight_update_gap': final_weight_gap,
                'weight_floor_components': np.flatnonzero(pi<=self.numerical_policy.minimum_weight+1e-10).tolist(),
                'final_stationarity_acceptable': all(r['gradient_acceptable'] for r in final_scores) and final_weight_gap <= self.numerical_policy.gradient_tolerance,
                'hit_cap': self._em_stop_reason == 'max_iter',
                'component_cap_count': sum(record['hit_cap'] for record in records),
                'component_unresolved_count': sum(not record['gradient_acceptable'] for record in records),
                'component_rejection_count': sum(not record['accepted'] for record in records),
                'nonmonotone_proposal_rejections': self._nonmonotone_proposal_rejections,
                'last_component_max_projected_gradient': max(record['projected_gradient'] for record in self._component_diagnostics[-1]),
                'initialization': self._initialization_diagnostics, 'component_iterations': self._component_diagnostics,
                'trace_kind': 'actual EM likelihood iterates', 'global_optimum_certified': False}

    def _post_m_step_constraints(self, pi_new, alpha_new):
        if self.fixed_pi is not None:
            pi_new = self.fixed_pi.copy()
        alpha_new = _apply_fixed_mu(alpha_new, self._current_dist_comb, self.fixed_mu)
        return pi_new, alpha_new

    def _append_candidate_result(self, result):
        if not result.get("worked", False):
            dist_comb = result.get("dist_comb")
            if dist_comb is None:
                dist_comb = _dist_comb_from_names(result.get("dist_names", []))
            self.not_worked_dist.append(dist_comb)
            self.candidate_failures.append({'dist_names': _dist_names_from_comb(dist_comb),
                                            'error': result.get('error', 'Unknown candidate failure')})
            return

        dist_comb = result.get("dist_comb")
        if dist_comb is None:
            dist_comb = _dist_comb_from_names(result["dist_names"])

        self.list_aic.append(result["aic"])
        self.list_bic.append(result["bic"])
        self.list_icl.append(result["icl"])
        self.list_pi.append(result["pi"])
        self.list_alpha.append(result["alpha"])
        self.list_cluster.append(result["cluster"])
        self.list_log_likelihood.append(result["log_likelihood"])
        self.list_gamma_matrix.append(result["gamma_matrix"])
        self.worked_dist.append(dist_comb)
        self.list_all_log_likelihood.append(result["all_log_likelihood"])
        self.em_diagnostics.append(result.get("em_diagnostics", {}))
        if self.verbose:
            print("distribution fitted", get_dist_names(dist_comb))

    def _finalize_fit_storage(self):
        if self.strict_density:
            # A +infinite t/skew-t df is a valid normal/skew-normal boundary.
            # Strict E-steps have already checked the actual returned density.
            self.nan_ind = [i for i, (pi, ll) in enumerate(zip(self.list_pi, self.list_log_likelihood))
                            if not np.isfinite(pi).all() or np.any(np.asarray(pi) <= 0) or not np.isfinite(ll)]
            self.list_pi = remove_elements_by_index(self.list_pi, self.nan_ind)
            self.list_alpha = remove_elements_by_index(self.list_alpha, self.nan_ind)
        else:
            self.list_pi,self.list_alpha,self.nan_ind=remove_nan_tuples(self.list_pi,self.list_alpha)
        self.em_diagnostics=remove_elements_by_index(self.em_diagnostics,self.nan_ind)
        self.list_aic=remove_elements_by_index(self.list_aic,self.nan_ind)
        self.list_bic=remove_elements_by_index(self.list_bic, self.nan_ind)
        self.list_icl=remove_elements_by_index(self.list_icl, self.nan_ind)
        self.list_cluster=remove_elements_by_index(self.list_cluster, self.nan_ind)
        self.list_log_likelihood=remove_elements_by_index(self.list_log_likelihood, self.nan_ind)
        self.list_gamma_matrix=remove_elements_by_index(self.list_gamma_matrix, self.nan_ind)
        self.not_worked_dist.extend(w for w in [self.worked_dist[h] for h in self.nan_ind])
        self.worked_dist=remove_elements_by_index(self.worked_dist, self.nan_ind)
        self.list_all_log_likelihood=remove_elements_by_index(self.list_all_log_likelihood, self.nan_ind)

    def _collapse_assignment_permutations(self):
        """Keep the best fit for each unordered family multiset."""
        if not self.assignment_permutations or len(self.worked_dist) <= 1:
            return

        best_by_key = {}
        key_order = []
        for idx, dist_comb in enumerate(self.worked_dist):
            key = _assignment_equivalence_key(dist_comb)
            if key not in best_by_key:
                best_by_key[key] = idx
                key_order.append(key)
                continue
            current = best_by_key[key]
            if (self.list_bic[idx], -self.list_log_likelihood[idx]) < (
                self.list_bic[current], -self.list_log_likelihood[current]
            ):
                best_by_key[key] = idx

        keep = [best_by_key[key] for key in key_order]
        fields = [
            "list_aic", "list_bic", "list_icl", "list_pi", "list_alpha",
            "list_cluster", "list_log_likelihood", "list_gamma_matrix",
            "worked_dist", "list_all_log_likelihood", "em_diagnostics",
        ]
        for field in fields:
            values = getattr(self, field)
            setattr(self, field, [values[idx] for idx in keep])

    def fit(self,sample):
        self.data = self._process_data(sample)
        self.n=len(sample)
        if self.numerical_policy is not None and (self.data.ndim != 2 or self.n == 0 or not np.isfinite(self.data).all()):
            raise ValueError("Constrained EM requires a nonempty finite two-dimensional sample.")
        # self.sample=self._process_data(sample)
        self._reset_fit_storage()
        if self.n_jobs == 1 or len(self.dist_combs) <= 1:
            for dist_comb in self.dist_combs:
                self._current_dist_comb = dist_comb
                self._append_candidate_result(self._fit_single_candidate(dist_comb))
        else:
            n_workers = min(self.n_jobs, len(self.dist_combs))
            worker_args = [
                (
                    self.data, self.k, _dist_names_from_comb(dist_comb), self.tol,
                    self.initialization, self.print_log_likelihood,
                    self.max_iter, self.verbose, self.debug, self.em_type,
                    self.gh_mstep_kwargs, self.smsn_mstep_kwargs,
                    self.init_fit_kwargs, self.fixed_pi,
                    self.fixed_mu, self.mstep_max_iter, self.mstep_maxfun,
                    self.mstep_accept_tol,
                    self.numerical_policy, self.candidate_initializations, self.strict_density,
                )
                for dist_comb in self.dist_combs
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for result in executor.map(_fit_fmvmm_candidate_worker, worker_args):
                    self._append_candidate_result(result)

        self._finalize_fit_storage()
        self._collapse_assignment_permutations()
        self.fitted=True
        if self.verbose:
            print("Model fitted successfully")

    def get_params(self):
        return self.list_pi, self.list_alpha

    def predict(self):
        return self.list_cluster

    def predict_new(self, X):
        # data_lol = x.values.tolist()
        cluster_all = []
        for l in range(len(self.dist_combs)):
            # cluster, _ = mixture_clusters(self.gamma_matrix[l], data_lol)
            cluster_all.append(self._predict(X))

        return cluster_all


    def best_mixture(self):
        best_mix = self.worked_dist[np.argmin(self.list_bic)]
        return [str(best_mix[i].__name__) for i in range(len(best_mix))]

    def best_params(self):
        return self.list_pi[np.argmin(self.list_bic)], self.list_alpha[np.argmin(self.list_bic)]

    def best_predict(self):
        return self.list_cluster[np.argmin(self.list_bic)]

    def best_predict_new(self, x):
        data_lol = x.values.tolist()
        cluster_all = []
        for l in range(len(self.dist_combs)):
            cluster, _ = mixture_clusters(self.gamma_matrix[l], data_lol)
            cluster_all.append(cluster)
        return cluster_all[np.argmin(self.list_bic)]

    def best_aic(self):
        return np.min(self.list_aic)

    def best_bic(self):
        return np.min(self.list_bic)
    def not_worked(self):
        print("Distribution Combinations That Could Not Be Fitted:")
        for i in range(len(self.not_worked_dist)):
            print(i, [str(self.not_worked_dist[i][j].__name__) for j in range(len(self.not_worked_dist[i]))])

    def worked(self):
        print("Distribution Combinations That Could Be Fitted:")
        for i in range(len(self.worked_dist)):
            print(i, [str(self.worked_dist[i][j].__name__) for j in range(len(self.worked_dist[i]))])

    def get_maximum_likelihood(self):
        return self.list_log_likelihood
    def get_all_likelihood(self):
        return self.list_all_log_likelihood


    # def get_info_mat(self):
    #     """
    #     Assemble the total Fisher information matrix for each 'worked_dist' mixture
    #     distribution, combining the mixture-proportions info block and the
    #     parameter info blocks for each component.
    #
    #     Returns:
    #       mixture_info_mats: list of 2D arrays, one per mixture distribution
    #       mixture_ses:       1D array of standard errors (concatenated across mixtures)
    #     """
    #     mixture_ses = []
    #     mixture_info_mats = []
    #
    #     # Loop over each mixture distribution
    #     for a, mix in enumerate(self.worked_dist):
    #         # mix is the list of components for this mixture
    #         # e.g. mix = [comp_1, comp_2, ..., comp_k]
    #         k = len(mix)  # number of components
    #
    #         # Extract data assigned to each component, etc.
    #         cwise_data = organize_data_by_clusters(np.array(self.data), self.list_cluster[a])
    #         mle_params = self.list_alpha[a]  # parameters for each component
    #         N_j, N = mixture_counts(self.list_gamma_matrix[a], mode="hard")
    #
    #         # 1) Mixture proportion info
    #         I_pi, I_pi_inv = mixture_proportions_info(self.list_pi[a], N_j)
    #         # I_pi, I_pi_inv are k×k
    #
    #         # 2) Component-level Fisher infos
    #         #    We'll collect the Fisher info for each component and store
    #         #    their inverses in I_inv_blocks for the final block assembly.
    #         mix_info = []
    #         param_dims = []    # dimension of each component's parameter vector
    #         I_inv_blocks = []
    #
    #         for b, mix_dist in enumerate(mix):
    #             # Compute Fisher info for the b-th component
    #             fisher_info, param_len = compute_info_individual_fmvmm(
    #                 mix_dist,   # log PDF function
    #                 mle_params[b],     # parameters for that component
    #                 cwise_data[b],     # data assigned to that component
    #                 epsilon=0.06
    #             )
    #             mix_info.append(fisher_info)
    #             param_dims.append(param_len)
    #
    #             # Invert it (or pseudo-invert if singular)
    #             try:
    #                 inv_block = np.linalg.inv(fisher_info)
    #             except np.linalg.LinAlgError:
    #                 inv_block = np.linalg.pinv(fisher_info)
    #
    #             I_inv_blocks.append(inv_block)
    #
    #         # 3) Now compute the total dimension:
    #         #    - k for the mixture proportions
    #         #    - plus the sum of parameter dims across all components
    #         total_param_dim = sum(param_dims)
    #         big_dim = k + total_param_dim
    #
    #         # Allocate the overall info matrix and its inverse
    #         I_total = np.zeros((big_dim, big_dim))
    #         I_total_inv = np.zeros((big_dim, big_dim))
    #
    #         # 4) Insert mixture proportion info in the top-left k×k block
    #         I_total[:k, :k] = I_pi
    #         I_total_inv[:k, :k] = I_pi_inv
    #
    #         # 5) Insert each component's block
    #         row_start = k  # begin filling right below/after the k×k block
    #         for j in range(k):
    #             param_dim_j = param_dims[j]
    #             row_end = row_start + param_dim_j
    #
    #             # Place the j-th component's fisher info block
    #             I_total[row_start:row_end, row_start:row_end] = mix_info[j]
    #             I_total_inv[row_start:row_end, row_start:row_end] = I_inv_blocks[j]
    #
    #             row_start = row_end  # advance to the next block
    #
    #         # 6) Compute standard errors from the diagonal of the total inverse
    #         #    (Ensure positive diagonals so we don't end up with sqrt of negative)
    #         var_diag = np.diag(ensure_positive_diagonal(I_total_inv))
    #         se_total = np.sqrt(var_diag)
    #
    #         # 7) Store the results
    #         mixture_ses.append(se_total)      # collect standard errors
    #         mixture_info_mats.append(I_total) # collect the big Fisher matrix
    #
    #     return mixture_info_mats, mixture_ses

    def get_info_mat(self, method="auto", ridge=1e-8, use_pinv=True,
                     parameterization="user", return_details=False):
        """
        Observed information and standard errors for each fitted FMVMM candidate.

        Parameters
        ----------
        method : {"auto", "soft", "score", "opg", "louis",
                  "hard", "classification", "modified_louis"}
            "auto" uses "soft"/"louis" for soft-EM fits and
            "classification" for hard-EM fits.

            Soft methods use the full observed-data score/Louis information,
            including membership uncertainty. Hard/classification methods treat
            labels as known; the missing-information term is zero by
            construction.
        ridge : float
            Ridge added before inversion for numerical stability.
        use_pinv : bool
            Use Moore-Penrose inverse by default; mixture information can be
            nearly singular when components overlap or labels are weakly
            identified.
        return_details : bool
            If True, also return diagnostics and covariance matrices.
        parameterization : {"user", "eta", "internal"}
            "user" returns all pi values followed by component parameters. The
            all-pi covariance is singular because sum(pi)=1, so the returned
            information matrix is a generalized information matrix. "eta" or
            "internal" returns the identifiable eta-logit parameterization.

        Notes
        -----
        Internal parameter order is:
          eta_1,...,eta_{k-1}, theta_1, ..., theta_k
        where eta_j = log(pi_j / pi_k). The returned SE vector is on this
        identifiable scale when parameterization="eta". The default
        parameterization="user" returns SEs for all pi values via delta method.
        """
        method = method.lower()
        parameterization = parameterization.lower()

        if method == "auto":
            method = "classification" if self.EM_type == "Hard" else "louis"
        if method in {"opg", "soft"}:
            method = "score"
        if method in {"hard", "modified_louis", "classification_louis"}:
            method = "classification"
        if method not in {"score", "louis", "classification"}:
            raise NotImplementedError(
                "method must be 'auto', 'score', 'opg', 'louis', "
                "'hard', 'classification', or 'modified_louis'."
            )
        if parameterization not in {"user", "eta", "internal"}:
            raise ValueError("parameterization must be 'user', 'eta', or 'internal'.")

        mixture_info_mats = []
        mixture_ses = []
        details_all = []

        X = np.asarray(self.data, dtype=float)

        for a, mix in enumerate(self.worked_dist):
            pi = np.asarray(self.list_pi[a], dtype=float)
            alpha = self.list_alpha[a]
            tau = np.asarray(self.list_gamma_matrix[a], dtype=float)
            k = len(mix)

            if method == "classification":
                labels = np.asarray(self.list_cluster[a], dtype=int)
                I_raw, complete_opg, missing_info, S_obs, param_dims = (
                    classification_information_fmvmm(X, pi, alpha, mix, labels)
                )
                information_label = "hard_classification_modified_louis"
            elif method == "louis":
                I_raw, complete_opg, missing_info, S_obs, param_dims = (
                    louis_score_decomposition_fmvmm(X, pi, alpha, mix, tau)
                )
                information_label = "soft_louis_complete_hessian_minus_missing_score_variance"
            else:
                S_obs, _, param_dims = observed_score_matrix_fmvmm(
                    X, pi, alpha, mix, tau
                )
                I_raw = S_obs.T @ S_obs
                I_raw = 0.5 * (I_raw + I_raw.T)
                complete_opg = None
                missing_info = None
                information_label = "soft_observed_score_opg"

            I_internal, cov_internal, se_internal = invert_information_matrix(
                I_raw, ridge=ridge, use_pinv=use_pinv
            )

            eta_dim = max(k - 1, 0)
            cov_eta = cov_internal[:eta_dim, :eta_dim] if eta_dim else np.zeros((0, 0))
            cov_pi = pi_cov_from_eta_cov(pi, cov_eta)
            se_pi = np.sqrt(np.maximum(np.diag(cov_pi), 0.0))
            I_user, cov_user, se_user, user_transform = transform_information_to_user_scale(
                pi, param_dims, cov_internal, ridge=ridge, use_pinv=use_pinv
            )

            score_sum = S_obs.sum(axis=0)
            try:
                condition_number = np.linalg.cond(I_internal)
            except np.linalg.LinAlgError:
                condition_number = np.inf

            if parameterization == "user":
                mixture_info_mats.append(I_user)
                mixture_ses.append(se_user)
                returned_parameterization = "user_all_pi"
            else:
                mixture_info_mats.append(I_internal)
                mixture_ses.append(se_internal)
                returned_parameterization = "eta_internal"

            details_all.append({
                "method": method,
                "information_label": information_label,
                "returned_parameterization": returned_parameterization,
                "internal_parameterization": "eta logits followed by component score_mat blocks",
                "user_parameterization": "all pi followed by component score_mat blocks",
                "eta_dim": eta_dim,
                "component_param_dims": param_dims,
                "info_internal": I_internal,
                "cov_internal": cov_internal,
                "se_internal": se_internal,
                "info_user": I_user,
                "cov_user": cov_user,
                "se_user": se_user,
                "user_transform": user_transform,
                "cov": cov_user if parameterization == "user" else cov_internal,
                "cov_eta": cov_eta,
                "cov_pi": cov_pi,
                "se_pi": se_pi,
                "observed_score_matrix": S_obs,
                "score_sum_norm": float(np.linalg.norm(score_sum)),
                "condition_number": float(condition_number),
                "complete_score_opg": complete_opg,
                "complete_information": complete_opg,
                "missing_information": missing_info,
            })

        if return_details:
            return mixture_info_mats, mixture_ses, details_all
        return mixture_info_mats, mixture_ses

    def get_info_mat_soft(self, *args, **kwargs):
        """
        Corrected full soft-mixture observed information. See get_info_mat().
        """
        kwargs.setdefault("method", "louis")
        return self.get_info_mat(*args, **kwargs)

    def get_info_mat_hard(self, *args, **kwargs):
        """
        Hard/classification modified-Louis information treating labels as known.
        The missing-information term for membership uncertainty is zero.
        """
        kwargs.setdefault("method", "classification")
        return self.get_info_mat(*args, **kwargs)

    def top_mixtures(self, n_top=10, criterion="bic"):
        """
        Return the top fitted candidate mixtures by AIC, BIC, ICL, or log-likelihood.

        The output is a list of dictionaries so it can be written directly to a
        table for model-selection leaderboards.
        """
        criterion = criterion.lower()
        if criterion not in {"aic", "bic", "icl", "loglik", "log_likelihood"}:
            raise ValueError("criterion must be one of 'aic', 'bic', 'icl', or 'loglik'.")

        rows = []
        for i, dist_comb in enumerate(self.worked_dist):
            rows.append({
                "index": i,
                "mixture": [str(dist_comb[j].__name__) for j in range(len(dist_comb))],
                "aic": float(self.list_aic[i]),
                "bic": float(self.list_bic[i]),
                "icl": float(self.list_icl[i]),
                "log_likelihood": float(self.list_log_likelihood[i]),
            })

        key = "log_likelihood" if criterion in {"loglik", "log_likelihood"} else criterion
        reverse = key == "log_likelihood"
        rows = sorted(rows, key=lambda row: row[key], reverse=reverse)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row[f"delta_{key}"] = (
                row[key] - rows[0][key] if reverse else row[key] - rows[0][key]
            )
        return rows[:n_top]

    def get_top_mixtures(self,n_top=10):
        """
        Backward-compatible BIC leaderboard used by earlier examples.
        """
        return [(row["bic"], row["mixture"]) for row in self.top_mixtures(n_top=n_top, criterion="bic")]
