"""
fmvmm.mixtures.DMM_Soft_NullModels

Dedicated null-model implementations for valid LRT fits under Soft DMM.

These classes DO NOT modify DMM_Soft. They implement constraints inside the
model class so that the fitted parameters satisfy the null by construction.

Null models included
--------------------
(1) DMM_Soft_IdenticalPrecision
    H0: s_1 = ... = s_K (common precision), means mu_k free.
    Uses your existing estimation routine: method="meanprecision_identical".

(2) DMM_Soft_IdenticalPi
    H0: pi_k = 1/K for all k.
    Enforced in the M-step by overriding pi update. Alpha updated as usual.

(3) DMM_Soft_IdenticalMean
    H0: mu_1 = ... = mu_K (common mean vector), but s_k free for each component.
    Implemented via dedicated constrained update inside the M-step:
        - update pi as usual
        - get an unconstrained alpha update as warm start (estimate_alphas)
        - replace alpha rows by constrained maximizers under mu common:
              coordinate ascent on (mu_common, s_1,...,s_K)
            where:
              * each s_k updated by Newton given fixed mu (same formula as _fit_s_known_m core)
              * mu updated by maximizing sum_k Q_k(mu; s_k) in ALR coords using L-BFGS-B

Notes
-----
- These null models are intended for LRT; df should be computed from parameter DOF:
    full (meanprecision): (K-1) + K*(p-1) + K        [pi + mu + s]
    identical precision: (K-1) + K*(p-1) + 1         df = K-1
    identical pi:        0     + K*(p-1) + K         df = K-1
    identical mean:      (K-1) + (p-1)    + K        df = (K-1)*(p-1)

- Mixture non-regularity still applies in general, but these are "interior" constraints
  (no boundary pi=0, no changing K), so chi-square is much more defensible than k vs k-1.

Author: patch for Samyajoy Pal's fmvmm framework
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma, gammaln
from scipy.optimize import minimize

from fmvmm.mixtures.DMM_Soft import DMM_Soft, estimate_alphas


# -----------------------------------------------------------------------------
# Numerics helpers
# -----------------------------------------------------------------------------

_EPS_X = 1e-300
_EPS_W = 1e-12
_S_MIN = 1e-8
_S_MAX = 1e12


def _trigamma(x):
    return polygamma(1, x)


def _alr_inverse_eta_to_simplex(eta: np.ndarray) -> np.ndarray:
    """ALR inverse with last component as reference."""
    eta = np.asarray(eta, dtype=float)
    z = np.exp(np.clip(eta, -700, 700))
    denom = 1.0 + np.sum(z)
    mu = np.empty((eta.size + 1,), dtype=float)
    mu[:-1] = z / denom
    mu[-1] = 1.0 / denom
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()
    return mu


def _simplex_to_alr_eta(mu: np.ndarray) -> np.ndarray:
    """eta_j = log(mu_j / mu_p), j=1..p-1"""
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()
    return np.log(mu[:-1] / mu[-1])


def _update_s_given_mu(logx, gamma_col, mu, s_init, max_iter=100, tol=1e-6):
    """
    Newton update for s given fixed mu (same formulas as your _fit_s_known_m core).
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()

    Nj = float(np.sum(gamma_col))
    Nj = max(Nj, _EPS_W)

    # c = sum_i gamma_i * (logx_i dot mu)
    c = float(np.sum(gamma_col * (logx @ mu)))

    s_old = float(np.clip(s_init, _S_MIN, _S_MAX))

    for _ in range(int(max_iter)):
        sm = np.clip(s_old * mu, _S_MIN, _S_MAX)

        a = Nj * float(digamma(s_old))
        b = Nj * float(np.sum(mu * digamma(sm)))
        first = a - b + c

        d = Nj * float(_trigamma(s_old))
        e = Nj * float(np.sum((mu ** 2) * _trigamma(sm)))
        second = d - e

        if not (np.isfinite(first) and np.isfinite(second)):
            s_new = max(_S_MIN, 0.9 * s_old)
        else:
            if first + s_old * second < 0:
                denom = (1.0 / s_old) + (1.0 / (s_old * s_old)) * (1.0 / second) * first
                if denom <= 0 or not np.isfinite(denom):
                    s_new = max(_S_MIN, 0.9 * s_old)
                else:
                    s_new = 1.0 / denom
            elif second < 0:
                s_new = s_old - first / second
            else:
                s_new = max(_S_MIN, 0.9 * s_old)

        s_new = float(np.clip(s_new, _S_MIN, _S_MAX))
        if abs(s_new - s_old) <= tol * (1.0 + abs(s_old)):
            return s_new
        s_old = s_new

    return s_old


def _mu_objective_and_grad_eta(eta, Njs, s_vec, lbar_mat):
    """
    Negative Q(mu) and grad wrt eta for common mu.

    Njs: (K,)
    s_vec: (K,)
    lbar_mat: (K,p) where lbar[k] = (1/Nk) sum_i gamma_ik log x_i
    """
    mu = _alr_inverse_eta_to_simplex(eta)     # (p,)
    K = len(Njs)

    Q = 0.0
    g_mu = np.zeros_like(mu)

    for k in range(K):
        Nk = float(Njs[k])
        sk = float(s_vec[k])
        sm = np.clip(sk * mu, _S_MIN, _S_MAX)

        Q += -Nk * float(np.sum(gammaln(sm))) + Nk * sk * float(np.sum(mu * lbar_mat[k]))

        g_mu += Nk * sk * (lbar_mat[k] - digamma(sm))

    g_dot = float(np.sum(mu * g_mu))
    grad_eta = mu[:-1] * (g_mu[:-1] - g_dot)

    return -float(Q), -grad_eta


def _update_mu_common(mu_init, Njs, s_vec, lbar_mat, max_iter=200):
    mu0 = np.asarray(mu_init, dtype=float)
    mu0 = np.clip(mu0, _EPS_W, 1.0)
    mu0 /= mu0.sum()
    eta0 = _simplex_to_alr_eta(mu0)

    def fun_and_grad(e):
        f, g = _mu_objective_and_grad_eta(e, Njs, s_vec, lbar_mat)
        if not (np.isfinite(f) and np.all(np.isfinite(g))):
            return 1e100, np.zeros_like(g)
        return f, g

    res = minimize(
        fun=lambda e: fun_and_grad(e)[0],
        x0=eta0,
        jac=lambda e: fun_and_grad(e)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    return _alr_inverse_eta_to_simplex(res.x)


def _constrained_identical_mean_update(X, gamma, alpha_warm, outer_max_iter=50, outer_tol=1e-6):
    """
    Enforce mu_1=...=mu_K (common mu), with s_k free.

    Returns alpha_new (K,p).
    """
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    alpha_warm = np.asarray(alpha_warm, dtype=float)

    n, p = X.shape
    K = gamma.shape[1]

    logx = np.log(np.clip(X, _EPS_X, None))
    Njs = np.sum(gamma, axis=0)
    Njs = np.maximum(Njs, _EPS_W)

    # lbar[k] = (logx^T @ gamma[:,k]) / Njs[k]
    lbar_mat = (logx.T @ gamma) / Njs   # (p,K)
    lbar_mat = lbar_mat.T              # (K,p)

    # init mu common from warm alpha means average
    s_vec = np.clip(np.sum(alpha_warm, axis=1), _S_MIN, _S_MAX)
    mu_mat = alpha_warm / np.sum(alpha_warm, axis=1, keepdims=True)
    mu = np.mean(mu_mat, axis=0)
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()

    prev = np.concatenate([mu, s_vec])

    for _ in range(int(outer_max_iter)):
        # update each s_k given mu
        for k in range(K):
            s_vec[k] = _update_s_given_mu(logx, gamma[:, k], mu, float(s_vec[k]), max_iter=100, tol=1e-6)

        # update mu given s_vec
        mu = _update_mu_common(mu, Njs, s_vec, lbar_mat, max_iter=200)

        cur = np.concatenate([mu, s_vec])
        if np.linalg.norm(cur - prev) <= outer_tol * (1.0 + np.linalg.norm(prev)):
            break
        prev = cur

    alpha_new = (s_vec[:, None] * mu[None, :])
    alpha_new = np.clip(alpha_new, _EPS_W, None)
    return alpha_new


# -----------------------------------------------------------------------------
# Null models
# -----------------------------------------------------------------------------

class DMM_Soft_IdenticalPrecision(DMM_Soft):
    """
    H0: s_1 = ... = s_K (common precision), mu_k free.

    Uses your existing method="meanprecision_identical".
    df(full - null) = (K-1).
    """

    def __init__(
        self,
        n_clusters,
        tol=1e-4,
        initialization="kmeans",
        print_log_likelihood=False,
        max_iter=25,
        verbose=True,
    ):
        super().__init__(
            n_clusters=n_clusters,
            tol=tol,
            initialization=initialization,
            method="meanprecision_identical",
            print_log_likelihood=print_log_likelihood,
            max_iter=max_iter,
            verbose=verbose,
        )

    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
        super().fit(sample, true_m=true_m, true_s=true_s, post_m_step=None)
        # parameter count: (K-1) + K*(p-1) + 1
        K = self.k
        p = self.p
        self.total_parameters = int((K - 1) + K * (p - 1) + 1)
        return self


class DMM_Soft_IdenticalPi(DMM_Soft):
    """
    H0: pi_k = 1/K for all k.

    Enforced by overriding M-step pi update.
    Alpha update is unconstrained (method as chosen).

    df(full - null) = (K-1) (you remove eta block).
    """

    def _m_step(self, gamma_matrix, X, estimate_alphas_function, dist_comb=None, **kwargs):
        gamma_matrix = np.asarray(gamma_matrix, dtype=float)
        X = np.asarray(X, dtype=float)

        K = gamma_matrix.shape[1]
        pi_new = np.ones((K,), dtype=float) / float(K)

        alpha_new = estimate_alphas_function(
            X,
            gamma_matrix,
            self.alpha_temp,
            method=self.method,
            true_m=kwargs.get("true_m", None),
            true_s=kwargs.get("true_s", None),
            post_m_step=None,
        )
        alpha_new = np.asarray(alpha_new, dtype=float)
        return pi_new, alpha_new

    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
        super().fit(sample, true_m=true_m, true_s=true_s, post_m_step=None)
        # parameter count: 0 + K*(p-1) + K  (if method="meanprecision")
        # BUT DMM_Soft stores alpha as K*p, so we keep bookkeeping formula:
        # If you use meanprecision: alpha DOF is K*(p-1)+K
        K = self.k
        p = self.p
        if self.method == "meanprecision":
            self.total_parameters = int(0 + K * (p - 1) + K)
        elif self.method == "meanprecision_identical":
            self.total_parameters = int(0 + K * (p - 1) + 1)
        else:
            # fallback: treat alpha as K*p free (conservative for df reporting)
            self.total_parameters = int(0 + K * p)
        return self


class DMM_Soft_IdenticalMean(DMM_Soft):
    """
    H0: mu_1 = ... = mu_K (common mean), precision s_k free.

    Implemented as dedicated constrained update inside M-step.
    Recommended with method="meanprecision" only.
    df(full - null) = (K-1)*(p-1).
    """

    def __init__(
        self,
        n_clusters,
        tol=1e-4,
        initialization="kmeans",
        method="meanprecision",
        print_log_likelihood=False,
        max_iter=25,
        verbose=True,
        outer_max_iter=50,
        outer_tol=1e-6,
    ):
        super().__init__(
            n_clusters=n_clusters,
            tol=tol,
            initialization=initialization,
            method=method,
            print_log_likelihood=print_log_likelihood,
            max_iter=max_iter,
            verbose=verbose,
        )
        self.outer_max_iter = int(outer_max_iter)
        self.outer_tol = float(outer_tol)

    def _m_step(self, gamma_matrix, X, estimate_alphas_function, dist_comb=None, **kwargs):
        if self.method != "meanprecision":
            raise ValueError("DMM_Soft_IdenticalMean supports only method='meanprecision'.")

        gamma_matrix = np.asarray(gamma_matrix, dtype=float)
        X = np.asarray(X, dtype=float)

        # pi update as usual
        nj = gamma_matrix.sum(axis=0) + 10.0 * np.finfo(gamma_matrix.dtype).eps
        pi_new = nj / nj.sum()

        # warm start alpha from unconstrained update
        alpha_warm = estimate_alphas_function(
            X,
            gamma_matrix,
            self.alpha_temp,
            method=self.method,
            true_m=kwargs.get("true_m", None),
            true_s=kwargs.get("true_s", None),
            post_m_step=None,
        )
        alpha_warm = np.asarray(alpha_warm, dtype=float)

        # constrained identical-mean update
        alpha_new = _constrained_identical_mean_update(
            X, gamma_matrix, alpha_warm,
            outer_max_iter=self.outer_max_iter,
            outer_tol=self.outer_tol,
        )
        return pi_new, alpha_new

    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
        super().fit(sample, true_m=true_m, true_s=true_s, post_m_step=None)
        # parameter count: (K-1) + (p-1) + K
        K = self.k
        p = self.p
        self.total_parameters = int((K - 1) + (p - 1) + K)
        return self

# -----------------------------------------------------------------------------
# NEW: Uniform alpha within a component
# -----------------------------------------------------------------------------

class DMM_Soft_UniformAlphaComponent(DMM_Soft):
    """
    H0: For selected component k0,
        alpha_{k0,1} = ... = alpha_{k0,p} = c_k0  (free scalar)

    Other components free.

    df(full - null) = (p-1)
    """

    def __init__(
        self,
        n_clusters,
        component_index=0,
        tol=1e-4,
        initialization="kmeans",
        method="meanprecision",
        print_log_likelihood=False,
        max_iter=25,
        verbose=True,
    ):
        super().__init__(
            n_clusters=n_clusters,
            tol=tol,
            initialization=initialization,
            method=method,
            print_log_likelihood=print_log_likelihood,
            max_iter=max_iter,
            verbose=verbose,
        )
        self.component_index = int(component_index)

    def _m_step(self, gamma_matrix, X, estimate_alphas_function, dist_comb=None, **kwargs):

        gamma_matrix = np.asarray(gamma_matrix, dtype=float)
        X = np.asarray(X, dtype=float)

        # standard pi update
        nj = gamma_matrix.sum(axis=0) + 10.0 * np.finfo(gamma_matrix.dtype).eps
        pi_new = nj / nj.sum()

        # unconstrained alpha update
        alpha_new = estimate_alphas_function(
            X,
            gamma_matrix,
            self.alpha_temp,
            method=self.method,
            true_m=kwargs.get("true_m", None),
            true_s=kwargs.get("true_s", None),
            post_m_step=None,
        )
        alpha_new = np.asarray(alpha_new, dtype=float)

        # enforce equality inside chosen component
        k0 = self.component_index
        c_hat = np.mean(alpha_new[k0])
        alpha_new[k0, :] = c_hat

        alpha_new = np.clip(alpha_new, _EPS_W, None)
        return pi_new, alpha_new

    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
        super().fit(sample, true_m=true_m, true_s=true_s, post_m_step=None)

        K = self.k
        p = self.p

        # full: (K-1) + K*(p-1) + K
        # null removes (p-1) DOF in chosen component
        self.total_parameters = int((K - 1) + K * (p - 1) + K - (p - 1))
        return self


# -----------------------------------------------------------------------------
# NEW: Fixed uniform Dirichlet component alpha = 1
# -----------------------------------------------------------------------------

class DMM_Soft_FixedUniformAlphaComponent(DMM_Soft):
    """
    H0: For selected component k0,
        alpha_{k0,j} = 1 for all j  (fully fixed)

    Other components free.

    df(full - null) = p
    """

    def __init__(
        self,
        n_clusters,
        component_index=1,
        tol=1e-4,
        initialization="kmeans",
        method="meanprecision",
        print_log_likelihood=False,
        max_iter=25,
        verbose=True,
    ):
        super().__init__(
            n_clusters=n_clusters,
            tol=tol,
            initialization=initialization,
            method=method,
            print_log_likelihood=print_log_likelihood,
            max_iter=max_iter,
            verbose=verbose,
        )
        self.component_index = int(component_index)

    def _m_step(self, gamma_matrix, X, estimate_alphas_function, dist_comb=None, **kwargs):

        gamma_matrix = np.asarray(gamma_matrix, dtype=float)
        X = np.asarray(X, dtype=float)

        # standard pi update
        nj = gamma_matrix.sum(axis=0) + 10.0 * np.finfo(gamma_matrix.dtype).eps
        pi_new = nj / nj.sum()

        # unconstrained alpha update
        alpha_new = estimate_alphas_function(
            X,
            gamma_matrix,
            self.alpha_temp,
            method=self.method,
            true_m=kwargs.get("true_m", None),
            true_s=kwargs.get("true_s", None),
            post_m_step=None,
        )
        alpha_new = np.asarray(alpha_new, dtype=float)

        # enforce fixed uniform alpha
        k0 = self.component_index
        p = alpha_new.shape[1]
        alpha_new[k0, :] = np.ones(p)

        alpha_new = np.clip(alpha_new, _EPS_W, None)
        return pi_new, alpha_new

    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
        super().fit(sample, true_m=true_m, true_s=true_s, post_m_step=None)

        K = self.k
        p = self.p

        # full: (K-1) + K*(p-1) + K
        # remove p parameters in selected component
        self.total_parameters = int((K - 1) + K * (p - 1) + K - p)
        return self
