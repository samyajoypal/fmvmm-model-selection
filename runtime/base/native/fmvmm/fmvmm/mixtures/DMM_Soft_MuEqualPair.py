"""
fmvmm.mixtures.DMM_Soft_MuEqualPair

Constrained Dirichlet Mixture Model (Soft EM)

Null model: equality of Dirichlet mean vectors for a chosen pair of components:
    mu[j1] == mu[j2], but with different precisions allowed (s_j free).

We enforce the constraint via dedicated constrained maximization inside the M-step:
- Update pi as usual (soft EM).
- Update alpha for non-constrained components using existing estimate_alphas.
- For the constrained pair, do coordinate ascent on (mu_shared, s1, s2):
    * update s1, s2 via Newton (same formula as your _fit_s_known_m logic)
    * update mu_shared by maximizing Q_a(mu)+Q_b(mu) in ALR coords using L-BFGS-B

This file does NOT modify DMM_Soft. It provides a dedicated null-model class
intended for LRT and size/power studies.

Author: patch for Samyajoy Pal's fmvmm framework
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma, gammaln
from scipy.optimize import minimize

from fmvmm.mixtures.DMM_Soft import DMM_Soft, estimate_alphas


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

_EPS_X = 1e-300
_EPS_W = 1e-12
_S_MIN = 1e-8
_S_MAX = 1e12


def _trigamma(x):
    return polygamma(1, x)


def _alr_inverse_eta_to_simplex(eta: np.ndarray) -> np.ndarray:
    """
    ALR inverse with last component as reference:
        mu_j = exp(eta_j) / (1 + sum exp(eta))
        mu_p = 1 / (1 + sum exp(eta))
    """
    eta = np.asarray(eta, dtype=float)
    z = np.exp(np.clip(eta, -700, 700))  # avoid overflow
    denom = 1.0 + np.sum(z)
    mu = np.empty((eta.size + 1,), dtype=float)
    mu[:-1] = z / denom
    mu[-1] = 1.0 / denom
    # floors for numerical safety
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()
    return mu


def _simplex_to_alr_eta(mu: np.ndarray) -> np.ndarray:
    """eta_j = log(mu_j / mu_p), j=1..p-1"""
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()
    return np.log(mu[:-1] / mu[-1])


# -----------------------------------------------------------------------------
# Constrained pair update: (mu_shared, s1, s2)
# -----------------------------------------------------------------------------

def _update_s_given_mu(
    logx: np.ndarray,
    gamma_col: np.ndarray,
    mu: np.ndarray,
    s_init: float,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """
    Newton update for s given fixed mu (same formula as your _fit_s_known_m core).

    Inputs
    ------
    logx: (n,p) log(X)
    gamma_col: (n,) responsibilities for this component
    mu: (p,) fixed mean on simplex
    s_init: initial precision (positive)

    Returns
    -------
    s_new: updated precision
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()

    n = logx.shape[0]
    Nj = float(np.sum(gamma_col))
    Nj = max(Nj, _EPS_W)

    # c = sum_i sum_j mu_j * gamma_i * log x_ij
    # vectorized: sum_i gamma_i * (logx_i dot mu)
    c = float(np.sum(gamma_col * (logx @ mu)))

    s_old = float(np.clip(s_init, _S_MIN, _S_MAX))

    for _ in range(max_iter):
        sm = np.clip(s_old * mu, _S_MIN, _S_MAX)

        a = Nj * float(digamma(s_old))
        b = Nj * float(np.sum(mu * digamma(sm)))
        first = a - b + c

        d = Nj * float(_trigamma(s_old))
        e = Nj * float(np.sum((mu ** 2) * _trigamma(sm)))
        second = d - e

        # same case logic as your code, but guarded
        if not (np.isfinite(first) and np.isfinite(second)):
            # numeric fallback: shrink s a bit, keep positive
            s_new = max(_S_MIN, 0.9 * s_old)
        else:
            # original two candidate steps (same formulas)
            if first + s_old * second < 0:
                # sj_new = 1 / (1/s + (1/s^2)*(1/second)*first)
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


def _mu_objective_and_grad_eta(
    eta: np.ndarray,
    Nj1: float,
    Nj2: float,
    s1: float,
    s2: float,
    lbar1: np.ndarray,
    lbar2: np.ndarray,
):
    """
    Negative Q(mu) and gradient wrt eta for shared mu.
    Q(mu) = Q1(mu; s1) + Q2(mu; s2), with s fixed.

    Only the mu-dependent part is needed:
      Qh(mu) = -Njh * sum_j logGamma(sh * mu_j) + Njh * sh * sum_j mu_j * lbar_hj + const

    where lbar_hj = (1/Njh) sum_i gamma_i,h * log x_ij.

    We optimize in ALR coordinates eta (p-1), mu = alr_inverse(eta).
    """
    mu = _alr_inverse_eta_to_simplex(eta)

    sm1 = np.clip(s1 * mu, _S_MIN, _S_MAX)
    sm2 = np.clip(s2 * mu, _S_MIN, _S_MAX)

    # Q(mu) up to constants:
    # -Nj * sum logGamma(s mu) + Nj*s*(mu dot lbar)
    Q1 = -Nj1 * float(np.sum(gammaln(sm1))) + Nj1 * s1 * float(np.sum(mu * lbar1))
    Q2 = -Nj2 * float(np.sum(gammaln(sm2))) + Nj2 * s2 * float(np.sum(mu * lbar2))
    Q = Q1 + Q2

    # g_mu = dQ/dmu (no simplex constraint needed due to eta parametrization)
    # d/dmu_j [-Nj logGamma(s mu_j)] = -Nj * s * psi(s mu_j)
    # d/dmu_j [Nj*s*mu_j*lbar_j] = Nj*s*lbar_j
    g1 = Nj1 * s1 * (lbar1 - digamma(sm1))
    g2 = Nj2 * s2 * (lbar2 - digamma(sm2))
    g_mu = g1 + g2  # (p,)

    # chain rule for softmax/ALR:
    # (J^T g)_j = mu_j * (g_j - sum_i mu_i g_i), j=1..p-1
    g_dot = float(np.sum(mu * g_mu))
    grad_eta = mu[:-1] * (g_mu[:-1] - g_dot)

    # We minimize negative Q
    return -Q, -grad_eta


def _update_mu_shared(
    mu_init: np.ndarray,
    Nj1: float,
    Nj2: float,
    s1: float,
    s2: float,
    lbar1: np.ndarray,
    lbar2: np.ndarray,
    max_iter: int = 200,
):
    """
    Update mu_shared by maximizing Q1(mu; s1) + Q2(mu; s2) on the simplex,
    using ALR coordinates and L-BFGS-B.
    """
    mu0 = np.asarray(mu_init, dtype=float)
    mu0 = np.clip(mu0, _EPS_W, 1.0)
    mu0 /= mu0.sum()
    eta0 = _simplex_to_alr_eta(mu0)

    def fun_and_grad(eta):
        f, g = _mu_objective_and_grad_eta(eta, Nj1, Nj2, s1, s2, lbar1, lbar2)
        if not (np.isfinite(f) and np.all(np.isfinite(g))):
            # fallback: penalize
            return 1e100, np.zeros_like(g)
        return f, g

    res = minimize(
        fun=lambda e: fun_and_grad(e)[0],
        x0=eta0,
        jac=lambda e: fun_and_grad(e)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )

    mu_new = _alr_inverse_eta_to_simplex(res.x)
    return mu_new


def _constrained_pair_update_meanprecision(
    X: np.ndarray,
    gamma: np.ndarray,
    alpha_old: np.ndarray,
    j1: int,
    j2: int,
    outer_max_iter: int = 50,
    outer_tol: float = 1e-6,
):
    """
    Perform constrained update for alpha at components j1 and j2:
      mu_j1 == mu_j2 == mu_shared, s_j1 and s_j2 free.

    Returns updated alpha_j1, alpha_j2.
    """
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    alpha_old = np.asarray(alpha_old, dtype=float)

    n, p = X.shape

    logx = np.log(np.clip(X, _EPS_X, None))

    g1 = gamma[:, j1]
    g2 = gamma[:, j2]
    Nj1 = float(np.sum(g1))
    Nj2 = float(np.sum(g2))
    Nj1 = max(Nj1, _EPS_W)
    Nj2 = max(Nj2, _EPS_W)

    # lbar_h = (1/Njh) sum_i gamma_ih log x_i
    lbar1 = (logx.T @ g1) / Nj1
    lbar2 = (logx.T @ g2) / Nj2

    # init from alpha_old means, average
    a1 = np.clip(alpha_old[j1], _EPS_W, None)
    a2 = np.clip(alpha_old[j2], _EPS_W, None)
    s1 = float(np.clip(np.sum(a1), _S_MIN, _S_MAX))
    s2 = float(np.clip(np.sum(a2), _S_MIN, _S_MAX))
    mu1 = a1 / np.sum(a1)
    mu2 = a2 / np.sum(a2)
    mu = 0.5 * (mu1 + mu2)
    mu = np.clip(mu, _EPS_W, 1.0)
    mu /= mu.sum()

    # coordinate ascent outer loop
    prev = np.concatenate([mu, [s1, s2]])

    for _ in range(int(outer_max_iter)):
        # update s1, s2 given mu
        s1 = _update_s_given_mu(logx, g1, mu, s1, max_iter=100, tol=1e-6)
        s2 = _update_s_given_mu(logx, g2, mu, s2, max_iter=100, tol=1e-6)

        # update mu given s1,s2
        mu = _update_mu_shared(mu, Nj1, Nj2, s1, s2, lbar1, lbar2, max_iter=200)

        cur = np.concatenate([mu, [s1, s2]])
        if np.linalg.norm(cur - prev) <= outer_tol * (1.0 + np.linalg.norm(prev)):
            break
        prev = cur

    alpha1_new = s1 * mu
    alpha2_new = s2 * mu
    alpha1_new = np.clip(alpha1_new, _EPS_W, None)
    alpha2_new = np.clip(alpha2_new, _EPS_W, None)
    return alpha1_new, alpha2_new


# -----------------------------------------------------------------------------
# Public model class
# -----------------------------------------------------------------------------

class DMM_Soft_MuEqualPair(DMM_Soft):
    """
    Soft DMM with constraint mu[j1] == mu[j2] but precision allowed to differ.

    Notes
    -----
    This class is intended as a *null model* for LRTs.

    df reduction (vs full mean-precision model) is (p-1):
        because mu is a simplex vector with p-1 DOF, and you tie one mean to another.

    Parameters
    ----------
    n_clusters : int
        Number of mixture components K.
    mu_equal_pair : tuple(int,int)
        Pair (j1, j2) of component indices (0-based) whose means are forced equal.
    method : str
        Must be "meanprecision" (recommended). Other methods are not supported
        here to keep the null fit well-defined.
    """

    def __init__(
        self,
        n_clusters: int,
        mu_equal_pair: tuple[int, int],
        tol: float = 1e-4,
        initialization: str = "kmeans",
        method: str = "meanprecision",
        print_log_likelihood: bool = False,
        max_iter: int = 25,
        verbose: bool = True,
        outer_max_iter: int = 50,
        outer_tol: float = 1e-6,
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

        j1, j2 = mu_equal_pair
        if j1 == j2:
            raise ValueError("mu_equal_pair must contain two distinct component indices.")
        if not (0 <= j1 < n_clusters and 0 <= j2 < n_clusters):
            raise ValueError("mu_equal_pair indices out of range for K=n_clusters.")

        self.mu_equal_pair = (int(j1), int(j2))
        self.outer_max_iter = int(outer_max_iter)
        self.outer_tol = float(outer_tol)

    def _m_step(self, gamma_matrix, X, estimate_alphas_function, dist_comb=None, **kwargs):
        """
        Override M-step:
          1) Update pi as usual (soft EM).
          2) Update alpha for all components using existing estimate_alphas (full update).
          3) Replace alpha rows for the constrained pair by the constrained maximizer
             (mu shared, s free) using dedicated coordinate ascent.

        Important: This is NOT a post_m_step projection. It is a constrained maximization
        step specifically for the chosen pair.
        """
        if self.method != "meanprecision":
            raise ValueError(
                "DMM_Soft_MuEqualPair currently supports only method='meanprecision' "
                "to ensure a valid constrained null fit."
            )

        gamma_matrix = np.asarray(gamma_matrix, dtype=float)
        X = np.asarray(X, dtype=float)

        # --- standard pi update ---
        nj = gamma_matrix.sum(axis=0) + 10.0 * np.finfo(gamma_matrix.dtype).eps
        pi_new = nj / nj.sum()

        # --- full alpha update (unconstrained) as a starting point ---
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

        # --- constrained replacement for the chosen pair ---
        j1, j2 = self.mu_equal_pair
        a1_new, a2_new = _constrained_pair_update_meanprecision(
            X=X,
            gamma=gamma_matrix,
            alpha_old=alpha_new,      # use the updated alpha as warm start
            j1=j1,
            j2=j2,
            outer_max_iter=self.outer_max_iter,
            outer_tol=self.outer_tol,
        )
        alpha_new[j1, :] = a1_new
        alpha_new[j2, :] = a2_new

        return pi_new, alpha_new

    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
        """
        Fit constrained DMM. post_m_step is ignored for safety.
        """
        super().fit(sample, true_m=true_m, true_s=true_s, post_m_step=None)

        # Adjust parameter count for LRT df bookkeeping:
        # Full alpha has K*p free, but tying two means removes (p-1) DOF.
        # (pi block unchanged here)
        self.total_parameters = int(self.total_parameters - (self.p - 1))

        return self
