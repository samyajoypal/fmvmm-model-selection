"""
fmvmm.mixtures.DMM_Soft_Constrained

Constrained Dirichlet Mixture Model (Soft EM)

Implements equality constraints on selected alpha entries
via proper reparameterization inside the M-step.

Does NOT modify DMM_Soft.
Used only for valid LRT null models.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.special import gammaln
from scipy.optimize import minimize

from fmvmm.mixtures.DMM_Soft import DMM_Soft


_EPS_ALPHA = 1e-12          # alpha positivity guard
_EPS_X = 1e-300             # log(x) guard
_MAX_LOG_ALPHA = 40.0       # exp(40) ~ 2.35e17, prevents overflow


@dataclass(frozen=True)
class _TieMap:
    """
    Holds mapping for reparameterization:
    - full alpha has K*p entries (flattened)
    - free vector has q entries
    - each full index maps to one free index
    """
    K: int
    p: int
    full_to_free: np.ndarray   # shape (K*p,), int
    q: int


def _build_tie_map(K: int, p: int, equality_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> _TieMap:
    """
    Build mapping full_index -> free_index where equality constraints
    force multiple full indices to share the same free coordinate.

    We do this with a union-find on indices 0..K*p-1.
    """
    n_full = K * p

    parent = np.arange(n_full, dtype=int)
    rank = np.zeros(n_full, dtype=int)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    def idx(j: int, m: int) -> int:
        return j * p + m

    # union constraints
    for (j1, m1), (j2, m2) in equality_pairs:
        if not (0 <= j1 < K and 0 <= j2 < K and 0 <= m1 < p and 0 <= m2 < p):
            raise ValueError("Invalid equality_pairs indices.")
        union(idx(j1, m1), idx(j2, m2))

    # compress + assign free indices to each component representative
    reps = np.array([find(i) for i in range(n_full)], dtype=int)

    # map each unique representative to a free index 0..q-1
    uniq_reps, inv = np.unique(reps, return_inverse=True)
    # inv gives free index for each full entry
    full_to_free = inv.astype(int)
    q = len(uniq_reps)

    return _TieMap(K=K, p=p, full_to_free=full_to_free, q=q)


def _pack_free_from_alpha(alpha: np.ndarray, tie: _TieMap) -> np.ndarray:
    """
    alpha: (K,p)
    returns free vector (q,)
    """
    a_flat = np.asarray(alpha, dtype=float).reshape(-1)
    # For each free coordinate, pick the first occurrence in full space
    free = np.zeros(tie.q, dtype=float)
    # stable: take mean over tied positions (should already be close)
    for k in range(tie.q):
        mask = (tie.full_to_free == k)
        free[k] = float(np.mean(a_flat[mask]))
    return free


def _unpack_alpha_from_free(free: np.ndarray, tie: _TieMap) -> np.ndarray:
    """
    free: (q,)
    returns alpha: (K,p)
    """
    free = np.asarray(free, dtype=float)
    a_flat = free[tie.full_to_free]  # broadcast ties
    return a_flat.reshape(tie.K, tie.p)


def _Q_dirichlet(alpha: np.ndarray, X: np.ndarray, gamma: np.ndarray) -> float:
    """
    Expected complete-data loglikelihood Q(alpha) for Dirichlet mixture,
    given responsibilities gamma (soft assignments) and data X (compositions).

    Q(alpha) = sum_{i,k} gamma_{ik} [ log Dir(x_i | alpha_k) ]
    We ignore pi part here because we only optimize alpha in this routine.

    This matches the formulas in DMM_Soft._log_pdf_dirichlet, just vectorized.
    """
    alpha = np.asarray(alpha, dtype=float)
    X = np.asarray(X, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    K, p = alpha.shape
    N = X.shape[0]

    # guards
    alpha = np.clip(alpha, _EPS_ALPHA, None)
    logX = np.log(np.clip(X, _EPS_X, None))

    # log Dirichlet density per (i,k):
    # l_{ik} = sum_j (alpha_kj - 1) log x_ij - sum_j log Gamma(alpha_kj) + log Gamma(sum_j alpha_kj)
    # Compute in vectorized form:
    sum_alpha = np.sum(alpha, axis=1)                         # (K,)
    t3 = gammaln(sum_alpha)                                   # (K,)
    t2 = np.sum(gammaln(alpha), axis=1)                       # (K,)

    # term1: (N,K) = logX (N,p) dot (alpha-1)^T (p,K)
    term1 = logX @ (alpha - 1.0).T                            # (N,K)

    logpdf = term1 - t2[None, :] + t3[None, :]                # (N,K)

    # Q = sum_{i,k} gamma_{ik} * logpdf_{ik}
    return float(np.sum(gamma * logpdf))


class DMM_Soft_EqualityAlpha(DMM_Soft):
    """
    Dirichlet Mixture Model with equality constraints on alpha entries,
    enforced via TRUE reparameterization in the alpha M-step.

    Parameters
    ----------
    equality_pairs : list of tuples
        Each element is ((j1, m1), (j2, m2))
        meaning alpha[j1, m1] == alpha[j2, m2]
    """

    def __init__(
        self,
        n_clusters,
        equality_pairs,
        tol=1e-4,
        initialization="kmeans",
        method="meanprecision",            # kept for API compatibility; not used in constrained alpha step
        print_log_likelihood=False,
        max_iter=25,
        verbose=True,
        alpha_opt_maxiter: int = 200,
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

        self.equality_pairs = list(equality_pairs)
        self.alpha_opt_maxiter = int(alpha_opt_maxiter)

        # built after seeing p in fit
        self._tie_map: _TieMap | None = None

    # ---------------------------------------------------------
    # Constrained alpha M-step (true reparameterization)
    # ---------------------------------------------------------

    def _alpha_mstep_constrained(self, X: np.ndarray, gamma: np.ndarray, alpha_init: np.ndarray) -> np.ndarray:
        """
        Compute constrained alpha maximizing Q(alpha) under equality constraints
        via reparameterization and unconstrained optimization in log-space.

        We optimize free parameters u in R^q where alpha_free = exp(u),
        then broadcast ties to full alpha.
        """
        if self._tie_map is None:
            raise RuntimeError("Tie map not initialized. Call fit() first.")

        tie = self._tie_map

        # init from previous alpha (projected into free coords)
        free0 = _pack_free_from_alpha(alpha_init, tie)
        free0 = np.clip(free0, _EPS_ALPHA, None)

        # optimize in log-space for positivity + stability
        u0 = np.log(np.clip(free0, _EPS_ALPHA, None))
        u0 = np.clip(u0, -_MAX_LOG_ALPHA, _MAX_LOG_ALPHA)

        def obj(u: np.ndarray) -> float:
            u = np.clip(u, -_MAX_LOG_ALPHA, _MAX_LOG_ALPHA)
            free = np.exp(u)
            alpha = _unpack_alpha_from_free(free, tie)
            # negative because we minimize
            return -_Q_dirichlet(alpha, X, gamma)

        res = minimize(
            obj,
            u0,
            method="L-BFGS-B",
            options={"maxiter": self.alpha_opt_maxiter, "ftol": 1e-9},
        )

        u_hat = np.clip(res.x, -_MAX_LOG_ALPHA, _MAX_LOG_ALPHA)
        free_hat = np.exp(u_hat)
        alpha_hat = _unpack_alpha_from_free(free_hat, tie)
        alpha_hat = np.clip(alpha_hat, _EPS_ALPHA, None)
        return alpha_hat

    # ---------------------------------------------------------
    # Override _m_step
    # ---------------------------------------------------------

    def _m_step(self, gamma_matrix, X, estimate_alphas_function=None, dist_comb=None, **kwargs):
        """
        Soft M-step:
          1) pi update (same as base)
          2) alpha update by constrained Q maximization (true reparameterization)
        """
        gamma_matrix = np.asarray(gamma_matrix, dtype=float)
        X = np.asarray(X, dtype=float)

        # --- pi update (unchanged) ---
        nj = gamma_matrix.sum(axis=0) + 10.0 * np.finfo(gamma_matrix.dtype).eps
        pi_new = nj / nj.sum()

        # --- constrained alpha update ---
        alpha_init = np.asarray(self.alpha_temp, dtype=float)
        alpha_new = self._alpha_mstep_constrained(X, gamma_matrix, alpha_init)

        return pi_new, alpha_new

    # ---------------------------------------------------------
    # Override fit: set tie map + correct df
    # ---------------------------------------------------------

##    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
##        """
##        Fit constrained DMM.
##
##        We call DMM_Soft.fit but our overridden _m_step will be used.
##        """
##        # run parent's fit (will set self.p)
##        super().fit(sample, true_m=true_m, true_s=true_s, post_m_step=post_m_step)
##
##        # tie map needs K and p (now known)
##        self._tie_map = _build_tie_map(self.k, self.p, self.equality_pairs)
##
##        # IMPORTANT: adjust total parameter count for LRT df.
##        # Each independent equality reduces 1 dof, but union-find handles redundancy:
##        # full has K*p alpha params; constrained has q free alpha params.
##        # Reduction = (K*p - q).
##        full_alpha = self.k * self.p
##        q = self._tie_map.q
##        reduction = int(full_alpha - q)
##
##        self.total_parameters = (self.k - 1) + q
##
##        # keep a convenience attribute
##        self._alpha_df_reduction = reduction
##
##        return self
    def fit(self, sample, true_m=None, true_s=None, post_m_step=None):
        # preprocess once to know p
        X = self._process_data(sample)
        self.p = X.shape[1]
        self._tie_map = _build_tie_map(self.k, self.p, self.equality_pairs)

        # now run EM (our _m_step will see _tie_map)
        super().fit(X, true_m=true_m, true_s=true_s, post_m_step=post_m_step)

        # parameter count: (K-1) eta + q free alpha
        q = self._tie_map.q
        self.total_parameters = (self.k - 1) + q
        self._alpha_df_reduction = int(self.k * self.p - q)

        return self
