# fmvmm/utils/utils_dist.py
# Utilities for GH-family (and similar) score-based information via OPG
#
# Key idea:
# - Pack constrained parameters into an unconstrained vector u.
# - Unpack u back to constrained (valid) parameters.
# - Compute per-observation score vectors (finite-difference on u).
# - Build OPG information: I = S^T S, then cov = I^{-1}, se = sqrt(diag(cov)).

from __future__ import annotations

import numpy as np


# ============================================================
# SPD parameterization via Cholesky
# ============================================================

def _pack_spd_cholesky(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sigma (SPD) -> unconstrained Cholesky params: (log diag(L), offdiag(L))."""
    Sigma = np.asarray(Sigma, float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square 2D array.")
    p = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    diag_raw = np.log(np.diag(L))
    tril = np.tril_indices(p, k=-1)
    off = L[tril]
    return diag_raw, off


def _unpack_spd_cholesky(diag_raw: np.ndarray, off: np.ndarray, p: int) -> np.ndarray:
    """Unconstrained Cholesky params -> Sigma (SPD) via L L^T."""
    diag_raw = np.asarray(diag_raw, float).reshape(p,)
    off = np.asarray(off, float).ravel()
    m = p * (p - 1) // 2
    if off.size != m:
        raise ValueError(f"Expected offdiag size {m}, got {off.size}.")

    L = np.zeros((p, p), float)
    np.fill_diagonal(L, np.exp(diag_raw))
    tril = np.tril_indices(p, k=-1)
    L[tril] = off
    return L @ L.T


# ============================================================
# Generic GH-family packing/unpacking
# Parameter order in constrained theta:
# (lmbda, chi, psi, mu, sigma, gamma)
# free controls which of (lmbda, chi, psi) are estimated.
# chi, psi are positive when free -> log-transform in u.
# ============================================================

def pack_gh_family_unconstrained(
    *,
    p: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    lmbda: float | None = None,
    chi: float | None = None,
    psi: float | None = None,
    free: tuple[str, ...] = ("lmbda", "chi", "psi"),
) -> np.ndarray:
    """
    Generic packer for GH-like families.

    u layout:
      [free mixing params..., mu(p), chol_diag_raw(p), chol_off(p(p-1)/2), gamma(p)]

    Notes
    -----
    - If "chi" in free: stores log(chi).
    - If "psi" in free: stores log(psi).
    - lmbda is stored on the real line.
    """
    mu = np.asarray(mu, float).reshape(p,)
    gamma = np.asarray(gamma, float).reshape(p,)
    sigma = np.asarray(sigma, float).reshape(p, p)

    u_parts: list[np.ndarray] = []

    # mixing params
    if "lmbda" in free:
        if lmbda is None:
            raise ValueError("lmbda must be provided if 'lmbda' is free.")
        u_parts.append(np.array([float(lmbda)], float))

    if "chi" in free:
        if chi is None:
            raise ValueError("chi must be provided if 'chi' is free.")
        u_parts.append(np.array([np.log(max(float(chi), 1e-12))], float))

    if "psi" in free:
        if psi is None:
            raise ValueError("psi must be provided if 'psi' is free.")
        u_parts.append(np.array([np.log(max(float(psi), 1e-12))], float))

    # mu
    u_parts.append(mu)

    # sigma
    diag_raw, off = _pack_spd_cholesky(sigma)
    u_parts.append(diag_raw)
    u_parts.append(off)

    # gamma
    u_parts.append(gamma)

    return np.concatenate(u_parts)


def unpack_gh_family_unconstrained(
    u: np.ndarray,
    *,
    p: int,
    fixed: dict[str, float],
    free: tuple[str, ...] = ("lmbda", "chi", "psi"),
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reverse of pack_gh_family_unconstrained.

    fixed supplies values for params not in free, e.g. {"psi": 0.0}.

    Returns
    -------
    (lmbda, chi, psi, mu, sigma, gamma)
    """
    u = np.asarray(u, float).ravel()
    idx = 0

    # defaults from fixed
    lmbda = float(fixed.get("lmbda", 0.0))
    chi = float(fixed.get("chi", 1.0))
    psi = float(fixed.get("psi", 1.0))

    if "lmbda" in free:
        lmbda = float(u[idx]); idx += 1
    if "chi" in free:
        chi = float(np.exp(u[idx])); idx += 1
    if "psi" in free:
        psi = float(np.exp(u[idx])); idx += 1

    mu = u[idx:idx + p].copy(); idx += p

    diag_raw = u[idx:idx + p].copy(); idx += p
    m = p * (p - 1) // 2
    off = u[idx:idx + m].copy(); idx += m
    sigma = _unpack_spd_cholesky(diag_raw, off, p)

    gamma = u[idx:idx + p].copy(); idx += p

    if idx != u.size:
        raise ValueError(f"Unpack consumed {idx} entries but u has {u.size}.")

    return lmbda, chi, psi, mu, sigma, gamma


# ============================================================
# Finite-difference score matrix and OPG information
# ============================================================

def score_mat_fd_unconstrained(
    X: np.ndarray,
    *,
    u_hat: np.ndarray,
    unpack_fun,
    logpdf_fun,
    p: int,
    step: float = 1e-5,
    rel_step: bool = True,
) -> np.ndarray:
    """
    Compute per-observation score matrix S (n,d) wrt unconstrained u by central differences.

    Parameters
    ----------
    X : (n,p)
    u_hat : (d,)
    unpack_fun : callable(u, p=...) -> tuple
        Returns positional parameters accepted by ``logpdf_fun``.
    logpdf_fun : callable(X, *params) -> (n,) logpdf
    step : base finite-difference step
    rel_step : if True use step * max(1, |u_k|) per coordinate

    Returns
    -------
    S : (n,d) score matrix, S[i,k] = d/d u_k log f(X_i | theta(u_hat))
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    u_hat = np.asarray(u_hat, float).ravel()
    d = u_hat.size

    S = np.zeros((n, d), float)

    if rel_step:
        h = step * np.maximum(1.0, np.abs(u_hat))
    else:
        h = np.full(d, step, float)

    for k in range(d):
        uk = h[k]
        u_plus = u_hat.copy();  u_plus[k] += uk
        u_minus = u_hat.copy(); u_minus[k] -= uk

        params_plus = unpack_fun(u_plus, p=p)
        params_minus = unpack_fun(u_minus, p=p)

        lp = logpdf_fun(X, *params_plus)
        lm = logpdf_fun(X, *params_minus)

        lp = np.asarray(lp, float).ravel()
        lm = np.asarray(lm, float).ravel()
        if lp.size != n or lm.size != n:
            raise ValueError("logpdf_fun must return shape (n,) for input X of shape (n,p).")

        S[:, k] = (lp - lm) / (2.0 * uk)

    return S


def info_opg_from_scores(
    S: np.ndarray,
    *,
    ridge: float = 1e-8,
    use_pinv: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given per-observation scores S (n,d), compute OPG info, covariance, SE.

    I = S^T S
    cov = I^{-1} (or pinv)
    se = sqrt(diag(cov))

    Returns
    -------
    I : (d,d)
    cov : (d,d)
    se : (d,)
    """
    S = np.asarray(S, float)
    I = S.T @ S
    I = 0.5 * (I + I.T)
    if ridge and ridge > 0:
        I = I + ridge * np.eye(I.shape[0])

    try:
        cov = np.linalg.pinv(I) if use_pinv else np.linalg.inv(I)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(I)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return I, cov, se


def gh_info_opg(
    X: np.ndarray,
    *,
    # constrained params
    lmbda: float,
    chi: float,
    psi: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    # family controls
    free: tuple[str, ...],
    fixed: dict[str, float],
    # functions
    logpdf_fun,
    step: float = 1e-5,
    rel_step: bool = True,
    ridge: float = 1e-8,
    use_pinv: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-call helper for GH-family component OPG.

    Returns
    -------
    I, cov, se, S, u_hat
    """
    X = np.asarray(X, float)
    p = X.shape[1]

    u_hat = pack_gh_family_unconstrained(
        p=p, lmbda=lmbda, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma, free=free
    )

    def _unpack(u, *, p):
        return unpack_gh_family_unconstrained(u, p=p, fixed=fixed, free=free)

    S = score_mat_fd_unconstrained(
        X, u_hat=u_hat, unpack_fun=_unpack, logpdf_fun=logpdf_fun, p=p, step=step, rel_step=rel_step
    )
    I, cov, se = info_opg_from_scores(S, ridge=ridge, use_pinv=use_pinv)
    return I, cov, se, S, u_hat


def pack_mvn_unconstrained(*, p: int, mu, sigma):
    """
    Pack MVN params (mu, Sigma SPD) into unconstrained vector u.
    u = [mu (p), chol_diag_raw (p), chol_off (p*(p-1)/2)]
    """
    mu = np.asarray(mu, float).reshape(p,)
    sigma = np.asarray(sigma, float).reshape(p, p)

    diag_raw, off = _pack_spd_cholesky(sigma)
    return np.concatenate([mu, diag_raw, off])


def unpack_mvn_unconstrained(u, *, p: int):
    """
    Unpack unconstrained u into (mu, Sigma).
    """
    u = np.asarray(u, float).ravel()
    idx = 0

    mu = u[idx:idx+p].copy(); idx += p
    diag_raw = u[idx:idx+p].copy(); idx += p
    m = p*(p-1)//2
    off = u[idx:idx+m].copy(); idx += m

    sigma = _unpack_spd_cholesky(diag_raw, off, p)

    if idx != u.size:
        raise ValueError(f"Unpack consumed {idx} entries but u has {u.size}.")

    return mu, sigma


def _pack_bounded(value: float, kind: str) -> float:
    value = float(value)
    if kind == "positive":
        return float(np.log(max(value, 1e-12)))
    if kind == "unit":
        value = float(np.clip(value, 1e-10, 1.0 - 1e-10))
        return float(np.log(value / (1.0 - value)))
    if kind == "real":
        return value
    raise ValueError(f"Unknown scalar transform {kind!r}.")


def _unpack_bounded(value: float, kind: str) -> float:
    value = float(value)
    if kind == "positive":
        return float(np.exp(value))
    if kind == "unit":
        if value >= 0:
            z = np.exp(-value)
            return float(1.0 / (1.0 + z))
        z = np.exp(value)
        return float(z / (1.0 + z))
    if kind == "real":
        return value
    raise ValueError(f"Unknown scalar transform {kind!r}.")


def pack_smsn_unconstrained(*, p: int, mu, sigma, shape=None, tail=None,
                            tail_transforms=()):
    """Pack SMSN parameters onto a full-rank unconstrained scale."""
    parts = [np.asarray(mu, dtype=float).reshape(p,)]
    if shape is not None:
        parts.append(np.asarray(shape, dtype=float).reshape(p,))
    diag_raw, off = _pack_spd_cholesky(np.asarray(sigma, dtype=float).reshape(p, p))
    parts.extend([diag_raw, off])

    tail_array = np.asarray([] if tail is None else tail, dtype=float).reshape(-1)
    transforms = tuple(tail_transforms)
    if tail_array.size != len(transforms):
        raise ValueError("tail and tail_transforms must have the same length.")
    if tail_array.size:
        parts.append(np.asarray([
            _pack_bounded(value, kind)
            for value, kind in zip(tail_array, transforms)
        ]))
    return np.concatenate(parts)


def unpack_smsn_unconstrained(u, *, p: int, has_shape: bool,
                              tail_transforms=()):
    """Inverse of pack_smsn_unconstrained in distribution call order."""
    u = np.asarray(u, dtype=float).reshape(-1)
    idx = 0
    mu = u[idx:idx + p].copy()
    idx += p
    shape = None
    if has_shape:
        shape = u[idx:idx + p].copy()
        idx += p
    diag_raw = u[idx:idx + p].copy()
    idx += p
    n_off = p * (p - 1) // 2
    off = u[idx:idx + n_off].copy()
    idx += n_off
    sigma = _unpack_spd_cholesky(diag_raw, off, p)

    transforms = tuple(tail_transforms)
    tail = np.asarray([
        _unpack_bounded(value, kind)
        for value, kind in zip(u[idx:idx + len(transforms)], transforms)
    ])
    idx += len(transforms)
    if idx != u.size:
        raise ValueError(f"Unpack consumed {idx} entries but u has {u.size}.")

    if has_shape:
        return (mu, sigma, shape, tail) if tail.size else (mu, sigma, shape)
    return (mu, sigma, tail) if tail.size else (mu, sigma)


def score_mat_smsn_fd(X, *, mu, sigma, logpdf_fun, shape=None, tail=None,
                      tail_transforms=(), step=1e-5):
    """Finite-difference SMSN score matrix on the fitted unconstrained scale."""
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    u_hat = pack_smsn_unconstrained(
        p=p,
        mu=mu,
        sigma=sigma,
        shape=shape,
        tail=tail,
        tail_transforms=tail_transforms,
    )
    return score_mat_fd_unconstrained(
        X,
        u_hat=u_hat,
        unpack_fun=lambda u, *, p: unpack_smsn_unconstrained(
            u,
            p=p,
            has_shape=shape is not None,
            tail_transforms=tail_transforms,
        ),
        logpdf_fun=logpdf_fun,
        p=p,
        step=step,
        rel_step=True,
    )


def score_mat_mvn_fd(X, *, u_hat, logpdf_fun, p: int, step=1e-5):
    """
    Per-observation score matrix for MVN wrt u (central differences).
    logpdf_fun(X, mu, Sigma) -> (n,) logpdf
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    u_hat = np.asarray(u_hat, float).ravel()
    d = u_hat.size
    S = np.zeros((n, d), float)

    # relative step per coordinate (more stable than fixed step)
    h = step * np.maximum(1.0, np.abs(u_hat))

    for k in range(d):
        up = u_hat.copy(); up[k] += h[k]
        um = u_hat.copy(); um[k] -= h[k]

        mu_p, sig_p = unpack_mvn_unconstrained(up, p=p)
        mu_m, sig_m = unpack_mvn_unconstrained(um, p=p)

        lp = logpdf_fun(X, mu_p, sig_p)
        lm = logpdf_fun(X, mu_m, sig_m)

        S[:, k] = (lp - lm) / (2.0 * h[k])

    return S

def score_mat_mvn_analytic(X, *, mu, sigma):
    """
    Analytical per-observation score matrix for MVN wrt u = [mu, diag_raw, off],
    where sigma = L L^T and diag_raw = log(diag(L)).

    Returns S: (n, d_u) with d_u = p + p + p(p-1)/2
    """
    X = np.asarray(X, float)
    mu = np.asarray(mu, float).reshape(-1,)
    sigma = np.asarray(sigma, float)
    n, p = X.shape
    if mu.size != p:
        raise ValueError("mu has wrong length.")

    # Build Cholesky and inverse
    L = np.linalg.cholesky(sigma)
    invS = np.linalg.inv(sigma)

    diff = X - mu[None, :]                       # (n,p)
    # score wrt mu: invS (x - mu)
    S_mu = diff @ invS.T                         # (n,p)

    # We need G(x) = 0.5*(invS * outer * invS - invS) for each obs
    # Let v = invS (x - mu) = S_mu^T for each obs => v_i is row S_mu[i]
    # invS outer invS = v v^T
    # So G_i = 0.5*(v v^T - invS)
    # Then dℓ/dL = 2 G L = (v v^T - invS) L

    # Precompute B = invS @ L once
    invS_L = invS @ L                            # (p,p)

    # For each obs, compute (v v^T) L efficiently as outer(v, v @ L)
    VL = S_mu @ L                                # (n,p) because S_mu is v^T
    # term1_i = (v v^T) L = outer(v, v^T L) => rows: v_j * (v^T L)
    # Implement: for each i, term1 = np.outer(v_i, VL_i)
    # Vectorize with broadcasting:
    term1 = S_mu[:, :, None] * VL[:, None, :]    # (n,p,p)

    # term2 = (invS) L, constant across i
    term2 = invS_L[None, :, :]                   # (1,p,p)

    dL = term1 - term2                           # (n,p,p) equals dℓ/dL

    # Map dL to unconstrained (diag_raw, off)
    # diag_raw: dℓ/dd_i = dℓ/dL_ii * L_ii
    diagL = np.diag(L)                           # (p,)
    S_diag = np.empty((n, p), float)
    for i in range(p):
        S_diag[:, i] = dL[:, i, i] * diagL[i]

    # off: lower-triangular k=-1 entries
    tril = np.tril_indices(p, k=-1)
    S_off = dL[:, tril[0], tril[1]]              # (n, p(p-1)/2)

    return np.concatenate([S_mu, S_diag, S_off], axis=1)

def info_from_scores(S, ridge=0.0):
    """
    Outer-product-of-gradients information matrix.

    Parameters
    ----------
    S : (n,d) ndarray
        Per-observation score matrix.
    ridge : float
        Optional ridge added to diagonal.

    Returns
    -------
    IM : (d,d)
    """
    I = S.T @ S
    I = 0.5 * (I + I.T)
    if ridge > 0:
        I += ridge * np.eye(I.shape[0])
    return I
