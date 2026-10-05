import numpy as np
from fmvmm.numerics.robust_slash import skew_slash_logpdf
from fmvmm.mixtures import skewslashmix_smsn
from fmvmm.mixsmsn.gen import gen_SS_multi
from fmvmm.mixtures.skewslashmix_smsn import dmvSS, d_mixedmvSS
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix_skewslash
from fmvmm.utils.utils_dist import info_opg_from_scores, score_mat_smsn_fd

def _as_scalar(value):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def logpdf(x,mu, sigma, lmbda, nu):
    return skew_slash_logpdf(x,mu, sigma, lmbda, nu)

def pdf(x,mu, sigma, lmbda, nu):
    return np.exp(logpdf(x,mu, sigma, lmbda, nu))

def loglike(x,mu, sigma, lmbda, nu):

    return np.sum(logpdf(x, mu, sigma, lmbda, nu))

def total_params(mu, sigma, lmbda, nu):
    p = len(mu)

    return 2*p + (p*(p+1)/2) +1

def rvs(mu, sigma, lmbda, nu, size = 1):

    return gen_SS_multi(size, mu, sigma, lmbda, _as_scalar(nu))

def fit(x, tol=1e-4, max_iter=25):
    model = skewslashmix_smsn.SkewSlashMix(1, tol=tol, max_iter=max_iter, verbose = False)
    model.fit(x)
    _, alphas = model.get_params()

    return alphas[0][0], alphas[0][1], alphas[0][2], np.array([alphas[0][3]])

def info_mat(X,mu,sigma,lmbda,nu):
    scores = score_mat(X, mu, sigma, lmbda, nu)
    information, _, _ = info_opg_from_scores(scores)
    return information

def score_mat(X,mu,sigma,lmbda,nu):
    return score_mat_smsn_fd(
        X,
        mu=mu,
        sigma=sigma,
        shape=lmbda,
        tail=np.asarray(nu, dtype=float).reshape(-1),
        tail_transforms=("positive",),
        logpdf_fun=logpdf,
    )

def expand_reduced_IM_to_full_no_pi_skewslash(IM_reduced, p, g):
    """
    For a Skew-Slash mixture, each cluster j has 'reduced' parameters:
      - mu_j (p)
      - shape_j (p)
      - Sigma_j (p*(p+1)//2)
    => cluster_params_reduced = 2*p + p*(p+1)//2

    Then, if g>1 => (g-1) mixture weights pi_j,
    plus 1 param for nu.

    => total dimension of IM_reduced:
         M_reduced_total = g*(2p + p*(p+1)//2) + (g-1) + 1

    This function:
      1) Removes the pi_j block if g>1 (the (g-1) rows/cols).
      2) Expands each Sigma_j from p*(p+1)//2 to p^2 by duplicating symmetric entries.
      3) Leaves mu_j, shape_j, and the final param nu in place.

    The final dimension is:
      M_full_no_pi = g*(2p + p^2) + 1

    Returns
    -------
    IM_full_no_pi : (M_full_no_pi, M_full_no_pi) np.ndarray
        The expanded info matrix with no pi_j parameters
        and full (p^2) blocks for each Sigma_j.
    """
    cluster_params_reduced = 2*p + (p*(p+1)//2)
    # total dimension in the original "reduced" matrix
    M_reduced_total = g*cluster_params_reduced + (g-1 if g>1 else 0) + 1

    if IM_reduced.shape != (M_reduced_total, M_reduced_total):
        raise ValueError(
            f"Expected IM_reduced to be {M_reduced_total} x {M_reduced_total}, "
            f"but got {IM_reduced.shape}."
        )

    # After removing pi => dimension is
    M_reduced_no_pi = g*cluster_params_reduced + 1

    # After expanding Sigma_j => dimension is
    M_full_no_pi = g*(2*p + p*p) + 1

    # 1) Remove pi block if g>1
    if g > 1:
        offset_pi = g*cluster_params_reduced
        keep_mask = np.ones(M_reduced_total, dtype=bool)
        # pi_j are offset_pi : offset_pi + (g-1)
        pi_indices = np.arange(offset_pi, offset_pi + (g-1))
        keep_mask[pi_indices] = False
        IM_reduced_no_pi = IM_reduced[keep_mask][:, keep_mask]
    else:
        IM_reduced_no_pi = IM_reduced

    if IM_reduced_no_pi.shape != (M_reduced_no_pi, M_reduced_no_pi):
        raise ValueError(
            f"After removing pi, expected shape ({M_reduced_no_pi}, {M_reduced_no_pi}), "
            f"got {IM_reduced_no_pi.shape}"
        )

    # 2) Build duplication matrix D_no_pi => shape (M_full_no_pi, M_reduced_no_pi)
    D_no_pi = np.zeros((M_full_no_pi, M_reduced_no_pi), dtype=float)

    offset_reduced = 0
    offset_full    = 0

    def copy_block_1to1(n_params):
        nonlocal offset_reduced, offset_full
        for k_ in range(n_params):
            D_no_pi[offset_full + k_, offset_reduced + k_] = 1.0
        offset_reduced += n_params
        offset_full    += n_params

    # For each cluster: mu_j(p), shape_j(p), Sigma_j(p*(p+1)//2)
    for j_ in range(g):
        # mu_j => p
        copy_block_1to1(p)
        # shape_j => p
        copy_block_1to1(p)

        # expand Sigma => from p*(p+1)//2 to p^2
        old_start = offset_reduced
        new_start = offset_full
        old_index = 0

        for row_ in range(p):
            for col_ in range(row_, p):
                old_idx = old_start + old_index
                old_index += 1

                new_idx1 = new_start + (row_*p + col_)
                new_idx2 = new_start + (col_*p + row_)

                D_no_pi[new_idx1, old_idx] = 1.0
                if new_idx2 != new_idx1:
                    D_no_pi[new_idx2, old_idx] = 1.0

        offset_reduced += (p*(p+1)//2)
        offset_full    += (p*p)

    # Finally, 1 param for nu
    copy_block_1to1(1)

    IM_full_no_pi = D_no_pi @ IM_reduced_no_pi @ D_no_pi.T
    return IM_full_no_pi
