import numpy as np
from fmvmm.numerics.robust_probability import skew_contaminated_normal_logpdf
from fmvmm.mixtures import skewcontmix_smsn
from fmvmm.mixsmsn.gen import gen_SCN_multi
from fmvmm.mixtures.skewcontmix_smsn import dmvnorm, dmvSNC, d_mixedmvSNC
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix_skewcn
from fmvmm.utils.utils_dist import info_opg_from_scores, score_mat_smsn_fd

def logpdf(x,mu, sigma, lmbda, nu):

    return skew_contaminated_normal_logpdf(
        x, np.asarray(mu).reshape(-1), sigma, np.asarray(lmbda).reshape(-1), nu)

def pdf(x,mu, sigma, lmbda, nu):

    return np.exp(logpdf(x, mu, sigma, lmbda, nu))

def loglike(x,mu, sigma, lmbda, nu):

    return np.sum(logpdf(x, mu, sigma, lmbda, nu))

def total_params(mu, sigma, lmbda, nu):
    p = len(mu)

    return 2*p + (p*(p+1)/2) +2

def rvs(mu, sigma, lmbda, nu, size = 1):

    return gen_SCN_multi(size, mu, sigma, lmbda, nu)

def fit(x):
    model = skewcontmix_smsn.SkewContMix(1, verbose = False)
    model.fit(x)
    _, alphas = model.get_params()

    return alphas[0][0], alphas[0][1], alphas[0][2], alphas[0][3]


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
        tail_transforms=("unit", "unit"),
        logpdf_fun=logpdf,
    )

def expand_reduced_IM_to_full_no_pi(IM_reduced, p, g):
    """
    For a Skew-Contaminated Normal mixture, each cluster j has:
        - mu_j: length p
        - shape_j: length p
        - Sigma_j: length p*(p+1)//2 (upper triangular)
      => 'cluster_params_reduced' = 2*p + p*(p+1)//2

    The total dimension (in the reduced representation) is:
        M_reduced_total = g * cluster_params_reduced + (g - 1 if g>1 else 0) + 2
      because there are (g-1) mixture weights pi_j if g>1, and 2 parameters (nu1, nu2).

    This function:
      1) Removes the (g-1) pi_j rows/cols (if g>1).
      2) Expands each Sigma_j block from size p*(p+1)//2 to p^2.
      3) Leaves mu_j, shape_j, and the final 2 parameters (nu1, nu2) alone.

    The final dimension = g*(2p + p^2) + 2,
      where each cluster block is now 2p + p^2 (full Sigma),
      plus 2 for (nu1, nu2).

    Parameters
    ----------
    IM_reduced : (M_reduced_total, M_reduced_total) np.ndarray
        The info matrix from info_matrix_skewcn(), storing upper-triangular Sigmas
        and including (g-1) mixture weights and 2 contamination parameters.
    p : int
    g : int

    Returns
    -------
    IM_full_no_pi : (M_full_no_pi, M_full_no_pi) np.ndarray
        The expanded info matrix without pi_j parameters,
        with full p^2 blocks for each Sigma_j, and 2 parameters for (nu1, nu2).
    """
    # 1) Basic dimensions
    cluster_params_reduced = 2*p + (p*(p+1)//2)       # mu_j(p) + shape_j(p) + half(Sigma_j)
    M_reduced_total = g*cluster_params_reduced + (g-1 if g>1 else 0) + 2  # +2 for (nu1, nu2)
    if IM_reduced.shape != (M_reduced_total, M_reduced_total):
        raise ValueError(
            f"Expected IM_reduced to be {M_reduced_total} x {M_reduced_total}, "
            f"got {IM_reduced.shape} instead."
        )

    # After removing pi (if g>1), dimension is:
    M_reduced_no_pi = g*cluster_params_reduced + 2  # no (g-1), but keep 2 for (nu1, nu2)

    # Then, after expanding Sigma blocks to full p^2:
    # each cluster block becomes 2*p + p^2
    # total is:
    M_full_no_pi = g*(2*p + p*p) + 2

    # 2) Remove the pi rows/cols if g>1
    offset_pi = g*cluster_params_reduced  # where pi-block starts
    if g > 1:
        keep_mask = np.ones(M_reduced_total, dtype=bool)
        pi_indices = np.arange(offset_pi, offset_pi + (g-1))
        keep_mask[pi_indices] = False
        # Keep the last 2 for (nu1, nu2)
        IM_reduced_no_pi = IM_reduced[keep_mask][:, keep_mask]
    else:
        # If g=1, no pi to remove
        IM_reduced_no_pi = IM_reduced

    # Check shape
    if IM_reduced_no_pi.shape != (M_reduced_no_pi, M_reduced_no_pi):
        raise ValueError(
            f"After removing pi, expected shape ({M_reduced_no_pi}, {M_reduced_no_pi}), "
            f"got {IM_reduced_no_pi.shape} instead."
        )

    # 3) Build duplication matrix D_no_pi, shape = (M_full_no_pi, M_reduced_no_pi)
    D_no_pi = np.zeros((M_full_no_pi, M_reduced_no_pi), dtype=float)

    # We'll copy mu_j, shape_j each with dimension p => direct
    # Then expand Sigma_j => from p*(p+1)//2 to p^2
    offset_reduced = 0
    offset_full    = 0

    def copy_block_1to1(n_params):
        nonlocal offset_reduced, offset_full
        for k_ in range(n_params):
            D_no_pi[offset_full + k_, offset_reduced + k_] = 1.0
        offset_reduced += n_params
        offset_full    += n_params

    # For each cluster j:
    for j_ in range(g):
        # mu_j => length p
        copy_block_1to1(p)
        # shape_j => length p
        copy_block_1to1(p)

        # Sigma_j => expand from p*(p+1)//2 to p^2
        old_start = offset_reduced
        new_start = offset_full
        old_index = 0
        for row_ in range(p):
            for col_ in range(row_, p):
                old_idx = old_start + old_index
                old_index += 1

                new_idx1 = new_start + (row_ * p + col_)
                new_idx2 = new_start + (col_ * p + row_)

                D_no_pi[new_idx1, old_idx] = 1.0
                if new_idx2 != new_idx1:
                    D_no_pi[new_idx2, old_idx] = 1.0

        offset_reduced += (p*(p+1)//2)
        offset_full    += (p*p)

    # 4) Finally, we have 2 parameters for (nu1, nu2) => direct copy
    copy_block_1to1(2)

    # 5) Multiply to get final IM
    IM_full_no_pi = D_no_pi @ IM_reduced_no_pi @ D_no_pi.T
    return IM_full_no_pi
