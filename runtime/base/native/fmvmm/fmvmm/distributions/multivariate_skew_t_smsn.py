import numpy as np
from fmvmm.numerics.robust_skew_t import skew_t_logpdf
from fmvmm.mixtures import skewtmix_smsn
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST
from fmvmm.mixsmsn.gen import gen_ST_multi
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix_skewt
from fmvmm.utils.utils_dist import info_opg_from_scores, score_mat_smsn_fd

def logpdf(x,mu, sigma, lmbda, nu):

    return skew_t_logpdf(x, np.asarray(mu).reshape(-1), sigma, np.asarray(lmbda).reshape(-1), nu)

def pdf(x,mu, sigma, lmbda, nu):

    return np.exp(logpdf(x,mu, sigma, lmbda, nu))

def loglike(x,mu, sigma, lmbda, nu):

    return np.sum(logpdf(x, mu, sigma, lmbda, nu))

def total_params(mu, sigma, lmbda, nu):
    p = len(mu)

    return 2*p + (p*(p+1)/2) +1

def rvs(mu, sigma, lmbda, nu, size = 1):

    return gen_ST_multi(size, mu, sigma, lmbda, nu)

def fit(x):
    model = skewtmix_smsn.SkewTMix(1, verbose = False)
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


def expand_reduced_IM_to_full_no_pi(IM_reduced, p, g):
    """
    Take a 'reduced' information matrix (including mu_j, shape_j, upper-triangular
    Sigma_j, possibly also mixture weights pi_j, and nu), then:

      1) Remove all rows/columns corresponding to pi_j (the mixture weights).
         (If g=1, there are no such rows/columns to remove.)
      2) Expand each covariance block from size p(p+1)/2 to p^2 by duplicating
         symmetric entries.
      3) Return the resulting full information matrix that has mu_j, shape_j,
         *full* Sigma_j, and nu — no mixture weights.

    Parameter layout in IM_reduced:
      - For j=1..g:
        mu_j (p), shape_j (p), Sigma_j (p*(p+1)//2)
      - If g>1 => plus (g-1) for pi
      - Finally +1 for nu
    That is:
      M_reduced_total = g*(2p + p(p+1)//2) + (g-1) + 1

    We remove the pi rows/columns, so dimension becomes:
      M_reduced_no_pi = g*(2p + p(p+1)//2) + 1

    Then we expand Sigma blocks to p^2:
      M_full_no_pi = g*(2p + p^2) + 1

    Returns
    -------
    IM_full_no_pi : (M_full_no_pi, M_full_no_pi) ndarray
    """

    # -------------------------------
    # 1) Basic dimension checks
    cluster_params_reduced = 2*p + (p*(p+1)//2)
    M_reduced_total = g*cluster_params_reduced + (g-1) + 1  # total in old matrix
    if IM_reduced.shape != (M_reduced_total, M_reduced_total):
        raise ValueError(
            f"Expected IM_reduced to be {M_reduced_total}x{M_reduced_total}, "
            f"but got {IM_reduced.shape}."
        )

    # The dimension after removing pi:
    M_reduced_no_pi = g*cluster_params_reduced + 1  # no (g-1)
    M_full_no_pi    = g*(2*p + p*p) + 1

    # -------------------------------
    # 2) Remove pi rows/columns if g>1
    #    pi_j occupy the block right after all cluster blocks:
    #    offset_pi = g*cluster_params_reduced, size = (g-1)
    offset_pi = g*cluster_params_reduced
    if g > 1:
        # Build an index mask for the rows/cols we want to keep:
        keep_mask = np.ones(M_reduced_total, dtype=bool)
        # Turn off the pi indices
        pi_indices = np.arange(offset_pi, offset_pi+(g-1), dtype=int)
        keep_mask[pi_indices] = False
        # keep all others => we also keep the last param for nu
        IM_reduced_no_pi = IM_reduced[keep_mask][:, keep_mask]
    else:
        # If g=1, there are no pi parameters
        IM_reduced_no_pi = IM_reduced

    # sanity check
    if IM_reduced_no_pi.shape != (M_reduced_no_pi, M_reduced_no_pi):
        raise ValueError(
            f"After removing pi, expect shape = {M_reduced_no_pi}x{M_reduced_no_pi} "
            f"but got {IM_reduced_no_pi.shape}"
        )

    # -------------------------------
    # 3) Build the duplication matrix D_no_pi
    #    which maps from "reduced no-pi" coords to "full no-pi" coords.
    D_no_pi = np.zeros((M_full_no_pi, M_reduced_no_pi), dtype=float)

    offset_reduced = 0
    offset_full    = 0

    def copy_block_1to1(n_params):
        """
        Write a diagonal identity sub-block of size n_params
        from offset_reduced -> offset_full.
        """
        nonlocal offset_reduced, offset_full
        for k_ in range(n_params):
            D_no_pi[offset_full + k_, offset_reduced + k_] = 1.0
        offset_reduced += n_params
        offset_full    += n_params

    # For each cluster j, we have: mu_j (p), shape_j(p), Sigma_j( p*(p+1)//2 )
    for j_ in range(g):
        # 3.1) mu_j => p
        copy_block_1to1(p)
        # 3.2) shape_j => p
        copy_block_1to1(p)

        # 3.3) expand Sigma from p*(p+1)//2 to p^2
        old_start = offset_reduced
        new_start = offset_full
        old_index = 0
        for r_ in range(p):
            for c_ in range(r_, p):
                # old param index
                old_idx = old_start + old_index
                old_index += 1

                # new param indices (two if r_ != c_)
                new_idx1 = new_start + (r_*p + c_)
                new_idx2 = new_start + (c_*p + r_)

                D_no_pi[new_idx1, old_idx] = 1.0
                if new_idx2 != new_idx1:
                    D_no_pi[new_idx2, old_idx] = 1.0

        offset_reduced += (p*(p+1)//2)
        offset_full    += (p*p)

    # We have removed pi, so no block for them.

    # 3.4) final param = nu => 1 param
    copy_block_1to1(1)

    # -------------------------------
    # 4) Multiply to get final result
    IM_full_no_pi = D_no_pi @ IM_reduced_no_pi @ D_no_pi.T
    return IM_full_no_pi
