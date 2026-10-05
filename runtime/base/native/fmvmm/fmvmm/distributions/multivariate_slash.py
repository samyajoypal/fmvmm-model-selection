import numpy as np
from fmvmm.numerics.robust_slash import slash_logpdf
from fmvmm.mixtures import slashmix_smsn
from fmvmm.mixsmsn.gen import gen_SS_multi
from fmvmm.mixtures.skewslashmix_smsn import dmvSS, d_mixedmvSS
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix_skewslash
from fmvmm.utils.utils_dist import info_opg_from_scores, score_mat_smsn_fd


def _as_scalar(value):
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def logpdf(x,mu, sigma, nu):
    return slash_logpdf(x,mu, sigma, nu)

def pdf(x,mu, sigma, nu):
    return np.exp(logpdf(x,mu, sigma, nu))

def loglike(x,mu, sigma, nu):

    return np.sum(logpdf(x, mu, sigma, nu))

def total_params(mu, sigma, nu):
    p = len(mu)

    return p + (p*(p+1)/2) +1

def rvs(mu, sigma, nu, size = 1):
    p = len(mu)

    return gen_SS_multi(size, mu, sigma, np.zeros(p), _as_scalar(nu))

def fit(x, tol=1e-4, max_iter=25):
    model = slashmix_smsn.SlashMix(1, tol=tol, max_iter=max_iter, verbose = False)
    model.fit(x)
    _, alphas = model.get_params()

    return alphas[0][0], alphas[0][1], np.array([alphas[0][3]])

def info_mat(X,mu,sigma,nu):
    scores = score_mat(X, mu, sigma, nu)
    information, _, _ = info_opg_from_scores(scores)
    return information

def score_mat(X,mu,sigma,nu):
    return score_mat_smsn_fd(
        X,
        mu=mu,
        sigma=sigma,
        tail=np.asarray(nu, dtype=float).reshape(-1),
        tail_transforms=("positive",),
        logpdf_fun=logpdf,
    )

import numpy as np

def expand_reduced_IM_to_full_no_pi_slash_from_skewslash(IM_reduced, p, g):
    """
    Take an info matrix from a Skew-Slash model (which has shape_j blocks),
    but we actually want a pure Slash model => remove shape_j from each cluster,
    remove pi_j if g>1, and expand Sigma.

    Original Skew-Slash "reduced" parameters per cluster:
      (mu_j: p), (shape_j: p), (Sigma_j: p*(p+1)//2)
      => cluster_params_skewslash = 2*p + p*(p+1)//2

    But if shape = 0, we do *not* want those p shape partials => we remove them.

    Steps:
      1) From each cluster block, skip the shape chunk of size p.
      2) Remove pi_j (g-1 if g>1).
      3) Expand Sigma from p*(p+1)//2 => p^2
      4) Keep mu_j (p) and final param nu.

    => final dimension = g*( p + p^2 ) + 1.

    The original dimension:
      M_reduced_total = g*(2p + p*(p+1)//2) + (g-1) + 1
      i.e. Skew-Slash blocks + pi + nu.

    After removing shape => each cluster effectively has p + p*(p+1)//2,
    so total becomes g*(p + p*(p+1)//2), then remove pi => + 1 for nu,
    then expand Sigma => p^2 => final g*( p + p^2 ) + 1.

    This function does that in a single pass with a custom "duplication" matrix.
    """
    # 1) Basic dimension checks
    cluster_params_skewslash = 2*p + (p*(p+1)//2)
    M_reduced_total = g*cluster_params_skewslash + (g-1 if g>1 else 0) + 1

    if IM_reduced.shape != (M_reduced_total, M_reduced_total):
        raise ValueError(
            f"Expected IM_reduced to be {M_reduced_total}x{M_reduced_total}, "
            f"got {IM_reduced.shape}."
        )

    # 2) After removing shape_j from each cluster, we have:
    #    cluster_params_slash = p + p*(p+1)//2
    # So the dimension "before removing pi" => g*(p + p*(p+1)//2) + 1(for nu).
    # We'll define M_reduced_no_shape = g*(p + p*(p+1)//2) + (g-1 if g>1 else 0) + 1
    # but we actually remove shape_j from each cluster first (p each).
    # Then we'll remove pi if g>1. Let's do it systematically via a mask.

    # Final dimension after removing shape & pi, but before expanding Sigma:
    M_reduced_no_shape_no_pi = g*(p + (p*(p+1)//2)) + 1

    # 3) Final dimension after expanding Sigma => p^2:
    M_full_no_pi_slash = g*(p + p*p) + 1

    # We'll build a mask that:
    #  - keeps only mu_j & Sigma_j from each cluster
    #  - removes shape_j block
    #  - keeps nu
    #  - optionally removes pi block if g>1
    keep_mask = np.ones(M_reduced_total, dtype=bool)

    # (a) Identify the cluster blocks
    # Each cluster block => size = 2p + p*(p+1)//2
    # Layout within each cluster block:
    #   0..(p-1) => mu_j
    #   p..(2p-1) => shape_j
    #   2p..(2p + p*(p+1)//2 -1) => Sigma_j
    # We want to "turn off" the shape chunk p..(2p-1).

    curr_offset = 0
    for j_ in range(g):
        # shape chunk start..end
        shape_start = curr_offset + p
        shape_end   = curr_offset + 2*p
        keep_mask[shape_start:shape_end] = False

        # Then skip Sigma chunk => keep it
        # so from shape_end.. shape_end + p*(p+1)//2 => remain True

        # Move to next cluster
        curr_offset += cluster_params_skewslash

    # (b) Remove pi if g>1
    if g > 1:
        offset_pi = g*cluster_params_skewslash
        pi_indices = np.arange(offset_pi, offset_pi + (g-1))
        keep_mask[pi_indices] = False

    # The last param is nu => we keep it => do nothing special

    # Now apply mask
    IM_no_shape_no_pi = IM_reduced[keep_mask][:, keep_mask]

    if IM_no_shape_no_pi.shape[0] != M_reduced_no_shape_no_pi:
        raise ValueError(
            f"After removing shape & pi, we expected dimension {M_reduced_no_shape_no_pi}, "
            f"got {IM_no_shape_no_pi.shape}."
        )

    # 4) Build a duplication matrix for final expansion from p*(p+1)//2 => p^2
    # But remember each cluster now is: ( mu_j(p), Sigma_j( p*(p+1)//2 ) ), total = p + p*(p+1)//2
    # plus 1 param for nu at the end => dimension M_reduced_no_shape_no_pi
    D = np.zeros((M_full_no_pi_slash, M_reduced_no_shape_no_pi), dtype=float)

    offset_in  = 0
    offset_out = 0

    def copy_block_1to1(n_params):
        nonlocal offset_in, offset_out
        for kk in range(n_params):
            D[offset_out + kk, offset_in + kk] = 1.0
        offset_in  += n_params
        offset_out += n_params

    # For each cluster => [ mu_j(p), Sigma_j( p*(p+1)//2 ) ]
    for j_ in range(g):
        # mu_j => p
        copy_block_1to1(p)

        # expand Sigma => p*(p+1)//2 => p^2
        old_start = offset_in
        new_start = offset_out
        old_idx_counter = 0
        for row_ in range(p):
            for col_ in range(row_, p):
                old_idx = old_start + old_idx_counter
                old_idx_counter += 1

                new_idx1 = new_start + (row_*p + col_)
                new_idx2 = new_start + (col_*p + row_)

                D[new_idx1, old_idx] = 1.0
                if new_idx2 != new_idx1:
                    D[new_idx2, old_idx] = 1.0

        offset_in  += (p*(p+1)//2)
        offset_out += (p*p)

    # final param => nu => 1
    copy_block_1to1(1)

    # multiply
    IM_full_no_pi_slash = D @ IM_no_shape_no_pi @ D.T
    return IM_full_no_pi_slash
