from fmvmm.mixtures import skewlaplacemix
import numpy as np
from fmvmm.numerics.robust_gh import gh_logpdf
from fmvmm.utils.utils_dist import info_opg_from_scores, score_mat_smsn_fd
from fmvmm.mixtures.skewlaplacemix import compute_empirical_info_and_se, compute_empirical_info_se_and_scores


def logpdf(x,mu,sigma,gamma):

    return gh_logpdf(x, (len(mu)+1)/2, 0., 1., mu, sigma, gamma)

def pdf(x,mu,sigma,gamma):

    return np.exp(logpdf(x,mu,sigma,gamma))

def loglike(x,mu,sigma,gamma):

    return np.sum(logpdf(x,mu,sigma,gamma))

def total_params(mu,sigma,gamma):
    p = len(mu)

    return int(2*p + (p*(p+1)/2))

def rvs(mu, sigma, gamma, size=1, random_state=None):

    return skewlaplacemix.sample_skewlaplace(mu, sigma, gamma, size, random_state=random_state)

def mean(mu, sigma, gamma):
    """
    mu:    shape (p,)
    sigma: shape (p, p) [not actually used here]
    gamma: shape (p,)

    Returns the mean vector of dimension (p,).
    """
    p = len(mu)
    # mu + (p+1)*gamma => shape (p,)
    return mu + (p + 1) * gamma

def var(mu, sigma, gamma):
    """
    mu:    shape (p,)
    sigma: shape (p, p)
    gamma: shape (p,)

    Returns the 'variance' or second-moment matrix of dimension (p, p).
    Under the MVSL parameterization used here,
    X = mu + W gamma + sqrt(W) Z, Z ~ N(0, sigma),
    W ~ Gamma((p+1)/2, scale=2). Therefore
    Var(X) = (p+1) * (sigma + 2 gamma gamma^T).
    """
    p = len(mu)
    # Outer product of gamma with itself => shape (p,p)
    gamma_outer = np.outer(gamma, gamma)

    return (p + 1) * (sigma + 2.0 * gamma_outer)


def fit(x):
    model = skewlaplacemix.SkewLaplaceMix(1, verbose = False)
    model.fit(x)
    _, alphas = model.get_params()

    return alphas[0][0], alphas[0][1], alphas[0][2]

# def info_mat(X,mu,sigma,lmbda):
#
#     p =len(mu)
#     g = 1
#     n = X.shape[0]
#
#     IM, _ = compute_empirical_info_and_se(None, pi = [1],
#             alpha= [[mu, sigma, lmbda]],data=X,gamma_res=np.array([np.ones(n)]).T, use_model=False)
#
#     final_IM = expand_reduced_IM_to_full_no_pi(IM, p, g)
#
#     return final_IM
#
# def expand_reduced_IM_to_full_no_pi(IM_reduced, p, g):
#     """
#     Expand and remove mixture-weight parameters from the 'reduced' info matrix
#     of a Skew-Laplace mixture, as produced by info_matrix_skewnormal(...).
#
#     In info_matrix_skewnormal, each cluster j has:
#       - mu_j : p
#       - shape_j : p
#       - Sigma_j : p*(p+1)//2   (upper-triangular portion)
#     => cluster_params_reduced = 2*p + (p*(p+1)//2)
#     => total dimension:  g*cluster_params_reduced + (g-1 if g>1 else 0)
#
#     Steps:
#       1) Remove (g-1) rows/cols for pi_j if g>1.
#       2) Expand each Sigma_j from p*(p+1)//2 to p^2.
#       3) Return the expanded info matrix with dimension
#          g*(2p + p^2), i.e. only mu_j, shape_j, full Sigma_j.
#
#     Parameters
#     ----------
#     IM_reduced : ndarray of shape (M_reduced, M_reduced)
#         The reduced info matrix from info_matrix_skewnormal(...).
#     p : int
#         Dimension of each component (size of mu_j, shape_j).
#     g : int
#         Number of mixture components.
#
#     Returns
#     -------
#     IM_full_no_pi : ndarray of shape (M_full_no_pi, M_full_no_pi)
#         The expanded info matrix, with no pi_j parameters and
#         full (p^2) blocks for each Sigma_j.
#     """
#     # 1) Dimensions in the "reduced" format
#     cluster_params_reduced = 2*p + (p*(p+1)//2)
#     M_reduced_total = g*cluster_params_reduced + (g-1 if g>1 else 0)  # no extra 'nu' in pure Skew-Normal
#     if IM_reduced.shape != (M_reduced_total, M_reduced_total):
#         raise ValueError(
#             f"Expected IM_reduced to be {M_reduced_total} x {M_reduced_total}, "
#             f"but got {IM_reduced.shape}."
#         )
#
#     # 2) After removing pi block (if g>1), we have:
#     M_reduced_no_pi = g*cluster_params_reduced
#
#     # 3) After expanding Sigma to p^2 for each cluster, final dimension is:
#     #    M_full_no_pi = g*(2p + p^2).
#     M_full_no_pi = g * (2*p + p*p)
#
#     # --- Remove the pi block if g>1
#     if g > 1:
#         offset_pi = g*cluster_params_reduced  # start of pi block
#         keep_mask = np.ones(M_reduced_total, dtype=bool)
#         # The pi parameters are offset_pi : offset_pi + (g-1)
#         pi_indices = np.arange(offset_pi, offset_pi + (g-1))
#         keep_mask[pi_indices] = False
#         IM_reduced_no_pi = IM_reduced[keep_mask][:, keep_mask]
#     else:
#         # If g=1, no pi
#         IM_reduced_no_pi = IM_reduced
#
#     if IM_reduced_no_pi.shape != (M_reduced_no_pi, M_reduced_no_pi):
#         raise ValueError(
#             f"After removing pi, expected shape ({M_reduced_no_pi}, {M_reduced_no_pi}), "
#             f"but got {IM_reduced_no_pi.shape}."
#         )
#
#     # --- 4) Build duplication matrix D_no_pi
#     # shape: (M_full_no_pi, M_reduced_no_pi)
#     D_no_pi = np.zeros((M_full_no_pi, M_reduced_no_pi), dtype=float)
#
#     offset_reduced = 0
#     offset_full = 0
#
#     def copy_block_1to1(num_params):
#         # copies 'num_params' parameters in a 1-to-1 fashion
#         nonlocal offset_reduced, offset_full
#         for kk in range(num_params):
#             D_no_pi[offset_full + kk, offset_reduced + kk] = 1.0
#         offset_reduced += num_params
#         offset_full    += num_params
#
#     # For each cluster j => we have mu_j (p), shape_j (p), Sigma_j (p*(p+1)//2)
#     for j_ in range(g):
#         # mu_j => p
#         copy_block_1to1(p)
#         # shape_j => p
#         copy_block_1to1(p)
#
#         # Expand Sigma_j from p*(p+1)//2 => p^2
#         old_start = offset_reduced
#         new_start = offset_full
#         old_index = 0
#
#         for row_ in range(p):
#             for col_ in range(row_, p):
#                 old_idx = old_start + old_index
#                 old_index += 1
#                 new_idx1 = new_start + (row_*p + col_)
#                 new_idx2 = new_start + (col_*p + row_)
#
#                 D_no_pi[new_idx1, old_idx] = 1.0
#                 if new_idx2 != new_idx1:
#                     D_no_pi[new_idx2, old_idx] = 1.0
#
#         offset_reduced += (p*(p+1)//2)
#         offset_full    += (p*p)
#
#     # --- 5) Final expanded info matrix
#     IM_full_no_pi = D_no_pi @ IM_reduced_no_pi @ D_no_pi.T
#     return IM_full_no_pi

def score_mat(X, mu, sigma, gamma):
    return score_mat_smsn_fd(
        X,
        mu=mu,
        sigma=sigma,
        shape=gamma,
        logpdf_fun=logpdf,
    )

def info_mat(X, mu, sigma, gamma):
    scores = score_mat(X, mu, sigma, gamma)
    information, _, _ = info_opg_from_scores(scores)
    return information
