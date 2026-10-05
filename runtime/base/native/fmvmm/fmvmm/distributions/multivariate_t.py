"""
Most of the codes are used from git repository mvem. Huge thanks to the
devoloper.

"""

import numpy as np
from fmvmm.numerics.robust_mvt import student_t_logpdf
from fmvmm.utils.utils_dist import info_opg_from_scores, score_mat_smsn_fd
import scipy.stats
from scipy.special import digamma, gammaln
import scipy
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix_t

def pdf(x, loc, shape, df, allow_singular=True):
    """
    Probability density function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param allow_singular: Whether to allow a singular matrix.
    :type allow_singular: bool, optional.
    :return: The density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    return np.exp(logpdf(x, loc, shape, df))

def logpdf(x, loc, shape, df,):
    """
    Log-probability density function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param allow_singular: Whether to allow a singular matrix.
    :type allow_singular: bool, optional.
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    return student_t_logpdf(x, loc, shape, df)

def loglike(x, loc, shape, df, allow_singular=True):
    """
    Log-likelihood function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param allow_singular: Whether to allow a singular matrix.
    :type allow_singular: bool, optional.
    :return: The log-likelihood for given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, loc, shape, df))

def total_params(loc, shape, df):
    p = len(loc)

    return 1 + p + (p*(p+1)/2)

def rvs(loc, shape, df, size=1, random_state=None):
    """
    Random number generator of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :param random_state: Used for drawing random variates. Defaults to None.
    :type random_state: None, int, np.random.RandomState, np.random.Generator, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    return scipy.stats.multivariate_t.rvs(loc, shape, df, size, random_state)

def mean(loc, shape, df):
    """
    Mean of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    assert df > 1, "mean of mv student's t is not defined for df <= 1"
    return loc

def var(loc, shape, df):
    """
    Variance of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    assert df > 2, "variance of mv student's t is not defined for df <= 2"
    return (df / (df - 2)) * shape

def fit(X, maxiter = 100, ptol = 1e-6, ftol = 1e-8, return_loglike = False):
    """
    Fit the parameters of the multivariate Student's t-distribution to data
    using an EM algorithm. We use the location-scale parameterisation.

    :param X: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type X: np.ndarray
    :param maxiter: The maximum number of iterations to use in the EM algorithm.
        Defaults to 100.
    :type nit: int, optional
    :param ptol: The relative convergence criterion for the estimated
        parameters. Defaults to 1e-6.
    :type ptol: float, optional
    :param ftol: The relative convergence criterion for the log-likelihood
        function. Defaults to np.inf.
    :type ftol: float, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration.
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<array> mu, <array> scale, <float> df). Also returns
        a list of log-likelihood values at each iteration of the EM algorithm if
        ``return_loglike=True``.
    :rtype: tuple
    """

    # EM Student's t:

    N, D = X.shape
    assert N > D, "Must have more observations than dimensions in _x_"

    # initialise values:
    nu = 4
    mu = np.mean(X, axis = 0)
    sigma = np.linalg.inv((nu / (nu - 2)) * np.cov(X.T))
    log_likelihood = np.array([np.sum(logpdf(X, mu, np.linalg.inv(sigma), nu))])

    for i in range(maxiter):

        # save old values
        nu_old = nu
        mu_old = mu
        sigma_old = sigma

        # Expectation Step
        X_ = X - mu
        delta2 = np.sum((X_ @ sigma) * X_, axis=1)
        E_tau = (nu + D) / (nu + delta2)
        E_logtau = digamma(0.5 * (nu + D)) - np.log(0.5 * (nu + delta2))

        # Maximization Step
        mu = (E_tau @ X) / np.sum(E_tau)
        alpha = X - mu
        sigma = np.linalg.inv((alpha * E_tau[:, np.newaxis]).T @ alpha / N)

        # ... if method = "EM"
        func = lambda x: digamma(x/2.) - np.log(x/2.) - 1 - np.mean(E_logtau) + np.mean(E_tau)
        nu = scipy.optimize.fsolve(func, nu) # set min, max ???

        # check for converagence (stopping criterion)
        have_params_converged = \
            np.all(np.abs(mu - mu_old) <= .5 * ptol * (abs(mu) + abs(mu_old))) & \
            np.all(np.abs(nu - nu_old) <= .5 * ptol * (abs(nu) + abs(nu_old))) & \
            np.all(np.abs(sigma - sigma_old) <= .5 * ptol * (abs(sigma) + abs(sigma_old)))
        if ftol < np.inf:
            ll = np.sum(logpdf(X, mu, np.linalg.inv(sigma), nu))
            has_fun_converged = np.abs(ll - log_likelihood[-1]) <= .5 * ftol * (
                np.abs(ll) + np.abs(log_likelihood[-1]))
            log_likelihood = np.append(log_likelihood, ll)
        else:
            has_fun_converged = True
        if have_params_converged and has_fun_converged: break

    sigma = np.linalg.inv(sigma)

    if return_loglike:
        return mu, sigma, nu, log_likelihood

    return mu, sigma, nu

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


def expand_reduced_IM_to_full_no_pi(IM_reduced, p, g):
    """
    For a Student-t mixture with:
      - cluster_params_reduced = p + (p*(p+1)//2)
          (that is, each cluster has 'mu_j' of length p,
           and Sigma_j of length p*(p+1)//2 for upper-triangular)
      - total dimension = g * cluster_params_reduced + (g-1 if g>1 else 0) + 1
          [the (g-1) for mixture weights if g>1, plus 1 for nu]

    This function:
      1) Removes the rows/columns corresponding to the pi_j (mixture weights),
         if g>1 (if g=1, there are none).
      2) Expands each Sigma_j block to full p^2 by duplicating symmetric entries.
      3) Returns the expanded information matrix, containing mu_j, full Sigma_j,
         and nu, but no pi_j parameters.

    The final dimension is:
      g * (p + p^2) + 1
        => for each cluster: p for mu_j, p^2 for Sigma_j, plus 1 for nu.
    """
    # --- 1) Check dimension of the reduced IM
    cluster_params_reduced = p + (p*(p+1)//2)
    M_reduced_total = g*cluster_params_reduced + (g-1) + 1  # (g-1) mixture weights, 1 for nu
    if IM_reduced.shape != (M_reduced_total, M_reduced_total):
        raise ValueError(
            f"Expected IM_reduced to be {M_reduced_total}x{M_reduced_total}, "
            f"got {IM_reduced.shape}."
        )

    # Dimension after removing pi (if g>1):
    M_reduced_no_pi = g*cluster_params_reduced + 1  # no (g-1)
    # Then the final dimension after expanding Sigma_j to p^2:
    M_full_no_pi = g*(p + p*p) + 1  # each cluster: p + (p^2), plus 1 for nu

    # --- 2) Remove mixture-weight rows/cols (if g>1)
    offset_pi = g*cluster_params_reduced
    if g > 1:
        keep_mask = np.ones(M_reduced_total, dtype=bool)
        # The pi parameters occupy the block [offset_pi : offset_pi + (g-1)]
        pi_indices = np.arange(offset_pi, offset_pi+(g-1))
        keep_mask[pi_indices] = False
        IM_reduced_no_pi = IM_reduced[keep_mask][:, keep_mask]
    else:
        # If g=1, no pi to remove
        IM_reduced_no_pi = IM_reduced

    if IM_reduced_no_pi.shape != (M_reduced_no_pi, M_reduced_no_pi):
        raise ValueError(
            "After removing pi, expected shape "
            f"{(M_reduced_no_pi, M_reduced_no_pi)} but got {IM_reduced_no_pi.shape}"
        )

    # --- 3) Build duplication matrix (D_no_pi) of shape (M_full_no_pi x M_reduced_no_pi)
    # We map from "reduced" param indexing to "full" param indexing.
    D_no_pi = np.zeros((M_full_no_pi, M_reduced_no_pi), dtype=float)

    # We copy mu_j directly (1-to-1) => p parameters,
    # Sigma_j => expand from p(p+1)//2 to p^2.
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
        # 3.1) mu_j => length p
        copy_block_1to1(p)

        # 3.2) Sigma_j => from p*(p+1)//2 to p^2
        old_start = offset_reduced
        new_start = offset_full
        old_index = 0

        for r_ in range(p):
            for c_ in range(r_, p):
                old_idx = old_start + old_index
                old_index += 1

                new_idx1 = new_start + (r_*p + c_)
                new_idx2 = new_start + (c_*p + r_)

                D_no_pi[new_idx1, old_idx] = 1.0
                if new_idx2 != new_idx1:
                    D_no_pi[new_idx2, old_idx] = 1.0

        offset_reduced += (p*(p+1)//2)
        offset_full    += (p*p)

    # 3.3) We removed pi. So no block for them here.

    # 3.4) final param = nu => 1 param
    copy_block_1to1(1)

    # --- 4) Multiply to get final result:  IM_full_no_pi = D_no_pi * IM_reduced_no_pi * D_no_pi^T
    IM_full_no_pi = D_no_pi @ IM_reduced_no_pi @ D_no_pi.T
    return IM_full_no_pi
