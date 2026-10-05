"""
Most of the codes are used from git repository mvem. Huge thanks to the
devoloper.

"""

import numpy as np
from fmvmm.numerics.robust_probability import skew_normal_logpdf
from fmvmm.utils.utils_dist import info_opg_from_scores, score_mat_smsn_fd
from scipy.stats import (multivariate_normal as mvn, norm)
from scipy.stats._multivariate import _squeeze_output
import scipy
from fmvmm.mixtures.skewnormmix_smsn import d_mixedmvSN, dmvSN
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix_skewnormal

def make_matrix_non_singular(sigma_matrix, epsilon=1e-6):
    """
    Ensure that a matrix is non-singular by adding a small diagonal perturbation.

    Parameters:
    - sigma_matrix: The input matrix to make non-singular.
    - epsilon: A small positive value to add to the diagonal elements.

    Returns:
    - A non-singular matrix.
    """
    # Check if the matrix is singular
    if np.linalg.matrix_rank(sigma_matrix) < min(sigma_matrix.shape):
        # The matrix is singular, let's make it non-singular
        sigma_matrix += np.eye(sigma_matrix.shape[0]) * epsilon

    return sigma_matrix

def _check_params(mu, sigma, lmbda):
    """
    Internal method to check if passed in parameters are valid.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: Tuple of the parameters as np.ndarrays, if no assertion error
        occurs.
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    lmbda = np.asarray(lmbda)

    # reshape mu, lmbda to (d,1)
    if len(mu.shape) == 1: mu = mu[:, None]
    if len(lmbda.shape) == 1: lmbda = lmbda[:, None]

    # assert shapes (d,1), (d,d), (d,1)
    d00, d01 = mu.shape
    d10, d11 = lmbda.shape
    d20, d21 = sigma.shape
    assert d00==d10==d20, "mu, sigma, lmbda must have same dimension d"
    assert d20==d21, "sigma must be qudratic with shape dxd"
    assert d01==1 and d11==1, "second dimension of mu, lmbda must be 1"

    return mu, sigma, lmbda

def mean(mu, sigma, lmbda):
    """
    Mean function of the multivariate skew normal distribution.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    mu, sigma, lmbda = _check_params(mu, sigma, lmbda)

    val, vec = np.linalg.eig(sigma)
    sigma_half = vec @ np.diag(val**(1/2)) @ vec.T
    delta = lmbda / np.sqrt(1 + np.sum(lmbda**2))
    return mu + np.sqrt(2/np.pi) * sigma_half @ delta

def var(mu, sigma, lmbda):
    """
    Variance function of the multivariate skew normal distribution.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    mu, sigma, lmbda = _check_params(mu, sigma, lmbda)

    val, vec = np.linalg.eig(sigma)
    sigma_half = vec @ np.diag(val**(1/2)) @ vec.T
    delta = lmbda / np.sqrt(1 + np.sum(lmbda**2))
    return sigma - (2/np.pi) * sigma_half @ delta @ delta.T @ sigma_half

def logpdf(x, mu, sigma, lmbda):
    """
    Log-probability density function of the multivariate skew normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    return skew_normal_logpdf(x, np.asarray(mu).reshape(-1), sigma,
                              np.asarray(lmbda).reshape(-1))

def pdf(x, mu, sigma, lmbda):
    """
    Probability density function of the multivariate skew normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    return np.exp(logpdf(x, mu, sigma, lmbda))

def loglike(x, mu, sigma, lmbda):
    """
    Log-likelihood function of the multivariate skew normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The log-likelihood for given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, mu, sigma, lmbda))

def total_params(mu, sigma, lmbda):
    p = len(mu)

    return 2*p + (p*(p+1)/2)

def rvs(mu, sigma, lmbda, size=1):
    """
    Random number generator of the multivariate skew normal distribution.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    mu, sigma, lmbda = _check_params(mu, sigma, lmbda)

    # switch to (Omega, alpha) parameterization
    cov = sigma
    val, vec = np.linalg.eig(cov)
    cov_half = vec @ np.diag(val**(-1/2)) @ vec.T
    shape = cov_half @ lmbda

    aCa      = shape.T @ cov @ shape
    delta    = (1 / np.sqrt(1 + aCa)) * cov @ shape
    cov_star = np.block([[np.ones(1),     delta.T],
                        [delta, cov]])
    x        = mvn(np.zeros(len(shape)+1), cov_star).rvs(size)
    if size == 1: x = x[None, :]
    x0, x1   = x[:, 0], x[:, 1:]
    inds     = x0 <= 0
    x1[inds] = -1 * x1[inds]
    return x1 + mu.T


def fit(x, maxiter = 100, ptol = 1e-6, ftol = 1e-10, eps=0.9, return_loglike = False):
    """
    Estimate the parameters of the multivariate skew normal distribution
    using an EM algorithm.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param maxiter: The maximum number of iterations to use in the EM algorithm.
        Defaults to 100.
    :type maxiter: int, optional
    :param ptol: The relative convergence criterion for the estimated
        parameters. Defaults to 1e-6.
    :type ptol: float, optional
    :param ftol: The relative convergence criterion for the log-likelihood
        function. Defaults to np.inf.
    :type ftol: float, optional
    :param eps: Initialization shrinkage of delta parameter. Defaults to 0.9.
    :type eps: float, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration.
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<array> mu, <array> scale, <array> nu). Also returns
        a list of log-likelihood values at each iteration of the EM algorithm if
        ``return_loglike=True``.
    :rtype: tuple
    """

    def _ll(xi, n, omega_inv_half, delta):

        d = xi.shape[1]
        lmbda = delta / np.sqrt(1 - np.sum(delta**2))
        wx = xi @ omega_inv_half
        cdf = scipy.stats.norm.cdf(wx @ lmbda)
        pdf = scipy.stats.multivariate_normal.pdf(wx, np.zeros(d), np.eye(d))
        ww = n * np.log(2) + np.sum(np.log(cdf)) + np.sum(np.log(pdf)) + n*np.log(np.linalg.det(omega_inv_half))
        return ww

    itereps = 10e-6
    n, p = x.shape

    # get pseudo moment estiators???
    m1 = np.mean(x, axis=0)
    center = x - m1
    m3 = np.mean(center**3, axis=0)
    a1 = np.sqrt(2/np.pi)
    b1 = (4/np.pi-1)*a1
    gam = np.sign(m3) * (np.abs(m3)/b1)**(1/3)

    mu0 = m1 - a1 * gam
    omega0 = (center.T @ center) / n + a1 * (gam[:, None] @ gam[None, :])

    val, vec = np.linalg.eig(omega0)
    omega0_inv_half = vec @ np.diag(val**(-1/2)) @ vec.T
    omega0_half = vec @ np.diag(val**(1/2)) @ vec.T

    delta0 = omega0_inv_half @ gam


    if (np.sum(delta0**2) > 1):
        delta0 = eps * delta0 / np.sqrt(np.sum(delta0**2))

    tau0 = 1
    mu1 = mu0
    omega1 = omega0
    delta1 = delta0

    ll_val = np.array([_ll(x - mu0, n, omega0_inv_half, delta0)])

    for i in range(maxiter):

        # E-step
        xi0 = x - mu0
        lmbda0 = delta0 / np.sqrt(1 - np.sum(delta0**2))
        w1 = 1 / np.sqrt(1 + np.sum(lmbda0**2))
        wa = xi0 @ (omega0_inv_half @ lmbda0)
        pdf = scipy.stats.norm.pdf(wa)
        cdf = scipy.stats.norm.cdf(wa)

        yexabs = tau0 * w1 * (pdf/cdf + wa)
        yex2   = tau0**2 * w1**2 * wa * (pdf/cdf + wa + 1/wa)

        # M-step
        mu1 = m1 - (omega0_half @ delta0)/tau0 * np.mean(yexabs)
        tau1 = np.mean(yex2)**(1/2)
        omega1 = (x-mu1).T @ (x-mu1) / n
        val, vec = np.linalg.eig(omega1)
        omega1_inv_half = vec @ np.diag(val**(-1/2)) @ vec.T
        omega1_half = vec @ np.diag(val**(1/2)) @ vec.T
        delta1 = omega1_inv_half @ (yexabs @ (x-mu1))/(n*tau1)


        # measure log likelihood
        ll_val = np.append(ll_val, _ll(x-mu1, n, omega1_inv_half, delta1))

        # check for converagence (stopping criterion)
        #have_params_converged = \
        #    np.all(np.abs(mu1 - mu0) <= .5 * ptol * (abs(mu1) + abs(mu0))) & \
        #    np.all(np.abs(omega1 - omega0) <= .5 * ptol * (abs(omega1) + abs(omega0))) & \
        #    np.all(np.abs(delta1 - delta0) <= .5 * ptol * (abs(delta1) + abs(delta0)))
        has_fun_converged = np.abs(ll_val[-1] - ll_val[-2]) <= .5 * ftol * ( np.abs(ll_val[-1]) + np.abs(ll_val[-2]))
        if has_fun_converged: break

        # update old parameters
        mu0 = mu1
        omega0_inv_half = omega1_inv_half
        omega0_half = omega1_half
        delta0 = delta1
        tau0 = tau1

    #lmbda1 = delta0 / np.sqrt(1-np.sum(delta0**2))
    #val, vec = np.linalg.eig(omega1)
    #inv_half = vec @ np.diag(val**(-1)) @ vec.T
    #lmbda0 = inv_half @ delta1 / np.sqrt(1 - np.sum(delta1**2))
    lmbda0 = delta1 / np.sqrt(1 - np.sum(delta1**2))
    if return_loglike:
        return mu1, omega1, lmbda0, ll_val
    return mu1, omega1, lmbda0

def info_mat(X,mu,sigma,lmbda):
    scores = score_mat(X, mu, sigma, lmbda)
    information, _, _ = info_opg_from_scores(scores)
    return information

def score_mat(X,mu,sigma,lmbda):
    return score_mat_smsn_fd(
        X,
        mu=mu,
        sigma=sigma,
        shape=lmbda,
        logpdf_fun=logpdf,
    )

def expand_reduced_IM_to_full_no_pi(IM_reduced, p, g):
    """
    Expand and remove mixture-weight parameters from the 'reduced' info matrix
    of a Skew-Normal mixture, as produced by info_matrix_skewnormal(...).

    In info_matrix_skewnormal, each cluster j has:
      - mu_j : p
      - shape_j : p
      - Sigma_j : p*(p+1)//2   (upper-triangular portion)
    => cluster_params_reduced = 2*p + (p*(p+1)//2)
    => total dimension:  g*cluster_params_reduced + (g-1 if g>1 else 0)

    Steps:
      1) Remove (g-1) rows/cols for pi_j if g>1.
      2) Expand each Sigma_j from p*(p+1)//2 to p^2.
      3) Return the expanded info matrix with dimension
         g*(2p + p^2), i.e. only mu_j, shape_j, full Sigma_j.

    Parameters
    ----------
    IM_reduced : ndarray of shape (M_reduced, M_reduced)
        The reduced info matrix from info_matrix_skewnormal(...).
    p : int
        Dimension of each component (size of mu_j, shape_j).
    g : int
        Number of mixture components.

    Returns
    -------
    IM_full_no_pi : ndarray of shape (M_full_no_pi, M_full_no_pi)
        The expanded info matrix, with no pi_j parameters and
        full (p^2) blocks for each Sigma_j.
    """
    # 1) Dimensions in the "reduced" format
    cluster_params_reduced = 2*p + (p*(p+1)//2)
    M_reduced_total = g*cluster_params_reduced + (g-1 if g>1 else 0)  # no extra 'nu' in pure Skew-Normal
    if IM_reduced.shape != (M_reduced_total, M_reduced_total):
        raise ValueError(
            f"Expected IM_reduced to be {M_reduced_total} x {M_reduced_total}, "
            f"but got {IM_reduced.shape}."
        )

    # 2) After removing pi block (if g>1), we have:
    M_reduced_no_pi = g*cluster_params_reduced

    # 3) After expanding Sigma to p^2 for each cluster, final dimension is:
    #    M_full_no_pi = g*(2p + p^2).
    M_full_no_pi = g * (2*p + p*p)

    # --- Remove the pi block if g>1
    if g > 1:
        offset_pi = g*cluster_params_reduced  # start of pi block
        keep_mask = np.ones(M_reduced_total, dtype=bool)
        # The pi parameters are offset_pi : offset_pi + (g-1)
        pi_indices = np.arange(offset_pi, offset_pi + (g-1))
        keep_mask[pi_indices] = False
        IM_reduced_no_pi = IM_reduced[keep_mask][:, keep_mask]
    else:
        # If g=1, no pi
        IM_reduced_no_pi = IM_reduced

    if IM_reduced_no_pi.shape != (M_reduced_no_pi, M_reduced_no_pi):
        raise ValueError(
            f"After removing pi, expected shape ({M_reduced_no_pi}, {M_reduced_no_pi}), "
            f"but got {IM_reduced_no_pi.shape}."
        )

    # --- 4) Build duplication matrix D_no_pi
    # shape: (M_full_no_pi, M_reduced_no_pi)
    D_no_pi = np.zeros((M_full_no_pi, M_reduced_no_pi), dtype=float)

    offset_reduced = 0
    offset_full = 0

    def copy_block_1to1(num_params):
        # copies 'num_params' parameters in a 1-to-1 fashion
        nonlocal offset_reduced, offset_full
        for kk in range(num_params):
            D_no_pi[offset_full + kk, offset_reduced + kk] = 1.0
        offset_reduced += num_params
        offset_full    += num_params

    # For each cluster j => we have mu_j (p), shape_j (p), Sigma_j (p*(p+1)//2)
    for j_ in range(g):
        # mu_j => p
        copy_block_1to1(p)
        # shape_j => p
        copy_block_1to1(p)

        # Expand Sigma_j from p*(p+1)//2 => p^2
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

    # --- 5) Final expanded info matrix
    IM_full_no_pi = D_no_pi @ IM_reduced_no_pi @ D_no_pi.T
    return IM_full_no_pi
