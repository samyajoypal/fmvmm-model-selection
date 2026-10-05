"""
Most of the codes are used from git repository mvem. Huge thanks to the
devoloper.

"""

from scipy.stats import multivariate_normal
import numpy as np
from fmvmm.utils.utils_dist import (
    pack_mvn_unconstrained,
    score_mat_mvn_analytic,
    score_mat_mvn_fd,
    info_opg_from_scores,
)

def fit(X, return_loglike=False):
    """
    Estimate the parameters of the multivariate normal distribution using
    maximum likelihood estimates.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param return_loglike: Return a list of log-likelihood values at each iteration.
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<array> mu, <array> sigma). Also returns a list of
        log-likelihood values at each iteration of the EM algorithm if ``return_loglike=True``.
    :rtype: tuple
    """
    mean = np.mean(X, axis = 0)
    cov = np.cov(X.T)
    if return_loglike:
        return mean, cov, loglike(X, mean, cov)
    return mean, cov

def pdf(x, mean, cov, allow_singular=True):
    """
    Probability density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :return: The density at each observation.
    :rtype: np.ndarray with shape (n,).
    """
    return multivariate_normal.pdf(x, mean, cov, allow_singular)

def logpdf(x, mean, cov, allow_singular=True):
    """
    Log-probability density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (n,).
    """
    return multivariate_normal.logpdf(x, mean, cov, allow_singular)

def loglike(x, mean, cov, allow_singular=True):
    """
    Log-likelihood function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :return: The log-likelihood for given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, mean, cov, allow_singular))

def total_params(mean, cov):
    p = len(mean)

    return p + (p*(p+1)/2)

def cdf(x, mean, cov, allow_singular=True, maxpts=1000000, abseps=1e-5, releps=1e-5):
    """
    Cumulative density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :param maxpts: The maximum number of points to use for integration.
    :type maxpts: int, optional
    :param abseps: The absolute error tolerance. Defaults to 1e-5.
    :type abseps: float, optional
    :param releps: The relative error tolerance. Defaults 1e-5.
    :type releps: float, optional
    :return: The cumulative density given all observations and parameters.
    :rtype: float
    """
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.cdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def logcdf(x, mean, cov, allow_singular=True, maxpts=1000000, abseps=1e-5, releps=1e-5):
    """
    Log-cumulative density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :param maxpts: The maximum number of points to use for integration.
    :type maxpts: int, optional
    :param abseps: The absolute error tolerance. Defaults to 1e-5.
    :type abseps: float, optional
    :param releps: The relative error tolerance. Defaults 1e-5.
    :type releps: float, optional
    :return: The log-cumulative density given all observations and parameters.
    :rtype: float
    """
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.logcdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def rvs(mean, cov, size=1, random_state=None):
    """
    Random number generator of the multivariate normal distribution.

    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :param random_state: Used for drawing random variates. Defaults to None.
    :type random_state: None, int, np.random.RandomState, np.random.Generator, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    return multivariate_normal.rvs(mean, cov, size, random_state)

def mean(mean, cov):
    """
    Mean function of the multivariate normal distribution.

    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    return mean

def var(mean, cov):
    """
    Variance function of the multivariate normal distribution.

    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    return cov



# def info_mat(X, mu, Sigma):
#     """
#     Construct the Fisher information matrix for a multivariate normal's
#     mean (p-dim) and full covariance (p^2-dim), ignoring the usual symmetry
#     constraints on Sigma.
#
#     Parameters
#     ----------
#     X : ndarray of shape (n, p)
#         Observed data (not strictly needed for the 'expected' Fisher info,
#         but we use n = X.shape[0]).
#     mu : ndarray of shape (p,)
#         The MLE of the mean (or any estimate at which you want the info).
#     Sigma : ndarray of shape (p, p)
#         The MLE of the covariance. Must be invertible.
#
#     Returns
#     -------
#     I : ndarray of shape (p + p^2, p + p^2)
#         The Fisher information block matrix with:
#           - top-left : n * inv(Sigma), size p x p
#           - bottom-right : (n/2) * (inv(Sigma) kronecker inv(Sigma)), size p^2 x p^2
#           - cross-blocks = 0
#         So total dimension is (p + p^2) x (p + p^2).
#
#     Notes
#     -----
#     1) Ordinarily, Sigma is symmetric, so only p(p+1)/2 distinct parameters.
#        Here we treat Sigma as if it has p^2 free entries. That will lead to
#        singularities if you truly consider small variations that break symmetry.
#     2) The 'observed' and 'expected' Fisher infos coincide for the normal at the MLE.
#     """
#     n, p = X.shape
#
#     # Invert the MLE covariance
#     inv_Sigma = np.linalg.inv(Sigma)
#
#     # Block for mu: p x p
#     I_mu = n * inv_Sigma
#
#     # Block for Sigma: p^2 x p^2
#     #   (n/2)*(inv(Sigma) kronecker inv(Sigma))
#     kron_inv = np.kron(inv_Sigma, inv_Sigma)
#     I_sigma = 0.5 * n * kron_inv
#
#     # Form the full (p + p^2)x(p + p^2) block matrix
#     I = np.zeros((p + p*p, p + p*p), dtype=float)
#
#     # top-left: I_mu
#     I[:p, :p] = I_mu
#     # bottom-right: I_sigma
#     I[p:, p:] = I_sigma
#
#     return I

def score_mat(X, mu, Sigma, step=1e-5, allow_singular=True):
    """
    Per-observation score matrix wrt unconstrained u = [mu, diag_raw, off].
    Uses analytic scores when Sigma is SPD. Falls back to FD otherwise.
    """
    X = np.asarray(X, float)
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    p = X.shape[1]

    # Analytic requires SPD (Cholesky exists)
    try:
        return score_mat_mvn_analytic(X, mu=mu, sigma=Sigma)
    except np.linalg.LinAlgError:
        if not allow_singular:
            raise

    # Fallback: FD (keeps old behavior for singular-ish Sigma)
    u_hat = pack_mvn_unconstrained(p=p, mu=mu, sigma=Sigma)

    def _logpdf_fun(A, m, S):
        return logpdf(A, m, S, allow_singular=allow_singular)

    S = score_mat_mvn_fd(
        X,
        u_hat=u_hat,
        logpdf_fun=_logpdf_fun,
        p=p,
        step=step,
    )
    return S


def info_mat(X, mu, Sigma, step=1e-5, ridge=1e-8, allow_singular=True):
    """
    OPG information matrix for MVN. Returns IM only (backward compatible).
    """
    S = score_mat(X, mu, Sigma, step=step, allow_singular=allow_singular)
    I, cov, se = info_opg_from_scores(S, ridge=ridge)
    return I
