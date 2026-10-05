"""
Most of the codes are used from git repository mvem. Huge thanks to the
devoloper.

"""

import numpy as np
from fmvmm.distributions import multivariate_genhyperbolic as ghypmv
from fmvmm.utils.utils_dist import (
    pack_gh_family_unconstrained,
    unpack_gh_family_unconstrained,
    score_mat_fd_unconstrained,
    info_opg_from_scores,
)

def logpdf(x, lmbda, psi, mu, sigma, gamma):
    """
    Log-probability density function of the multivariate variance gamma
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter > 0.
    :type lmbda: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (n,).
    """
    return ghypmv.logpdf(x, lmbda, 0, psi, mu, sigma, gamma)

def pdf(x, lmbda, psi, mu, sigma, gamma):
    """
    Probability density function of the multivariate variance gamma
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter > 0.
    :type lmbda: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The density at each observation.
    :rtype: np.ndarray with shape (n,).
    """
    return ghypmv.pdf(x, lmbda, 0, psi, mu, sigma, gamma)

def loglike(x, lmbda, psi, mu, sigma, gamma):
    """
    Log-likelihood function of the multivariate variance gamma distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter > 0.
    :type lmbda: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The log-likelihood given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, lmbda, psi, mu, sigma, gamma))

def total_params(lmbda, psi, mu, sigma, gamma):
    p = len(mu)

    return 1 + 2*p + (p*(p+1)/2)

def rvs(lmbda, psi, mu, sigma, gamma, size):
    """
    Random number generator of the multivariate variance gamma distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter > 0.
    :type lmbda: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :param size: The number of samples to draw.
    :type size: int
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    return ghypmv.rvs(lmbda, 0, psi, mu, sigma, gamma, size)

def mean(lmbda, psi, mu, sigma, gamma):
    """
    Mean function of the multivariate variance gamma distribution. We
    use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter > 0.
    :type lmbda: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    return ghypmv.mean(lmbda, 0, psi, mu, sigma, gamma)

def var(lmbda, psi, mu, sigma, gamma):
    """
    Variance function of the multivariate variance gamma distribution. We
    use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter > 0.
    :type lmbda: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (n,).
    """
    return ghypmv.var(lmbda, 0, psi, mu, sigma, gamma)

def fit(x, lmbda=1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None, return_loglike=False,
        weights=None):
    """
    Estimate the parameters of the variance gamma distribution. We
    use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: The initial value of lmbda > 0.
    :type lmbda: float, optional
    :param symmetric: Whether to fit a symmetric distribution or not. Default
        to False.
    :type symmetric: bool, optional
    :param standardize: Whether to standardize the data before fitting or not.
        Default to False.
    :type standardize: bool, optional
    :param nit: The maximum number of iterations to use in the EM algorithm.
        Defaults to 2000.
    :type nit: int, optional
    :param reltol: The relative convergence criterion for the log-likelihood
        function. Defaults to 1e-8.
    :type reltol: float, optional
    :param abstol: The relative convergence criterion for the log-likelihood
        function. Defaults to 1e-7.
    :type abstol: float, optional
    :param silent: Whether to print the log-likelihoods and parameter estimates
        during fitting or not. Defaults to False.
    :type silent: bool, optional
    :param fmu: If fmu!=None, force mu to fmu. Defaults to None.
    :type fmu: np.ndarray, optional
    :param fsigma: If fsigma!=None, force sigma to fsigma. Defaults to None.
    :type fsigma: np.ndarray, optional
    :param fgamma: If fgamma!=None, force gamma to fgamma. Defaults to None.
    :type fgamma: np.ndarray, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration.
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<float> lmbda, <float> psi, <array> mu, <array> sigma,
        <array> gamma). Also returns a list of log-likelihood values at each iteration
        of the EM algorithm if ``return_loglike=True``.
    :rtype: tuple
    """

    opt_pars = {"lmbda": True, "alpha_bar": False, "mu": True,
                "sigma": True, "gamma": True}

    fit = ghypmv.fitghypmv(
        x, lmbda=lmbda, alpha_bar=0, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars, weights=weights)

    psi = 2 * fit["lmbda"]
    if return_loglike:
        return fit["lmbda"], psi, fit["mu"], fit["sigma"], fit["gamma"], fit["ll"]
    return fit["lmbda"], psi, fit["mu"], fit["sigma"], fit["gamma"]


def fit_weighted(x, weights, **kwargs):
    return fit(x, weights=weights, **kwargs)

# def info_mat(x, lmbda, psi, mu, sigma, gamma):
#     from fmvmm.utils.utils_fmm import compute_info_scipy_fmvmm
#
#     IM = compute_info_scipy_fmvmm(logpdf,[lmbda, np.array([psi]), mu, sigma, gamma],x)
#
#     return IM

def score_mat(x, lmbda, psi, mu, sigma, gamma, step=1e-5):
    """
    Per-observation score matrix wrt the fitted VG coordinates:
    log(lambda), mu, Cholesky(Sigma), gamma. The fitted subfamily has
    chi fixed at 0 and psi = 2 * lambda.
    Returns S with shape (n, d_u).
    """
    return ghypmv.score_mat_vg_fit(x, lmbda, mu, sigma, gamma, step=step)


def info_mat(x, lmbda, psi, mu, sigma, gamma, step=1e-5, ridge=1e-8):
    """
    OPG information matrix for VG (chi fixed at 0).
    Backward compatible signature: returns IM only.
    """
    S = score_mat(x, lmbda, psi, mu, sigma, gamma, step=step)
    I, cov, se = info_opg_from_scores(S, ridge=ridge)
    return I
