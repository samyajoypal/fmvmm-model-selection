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

def logpdf(x, chi, psi, mu, sigma, gamma):
    """
    Log-probability density function of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param chi: Univariate parameter > 0.
    :type chi: float
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
    lmbda = (len(mu)+1)/2
    return ghypmv.logpdf(x, lmbda, chi, psi, mu, sigma, gamma)

def pdf(x, chi, psi, mu, sigma, gamma):
    """
    Probability density function of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param chi: Univariate parameter > 0.
    :type chi: float
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
    lmbda = (len(mu)+1)/2
    return ghypmv.pdf(x, lmbda, chi, psi, mu, sigma, gamma)

def loglike(x, chi, psi, mu, sigma, gamma):
    """
    Log-likelihood function of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param chi: Univariate parameter > 0.
    :type chi: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The log-likelihood for given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, chi, psi, mu, sigma, gamma))

def total_params(chi, psi, mu, sigma, gamma):
    p = len(mu)

    return 1 + 2*p + (p*(p+1)/2)

def rvs(chi, psi, mu, sigma, gamma, size):
    """
    Random number generator of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param chi: Univariate parameter > 0.
    :type chi: float
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
    lmbda = (len(mu)+1)/2
    return ghypmv.rvs(lmbda, chi, psi, mu, sigma, gamma, size)

def mean(chi, psi, mu, sigma, gamma):
    """
    Mean function of the multivariate hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param chi: Univariate parameter > 0.
    :type chi: float
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
    lmbda = (len(mu)+1)/2
    return ghypmv.mean(lmbda, chi, psi, mu, sigma, gamma)

def var(chi, psi, mu, sigma, gamma):
    """
    Variance function of the multivariate hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param chi: Univariate parameter > 0.
    :type chi: float
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
    lmbda = (len(mu)+1)/2
    return ghypmv.var(lmbda, chi, psi, mu, sigma, gamma)

def fit(x, alpha_bar=1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None, return_loglike=False,
        weights=None):
    """
    Estimate the parameters of the hyperbolic distribution. We use the
    (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param alpha_bar: The initial value of alpha_bar, a positive real number,
        where alpha_bar = chi =  psi.
    :type alpha_bar: float, optional
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
    :return: The fitted parameters (<float> chi, <float> psi, <array> mu, <array> sigma,
        <array> gamma). Also returns a list of log-likelihood values at each iteration
        of the EM algorithm if ``return_loglike=True``.
    :rtype: tuple
    """

    opt_pars = {"lmbda": False, "alpha_bar": True, "mu": fmu is None,
                "sigma": fsigma is None, "gamma": fgamma is None}
    lmbda = (x.shape[1]+1)/2
    fit = ghypmv.fitghypmv(
        x, lmbda=lmbda, alpha_bar=alpha_bar, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars, weights=weights)

    chi, psi = ghypmv._alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
    if return_loglike:
        return fit["lmbda"], chi, psi, fit["mu"], fit["sigma"], fit["gamma"], fit["ll"]
    return chi, psi, fit["mu"], fit["sigma"], fit["gamma"]


def fit_weighted(x, weights, **kwargs):
    return fit(x, weights=weights, **kwargs)

# def info_mat(x, chi, psi, mu, sigma, gamma):
#     from fmvmm.utils.utils_fmm import compute_info_scipy_fmvmm
#
#     IM = compute_info_scipy_fmvmm(logpdf,[np.array([chi]), np.array([psi]), mu, sigma, gamma],x)
#
#     return IM

def score_mat(x, chi, psi, mu, sigma, gamma, step=1e-5):
    """
    Per-observation score matrix wrt the fitted hyperbolic coordinates:
    log(alpha_bar), mu, Cholesky(Sigma), gamma.
    Returns S with shape (n, d_u).
    """
    lmbda_fixed = (len(mu) + 1) / 2.0
    return ghypmv.score_mat_alpha_bar(
        x, lmbda_fixed, ghypmv.alpha_bar_from_chi_psi(chi, psi), mu, sigma, gamma,
        free=("alpha_bar",), fixed={"lmbda": lmbda_fixed}, step=step,
    )


def info_mat(x, chi, psi, mu, sigma, gamma, step=1e-5, ridge=1e-8):
    """
    OPG information matrix for hyperbolic (lmbda fixed).
    Backward compatible signature: returns IM only.
    """
    S = score_mat(x, chi, psi, mu, sigma, gamma, step=step)
    I, cov, se = info_opg_from_scores(S, ridge=ridge)
    return I
