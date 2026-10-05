"""
Most of the codes are used from git repository mvem. Huge thanks to the
devoloper.

"""

import numpy as np
from fmvmm.numerics.robust_gh import gh_logpdf
import scipy.stats
from scipy.special import gammaln, kv, kve, digamma
from fmvmm.distributions.gig import _besselM3, _check_gig_pars
from fmvmm.distributions import gig
import warnings
import math
from fmvmm.utils.utils_dist import (
    pack_gh_family_unconstrained,
    unpack_gh_family_unconstrained,
    _pack_spd_cholesky,
    _unpack_spd_cholesky,
    score_mat_fd_unconstrained,
    info_opg_from_scores,
)

def custom_log(input_value):
    if input_value == 0:
        return np.log1p(input_value)
    else:
        return np.log(input_value)

def process_variable(var):
    if np.isnan(var) or var == 0:
        return 0.1
    else:
        return var



def logpdf(x, lmbda, chi, psi, mu, sigma, gamma):
    """Evaluate the unchanged normal/GIG family in log space.

    Scales must be full-rank positive definite. Variance-gamma densities can
    legitimately be infinite at their center; no distances are clipped.
    """
    return gh_logpdf(x, lmbda, chi, psi, mu, sigma, gamma)


def pdf(x, lmbda, chi, psi, mu, sigma, gamma):
    """
    Probability density function of the generalised hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    return np.exp(logpdf(x, lmbda, chi, psi, mu, sigma, gamma))

def loglike(x, lmbda, chi, psi, mu, sigma, gamma):
    """
    Log-likelihood function of the generalised hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    return np.sum(logpdf(x, lmbda, chi, psi, mu, sigma, gamma))


def total_params(lmbda, chi, psi, mu, sigma, gamma):
    p = len(mu)

    return 2 + 2*p + (p*(p+1)/2)


def rvs(lmbda, chi, psi, mu, sigma, gamma, size=1):
    """
    Random number generator of the generalised hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    d = len(mu)
    Z = np.random.normal(size=(size, d))
    A = np.linalg.cholesky(sigma).T
    W = gig.rvs(lmbda, chi, psi, size)

    return np.sqrt(W)[:, np.newaxis] * Z.dot(A) + np.tile(mu, (size, 1)) + np.outer(W, gamma)

def mean(lmbda, chi, psi, mu, sigma, gamma):
    """
    Mean function of the generalised hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    if psi==0: # gig -> invgamma
        if lmbda > -1:
            raise Exception("Mean for psi==0 is only defined for lambda < -1")
        alpha = -lmbda
        beta = 0.5*chi
        EW = beta/(alpha-1)
    elif psi>0:
        if chi==0: # gig -> gamma
            alpha = lmbda
            beta = 0.5*psi
            EW = alpha/beta
        elif chi>0: # gig
            alpha = np.sqrt(chi*psi)
            EW = np.sqrt(chi/psi) * kv(lmbda+1, alpha) / kv(lmbda, alpha)

    return mu + EW*gamma

def var(lmbda, chi, psi, mu, sigma, gamma):
    """
    Variance function of the generalised hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    if psi==0: # gig -> invgamma
        if lmbda > -2:
            raise Exception("Var for psi==0 is only defined for lambda < -2")
        alpha = -lmbda
        beta = 0.5*chi
        EW = beta/(alpha-1)
        VarW = beta**2 / ((alpha-1)**2 * (alpha-2))
    elif psi>0:
        if chi==0: # gig -> gamma
            alpha = lmbda
            beta = 0.5*psi
            EW = alpha/beta
            VarW = alpha/(beta**2)
        elif chi>0: # gig
            alpha = np.sqrt(chi*psi)
            EW = np.sqrt(chi/psi) * kv(lmbda+1, alpha) / kv(lmbda, alpha)
            EW2 = (chi/psi) * kv(lmbda+2, alpha) / kv(lmbda, alpha)
            VarW = EW2 - EW**2

    return VarW * np.outer(gamma, gamma) + EW * sigma


# def _alphabar2chipsi(alpha_bar, lmbda):
#     """
#     From (alpha_bar, lambda) parameterisation to (chi, psi)
#     """
#     #k1=process_variable(kv(lmbda+1, alpha_bar))
#     #k2=process_variable(kv(lmbda, alpha_bar))

#     #psi = process_variable(alpha_bar) * k1 / k2
#     if alpha_bar==0 and lmbda >0:
#         #VG Case
#         chi = 0
#         psi = 2*lmbda
#     elif alpha_bar==0 and lmbda <0:
#         #Student t case
#         chi = -2*(lmbda+1)
#         psi = 0
#     else:
#         psi = alpha_bar * kv(lmbda+1, alpha_bar) / kv(lmbda, alpha_bar)
#         chi = alpha_bar**2 / psi
#     #if psi==0:
#         #chi=alpha_bar**2/0.5
#     #else:
#         #chi = alpha_bar**2 / psi
#     return chi, psi

def _as_scalar_param(value, name):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Parameter '{name}' must be scalar.")
    return float(arr.reshape(-1)[0])


def _avoid_vg_singularity(lmbda, alpha_bar, d, eps=1e-8):
    """Retained private compatibility helper; never changes the family parameter.

    A finite-likelihood fitting domain must exclude lambda<=d/2 at the exact
    variance-gamma boundary. Density evaluation now exposes that singularity.
    """
    return _as_scalar_param(lmbda, "lmbda")


def _alphabar2chipsi(alpha_bar, lmbda, eps=1e-15):
    """Map to the identifiable E[W]=1 normal/GIG parameterization.

    Finite alpha_bar is never capped and small positive values remain positive.
    Scaled Bessel ratios avoid exponential underflow. The exact zero boundary
    requires a finite mixing mean; invalid boundaries raise explicitly.
    """
    alpha_bar = _as_scalar_param(alpha_bar, "alpha_bar")
    lmbda = _as_scalar_param(lmbda, "lmbda")
    if not np.isfinite([alpha_bar,lmbda]).all() or alpha_bar < 0:
        raise ValueError("alpha_bar must be finite nonnegative and lambda finite")
    if alpha_bar == 0:
        if lmbda > 0:
            return 0., 2*lmbda
        if lmbda < -1:
            return -2*(lmbda+1), 0.
        raise ValueError("E[W]=1 at alpha_bar=0 requires lambda>0 or lambda<-1")
    values = kve(np.array([lmbda,lmbda+1]), alpha_bar)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise FloatingPointError("Scaled Bessel ratio outside numerical range")
    logr = np.log(values[1])-np.log(values[0])
    chi = np.exp(np.log(alpha_bar)-logr)
    psi = np.exp(np.log(alpha_bar)+logr)
    if not np.isfinite([chi,psi]).all() or chi<=0 or psi<=0:
        raise FloatingPointError("Identifiable GIG parameters outside numerical range")
    return float(chi), float(psi)


def t_optfunc(thepars, delta_sum, xi_sum, n_rows):
    """
    Log-likelihood function of the inverse gamma distribution
    """
    nu = -2 * (-1 - np.exp(thepars))
    term1 = -n_rows * nu * np.log(nu/2 - 1)/2
    term2 = (nu/2 + 1) * xi_sum + (nu/2 - 1) * delta_sum
    term3 = n_rows * gammaln(nu/2)
    out = term1 + term2 + term3
    return out

# loglikelihood function of the gamma distribution
def vg_optfunc(thepars, xi_sum, eta_sum, n_rows):
    """
    Log-likelihood function of the gamma distribution
    """
    # print("doing vg")
    thepars = np.exp(thepars)
    term1 = n_rows * (thepars * np.log(thepars) - gammaln(thepars))
    term2 = (thepars - 1) * xi_sum - thepars * eta_sum
    out = -(term1 + term2)
    return out



def gig_optfunc(thepars, mix_pars_fixed, pars_order, delta_sum, eta_sum, xi_sum, n_rows):
    """
    Log-likelihood function of the generalized inverse gaussian distribution
    """
    # print("thepars",thepars)
    # print("mix_pars_fixed",mix_pars_fixed)
    out = np.nan
    tmp_pars = np.concatenate([thepars, mix_pars_fixed])
    # print("tmp_pars",tmp_pars)
    lmbda = tmp_pars[pars_order=="lmbda"]
    # print("lmbda",lmbda)
    alpha_bar = np.exp(tmp_pars[pars_order=="alpha_bar"])
    # print("alpha_bar",alpha_bar)
    chi, psi = _alphabar2chipsi(alpha_bar, lmbda)
    # print("chi, psi",chi, psi)
    if lmbda < 0 and psi == 0: # t
        out = t_optfunc(lmbda, delta_sum, xi_sum, n_rows)
    elif lmbda > 0 and chi == 0: # VG
        out = vg_optfunc(lmbda, xi_sum, eta_sum, n_rows)
    else: # ghyp, hyp, NIG
        term1 = (lmbda - 1) * xi_sum
        term2 = -chi * delta_sum/2
        term3 = -psi * eta_sum/2
        term4 = -n_rows * lmbda * np.log(chi)/2 + n_rows * lmbda * np.log(psi)/2 - \
            n_rows * _besselM3(lmbda, np.sqrt(chi * psi), logvalue = True)
        out = -(term1 + term2 + term3 + term4)
    return out

def _check_norm_pars(mu, sigma, gamma, dimension):
    """
    Check normal and skewness parameters  for consistency
    """
    if len(mu) != dimension:
        raise Exception("Parameter 'mu' must be of length " + str(dimension) + "!")
    if len(gamma) != dimension:
        raise Exception("Parameter 'gamma' must be of length " + str(dimension) + "!")
    if dimension > 1: # MULTIVARIATE
        if len(sigma.shape) != 2 or sigma.shape[0] != dimension or sigma.shape[1] != dimension:
            raise Exception("'sigma' must be a quadratic matrix with dimension " +\
                 str(dimension) + " x " + str(dimension) + "!")
    else: # UNIVARIATE
        if len(sigma) != dimension:
            raise Exception("Parameter 'sigma' must be a scalar!")

def fitghypmv(
    x, lmbda = 1, alpha_bar = 1, mu = None, sigma = None, gamma = None, symmetric = False,
    standardize = False, nit = 2000, reltol = 1e-8, abstol = 1e-7, silent = False,
    opt_pars = {"lmbda": True, "alpha_bar": True, "mu": True, "sigma": True, "gamma": True},
    weights = None, opt_maxiter = 100, opt_maxfun = 1000, opt_ftol = None,
    opt_gtol = None, opt_maxls = None, optimizer_options = None
):
    """
    Estimate the parameters of the generalised hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation. Covers all special
    cases as well.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: The intial value of lmbda. Defaults to 1.
    :tpe lmbda: float, optional
    :param alpha_bar: The initial value of alpha_bar. Defaults to 1.
    :type alpha_bar: float, optional
    :param mu: Optional initial value of mu, an array of shape (p,)
    :type mu: np.ndarray, optional
    :param sigma: Optional initial value of sigma, an array of shape (p,p)
    :type sigma: np.ndarray, optional
    :param gamma: Optional initial value of gamma, an array of shape (p,)
    :type gamma: np.ndarray, optional
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
    :param opt_pars: A dict of (lmbda, alpha_bar, mu, sigma, gamma) with boolean values,
        denoting if the parameters should be estimated or fixed to their initial value.
        Defaults to fitting all parameters.
    :type opt_pars: dict, optional
    :param opt_maxiter: Maximum L-BFGS-B iterations for scalar mixing-parameter updates.
    :type opt_maxiter: int, optional
    :param opt_maxfun: Maximum objective evaluations for scalar mixing-parameter updates.
    :type opt_maxfun: int, optional
    :param optimizer_options: Additional scipy.optimize.minimize options.
    :type optimizer_options: dict, optional
    :return: The fitted parameters (<float> lmbda, <float> alpha_bar, <array> mu, <array> sigma,
        <array> gamma) and a list of log-likelihood values at each iteration of the EM algorithm,
        the number of performed iterations, whether the algorithm coverged, and the final AIC.
    :rtype: dict
    """

    x = np.asarray(x, dtype=float)
    n, d = x.shape
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != n:
            raise ValueError("weights must have one entry per observation.")
        weights = np.clip(weights, 0.0, np.inf)
    n_eff = np.sum(weights)
    if n_eff <= np.finfo(float).eps:
        raise ValueError("At least one observation must have positive weight.")

    def _wmean(values):
        return np.sum(weights * values) / n_eff

    def _wsum(values):
        return np.sum(weights * values)

    def _weighted_cov(data, mean):
        centered = data - mean
        cov = (centered.T @ (weights[:, None] * centered)) / n_eff
        cov = 0.5 * (cov + cov.T)
        try:
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = np.min(eigvals)
            if min_eig < 1e-8:
                cov = cov + (1e-8 - min_eig) * np.eye(cov.shape[0])
        except np.linalg.LinAlgError:
            cov = cov + 1e-8 * np.eye(cov.shape[0])
        return cov

    inner_optimizer_options = {}
    if opt_maxiter is not None:
        inner_optimizer_options["maxiter"] = int(opt_maxiter)
    if opt_maxfun is not None:
        inner_optimizer_options["maxfun"] = int(opt_maxfun)
    if opt_ftol is not None:
        inner_optimizer_options["ftol"] = float(opt_ftol)
    if opt_gtol is not None:
        inner_optimizer_options["gtol"] = float(opt_gtol)
    if opt_maxls is not None:
        inner_optimizer_options["maxls"] = int(opt_maxls)
    if optimizer_options is not None:
        inner_optimizer_options.update(optimizer_options)

    lmbda = _avoid_vg_singularity(lmbda, alpha_bar, d)
    chi, psi = _alphabar2chipsi(alpha_bar, lmbda)


    # check parameters of the mixing distribution for consistency
    _check_gig_pars(lmbda, chi, psi)

    # .check.opt.pars(opt.pars, symmetric)
    #opt.pars <- .check.opt.pars(opt.pars, symmetric)

    m1 = np.sum(weights[:, None] * x, axis=0) / n_eff
    # center = x-m1
    center = m1 -x

    if mu is None: mu = m1
    if (gamma is None) or symmetric: gamma = np.zeros(d)
    if symmetric: opt_pars["gamma"] = False
    if sigma is None: sigma = _weighted_cov(x, m1)

    # check normal and skewness parameters  for consistency
    _check_norm_pars(mu, sigma, gamma, d)

    if standardize:
        # data will be standardized and initial values will be adapted
        tmp_mean = np.sum(weights[:, None] * x, axis=0) / n_eff
        sigma_chol = np.linalg.cholesky(np.linalg.inv(_weighted_cov(x, tmp_mean))).T
        x = x - tmp_mean
        x = x @ sigma_chol
        sigma = sigma_chol.T @ sigma @ sigma_chol
        gamma = sigma_chol @ gamma
        mu = sigma_chol @ (mu - tmp_mean)

    # Initialize fitting loop
    i = 0
    rel_closeness = 100
    abs_closeness = 100
    tmp_fit = {"convergence": 0, "message": None}
    lmbda = _avoid_vg_singularity(lmbda, alpha_bar, d)
    chi, psi = _alphabar2chipsi(alpha_bar, lmbda)
    # print("chi,psi:",chi,psi)
    ll = _wsum(logpdf(x, lmbda=lmbda, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))
    # print("ll",ll)

    ll_save = []
    lmbda_save_save = []
    chi_save = []
    psi_save = []
    mu_save = []
    sigma_save = []
    gamma_save = []

    ## Start interations
    while (abs_closeness > abstol) and (rel_closeness > reltol) and (i < nit):
        i = i + 1

        ##<------------------------ E-Step: EM update ------------------------------>
        ## The parameters mu, sigma and gamma become updated

        inv_sigma = np.linalg.inv(sigma)

        diff = x-mu
        Q = np.sum((diff @ inv_sigma.T) * diff, axis=1)

        # Offset = (gamma @ inv_sigma) @ gamma
        Offset = gamma @ inv_sigma @ gamma.T

        delta = gig.expect(lmbda-d/2, Q+chi, psi+Offset, func = "1/x")
        delta_bar = _wmean(delta)
        eta = gig.expect(lmbda-d/2, Q+chi, psi+Offset, func = "x")
        eta_bar = _wmean(eta)

        if opt_pars["gamma"]:
            gamma = -((weights * delta) @ center) / (n_eff*delta_bar*eta_bar - n_eff)
            # print("gamma",gamma)

        if opt_pars["mu"]:
            mu = (((weights * delta) @ x)/n_eff - gamma) / delta_bar
            # print("mu",mu)

        diff = x - mu
        tmp = (weights * delta)[:, None] * diff

        if opt_pars["sigma"]:
            sigma = (tmp.T @ diff)/n_eff - np.outer(gamma, gamma) * eta_bar
            # print("sigma",sigma)

        ##<------------------------ M-Step: EM update ------------------------------>
        ## Maximise the conditional likelihood function and estimate lambda, chi, psi

        inv_sigma = np.linalg.inv(sigma)
        Q = np.sum((diff @ inv_sigma) * diff, axis=1)

        Offset = gamma @ inv_sigma @ gamma.T

        xi_sum = _wsum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="logx"))


        if alpha_bar==0 and lmbda > 0 and not opt_pars["alpha_bar"] and opt_pars["lmbda"]:
            ##<------  VG case  ------>

            eta_sum = _wsum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="x"))

            minimizer_kwargs = {"method": "L-BFGS-B","args":(eta_sum, xi_sum, n_eff)}

            x0= np.log(lmbda)

            tmp_fit = scipy.optimize.minimize(
                vg_optfunc,
                x0,
                args=(eta_sum, xi_sum, n_eff),
                method='L-BFGS-B',
                options=inner_optimizer_options,
            ) #method='L-BFGS-B',np.log(lmbda)+2

            lmbda = _as_scalar_param(np.exp(tmp_fit["x"]), "lmbda")
            lmbda = _avoid_vg_singularity(lmbda, alpha_bar, d)


        elif alpha_bar==0 and lmbda<0 and not opt_pars["alpha_bar"] and opt_pars["lmbda"]:
            ##<------  Student-t case  ------>
            delta_sum = _wsum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="1/x"))

            x0= custom_log(-1 - lmbda) #np.log(lmbda) np.random.rand()
            tmp_fit = scipy.optimize.minimize(
                t_optfunc,
                x0,
                args=(delta_sum, xi_sum, n_eff),
                method='L-BFGS-B',
                options=inner_optimizer_options,
            ) #custom_log(-1 - lmbda)+10

            lmbda = _as_scalar_param((-1 - np.exp(tmp_fit["x"])), "lmbda")



        elif opt_pars["lmbda"] or opt_pars["alpha_bar"]:
            ##<------  ghyp, hyp, NIG case  ------>
            delta_sum = _wsum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="1/x"))
            eta_sum = _wsum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="x"))

            mix_pars = {"lmbda": lmbda, "alpha_bar": np.log(alpha_bar)}
            thepars = {x: mix_pars[x] for x in ["lmbda", "alpha_bar"] if opt_pars[x]}
            mix_pars_fixed = {x: mix_pars[x] for x in ["lmbda", "alpha_bar"] if not opt_pars[x]}
            pars_order = np.array(list(thepars.keys()) + list(mix_pars_fixed.keys()))
            thepars_v = np.array(list(thepars.values())).flatten()
            mix_pars_fixed_v = np.array(list(mix_pars_fixed.values())).flatten()



            tmp_fit = scipy.optimize.minimize(
                gig_optfunc,
                thepars_v,
                args=(mix_pars_fixed_v, pars_order, delta_sum, eta_sum, xi_sum, n_eff),
                method='L-BFGS-B',
                options=inner_optimizer_options,
            )

            par = np.concatenate([tmp_fit["x"], mix_pars_fixed_v])


            lmbda = _as_scalar_param(par[pars_order == "lmbda"], "lmbda")
            alpha_bar = _as_scalar_param(
                np.exp(par[pars_order == "alpha_bar"]), "alpha_bar")
            if np.isneginf(lmbda):
                lmbda=-50
            if np.isposinf(lmbda):
                lmbda=50
            if np.isneginf(alpha_bar):
                alpha_bar=-50
            if np.isposinf(alpha_bar):
                alpha_bar=50



        chi, psi = _alphabar2chipsi(alpha_bar, lmbda)


        ## Test for convergence
        ll_old = ll
        ll = _wsum(logpdf(x, lmbda=lmbda, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))

        abs_closeness = np.abs(ll - ll_old)
        rel_closeness = np.abs((ll - ll_old)/ll_old)

        if not np.isfinite(abs_closeness) or not np.isfinite(rel_closeness):
            warnings.warn("fit.ghypmv: Loglikelihood is not finite! Iteration stoped!\n" +\
                          "Loglikelihood :" + str(ll))
            break

        ll_save += [ll]
        lmbda_save_save += [lmbda]
        chi_save += [chi]
        psi_save += [psi]
        mu_save += [mu]
        sigma_save += [sigma]
        gamma_save += [gamma]

    ## END OF WHILE LOOP


    converged = False
    if i<nit and np.isfinite(rel_closeness) and np.isfinite(abs_closeness): converged = True
    # print("lmbda,alpha_bar",lmbda,alpha_bar)
    if(standardize):
        inv_sigma_chol = np.linalg.inv(sigma_chol)
        mu = inv_sigma_chol @ mu + tmp_mean
        sigma = inv_sigma_chol.T @ sigma @ inv_sigma_chol
        gamma = inv_sigma_chol @ gamma

        chi, psi = _alphabar2chipsi(alpha_bar, lmbda)
        ll = _wsum(logpdf(x, lmbda=lmbda, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))


    nbr_fitted_params = opt_pars["alpha_bar"] + opt_pars["lmbda"] + d * (opt_pars["mu"]+opt_pars["gamma"]) +\
        (d/2)*(d+1)*opt_pars["sigma"]

    aic = -2 * ll + 2 * nbr_fitted_params

    return {"lmbda": lmbda, "alpha_bar": alpha_bar, "mu": mu, "sigma": sigma, "gamma": gamma,
            "ll": ll_save, "n_iter": i, "converged": converged, "aic": aic}



def fit(x, lmbda=1, alpha_bar=1, symmetric=False, standardize=False, nit=2000,
        reltol=1e-8, abstol=1e-7, silent=False, flambda=None, falpha_bar=None,
        fmu=None, fsigma=None, fgamma=None, return_loglike=False, weights=None):
    """
    Estimate the parameters of the generalised hyperbolic distribution. We
    use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: The intial value of lmbda. Defaults to 1.
    :tpe lmbda: float, optional
    :param alpha_bar: The initial value of alpha_bar. Defaults to 1.
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
    :param flmbda: If flmbda!=None, force lmbda to flmbda. Defaults to None.
    :type flmbda: np.ndarray, optional
    :param falpha_bar: If falpha_bar!=None, force alpha_bar to falpha_bar.
        Defaults to None.
    :type falpha_bar: np.ndarray, optional
    :param fmu: If fmu!=None, force mu to fmu. Defaults to None.
    :type fmu: np.ndarray, optional
    :param fsigma: If fsigma!=None, force sigma to fsigma. Defaults to None.
    :type fsigma: np.ndarray, optional
    :param fgamma: If fgamma!=None, force gamma to fgamma. Defaults to None.
    :type fgamma: np.ndarray, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration.
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<float> lmbda, <float> chi, <float> psi, <array> mu,
        <array> sigma, <array> gamma). Also returns a list of log-likelihood values at
        each iteration of the EM algorithm if ``return_loglike=True``.
    :rtype: tuple
    """

    opt_pars = {"lmbda": True, "alpha_bar": True,
                "mu": True, "sigma": True, "gamma": True}

    lmbda = lmbda if flambda is None else flambda
    alpha_bar = alpha_bar if falpha_bar is None else falpha_bar

    fit = fitghypmv(
        x, lmbda=lmbda, alpha_bar=alpha_bar, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars, weights=weights)

    if fit["alpha_bar"] != 0:
        chi, psi = _alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
    elif fit["alpha_bar"]==0 and fit["lmbda"] > 0:
         chi, psi = 0, 2 * fit["lmbda"]
    elif fit["alpha_bar"]==0 and fit["lmbda"] < 0:
         chi, psi = -2 * (fit["lmbda"] + 1), 0


    if return_loglike:
        return fit["lmbda"], chi, psi, fit["mu"], fit["sigma"], fit["gamma"], fit["ll"]
    return fit["lmbda"], chi, psi, fit["mu"], fit["sigma"], fit["gamma"]


def fit_weighted(x, weights, **kwargs):
    return fit(x, weights=weights, **kwargs)


def alpha_bar_from_chi_psi(chi, psi):
    return float(np.sqrt(max(float(chi) * float(psi), 1e-300)))


def pack_gh_alpha_bar_unconstrained(*, p, lmbda, alpha_bar, mu, sigma, gamma,
                                    free=("lmbda", "alpha_bar")):
    parts = []
    if "lmbda" in free:
        parts.append(np.array([float(lmbda)], dtype=float))
    if "alpha_bar" in free:
        parts.append(np.array([np.log(max(float(alpha_bar), 1e-300))], dtype=float))
    parts.append(np.asarray(mu, dtype=float).reshape(p,))
    diag_raw, off = _pack_spd_cholesky(np.asarray(sigma, dtype=float).reshape(p, p))
    parts.extend([diag_raw, off, np.asarray(gamma, dtype=float).reshape(p,)])
    return np.concatenate(parts)


def unpack_gh_alpha_bar_unconstrained(u, *, p, fixed=None,
                                      free=("lmbda", "alpha_bar")):
    fixed = {} if fixed is None else fixed
    u = np.asarray(u, dtype=float).ravel()
    idx = 0
    lmbda = float(fixed.get("lmbda", 0.0))
    alpha_bar = float(fixed.get("alpha_bar", 1.0))

    if "lmbda" in free:
        lmbda = float(u[idx])
        idx += 1
    if "alpha_bar" in free:
        alpha_bar = float(np.exp(u[idx]))
        idx += 1

    mu = u[idx:idx + p].copy()
    idx += p
    diag_raw = u[idx:idx + p].copy()
    idx += p
    n_off = p * (p - 1) // 2
    off = u[idx:idx + n_off].copy()
    idx += n_off
    sigma = _unpack_spd_cholesky(diag_raw, off, p)
    gamma = u[idx:idx + p].copy()
    idx += p

    if idx != u.size:
        raise ValueError(f"Unpack consumed {idx} entries but u has {u.size}.")

    chi, psi = _alphabar2chipsi(alpha_bar, lmbda)
    return lmbda, chi, psi, mu, sigma, gamma


def score_mat_alpha_bar(x, lmbda, alpha_bar, mu, sigma, gamma, *,
                        free=("lmbda", "alpha_bar"), fixed=None,
                        step=1e-5, rel_step=True):
    x = np.asarray(x, dtype=float)
    p = x.shape[1]
    u_hat = pack_gh_alpha_bar_unconstrained(
        p=p, lmbda=lmbda, alpha_bar=alpha_bar, mu=mu, sigma=sigma,
        gamma=gamma, free=free,
    )

    def _unpack(u, *, p):
        return unpack_gh_alpha_bar_unconstrained(
            u, p=p, fixed=fixed, free=free,
        )

    return score_mat_fd_unconstrained(
        x,
        u_hat=u_hat,
        unpack_fun=_unpack,
        logpdf_fun=logpdf,
        p=p,
        step=step,
        rel_step=rel_step,
    )


def pack_vg_unconstrained(*, p, lmbda, mu, sigma, gamma):
    parts = [np.array([np.log(max(float(lmbda), 1e-300))], dtype=float)]
    parts.append(np.asarray(mu, dtype=float).reshape(p,))
    diag_raw, off = _pack_spd_cholesky(np.asarray(sigma, dtype=float).reshape(p, p))
    parts.extend([diag_raw, off, np.asarray(gamma, dtype=float).reshape(p,)])
    return np.concatenate(parts)


def unpack_vg_unconstrained(u, *, p):
    u = np.asarray(u, dtype=float).ravel()
    idx = 0
    lmbda = float(np.exp(u[idx]))
    idx += 1
    mu = u[idx:idx + p].copy()
    idx += p
    diag_raw = u[idx:idx + p].copy()
    idx += p
    n_off = p * (p - 1) // 2
    off = u[idx:idx + n_off].copy()
    idx += n_off
    sigma = _unpack_spd_cholesky(diag_raw, off, p)
    gamma = u[idx:idx + p].copy()
    idx += p
    if idx != u.size:
        raise ValueError(f"Unpack consumed {idx} entries but u has {u.size}.")
    return lmbda, 0.0, 2.0 * lmbda, mu, sigma, gamma


def score_mat_vg_fit(x, lmbda, mu, sigma, gamma, step=1e-5):
    x = np.asarray(x, dtype=float)
    p = x.shape[1]
    u_hat = pack_vg_unconstrained(
        p=p, lmbda=lmbda, mu=mu, sigma=sigma, gamma=gamma,
    )
    return score_mat_fd_unconstrained(
        x,
        u_hat=u_hat,
        unpack_fun=lambda u, *, p: unpack_vg_unconstrained(u, p=p),
        logpdf_fun=logpdf,
        p=p,
        step=step,
        rel_step=True,
    )


def pack_gst_unconstrained(*, p, lmbda, mu, sigma, gamma):
    tail = max(-(float(lmbda) + 1.0), 1e-300)
    parts = [np.array([np.log(tail)], dtype=float)]
    parts.append(np.asarray(mu, dtype=float).reshape(p,))
    diag_raw, off = _pack_spd_cholesky(np.asarray(sigma, dtype=float).reshape(p, p))
    parts.extend([diag_raw, off, np.asarray(gamma, dtype=float).reshape(p,)])
    return np.concatenate(parts)


def unpack_gst_unconstrained(u, *, p):
    u = np.asarray(u, dtype=float).ravel()
    idx = 0
    tail = float(np.exp(u[idx]))
    idx += 1
    lmbda = -1.0 - tail
    chi = 2.0 * tail
    mu = u[idx:idx + p].copy()
    idx += p
    diag_raw = u[idx:idx + p].copy()
    idx += p
    n_off = p * (p - 1) // 2
    off = u[idx:idx + n_off].copy()
    idx += n_off
    sigma = _unpack_spd_cholesky(diag_raw, off, p)
    gamma = u[idx:idx + p].copy()
    idx += p
    if idx != u.size:
        raise ValueError(f"Unpack consumed {idx} entries but u has {u.size}.")
    return lmbda, chi, 0.0, mu, sigma, gamma


def score_mat_gst_fit(x, lmbda, mu, sigma, gamma, step=1e-5):
    x = np.asarray(x, dtype=float)
    p = x.shape[1]
    u_hat = pack_gst_unconstrained(
        p=p, lmbda=lmbda, mu=mu, sigma=sigma, gamma=gamma,
    )
    return score_mat_fd_unconstrained(
        x,
        u_hat=u_hat,
        unpack_fun=lambda u, *, p: unpack_gst_unconstrained(u, p=p),
        logpdf_fun=logpdf,
        p=p,
        step=step,
        rel_step=True,
    )


# def info_mat(x, lmbda, chi, psi, mu, sigma, gamma):
#     from fmvmm.utils.utils_fmm import compute_info_scipy_fmvmm
#
#     IM = compute_info_scipy_fmvmm(logpdf,[lmbda, np.array([chi]), np.array([psi]), mu, sigma, gamma],x)
#
#     return IM

def score_mat(x, lmbda, chi, psi, mu, sigma, gamma, step=1e-5):
    """
    Per-observation scores in the fitted GH parameterization:
    lambda, log(alpha_bar), mu, Cholesky(Sigma), gamma.
    """
    return score_mat_alpha_bar(
        x, lmbda, alpha_bar_from_chi_psi(chi, psi), mu, sigma, gamma,
        free=("lmbda", "alpha_bar"), fixed={}, step=step,
    )

def info_mat(x, lmbda, chi, psi, mu, sigma, gamma, step=1e-5, ridge=1e-8):
    """
    OPG observed information matrix.
    """
    S = score_mat(x, lmbda, chi, psi, mu, sigma, gamma, step=step)
    I, cov, se = info_opg_from_scores(S, ridge=ridge)
    return I
