"""
Source codes are from R package mixsmsn. 
Converted to python as closely as possible. Many thanks to the authors.
"""

import numpy as np
from scipy.stats import norm, t, multivariate_normal
from scipy.integrate import quad
from scipy.linalg import sqrtm, inv
from scipy.special import gammaln, gamma

# Density/CDF of the Skew-Normal (SN) with location-scale

def dSN(y, mu=0, sigma2=1, shape=1):
    dens = 2 * norm.pdf(y, loc=mu, scale=np.sqrt(sigma2)) * norm.cdf(shape * ((y - mu) / np.sqrt(sigma2)))
    return dens

# Density/CDF of the Skew-t (ST) with location-scale
def dt_ls(x, loc=0, sigma2=1, shape=1, nu=4):
    d = (x - loc) / np.sqrt(sigma2)
    dens = 2 * t.pdf(d, df=nu) * t.cdf(np.sqrt((1 + nu) / (d**2 + nu)) * d * shape, df=nu + 1) / np.sqrt(sigma2)
    return dens

# Density/CDF of the Contaminated Skew-Normal (SNC)
def dSNC(y, mu, sigma2, shape, nu):
    dens = 2 * (
        nu[0] * norm.pdf(y, loc=mu, scale=np.sqrt(sigma2 / nu[1])) * norm.cdf(np.sqrt(nu[1]) * shape * (y - mu) / np.sqrt(sigma2)) +
        (1 - nu[0]) * norm.pdf(y, loc=mu, scale=np.sqrt(sigma2)) * norm.cdf(shape * (y - mu) / np.sqrt(sigma2))
    )
    return dens

# Density of the Skew-Slash (SS)
def dSS(y, mu, sigma2, shape, nu):
    resp = np.zeros_like(y)
    for i, yi in enumerate(y):
        f = lambda u: 2 * nu * u**(nu - 1) * norm.pdf(yi, loc=mu, scale=np.sqrt(sigma2 / u)) * norm.cdf(np.sqrt(u) * shape * (yi - mu) / np.sqrt(sigma2))
        resp[i], _ = quad(f, 0, 1)
    return resp

# Mixtures of Skew-Normal (SN)
def d_mixedSN(x, pi1, mu, sigma2, shape):
    g = len(pi1)
    dens = sum(pi1[j] * dSN(x, mu[j], sigma2[j], shape[j]) for j in range(g))
    return dens

# Mixtures of Skew-t (ST)
def d_mixedST(x, pi1, mu, sigma2, shape, nu):
    g = len(pi1)
    dens = sum(pi1[j] * dt_ls(x, mu[j], sigma2[j], shape[j], nu) for j in range(g))
    return dens

# Mixtures of Contaminated Skew-Normal (SNC)
def d_mixedSNC(x, pi1, mu, sigma2, shape, nu):
    g = len(pi1)
    dens = sum(pi1[j] * dSNC(x, mu[j], sigma2[j], shape[j], nu) for j in range(g))
    return dens

# Mixtures of Skew-Slash (SS)
def d_mixedSS(x, pi1, mu, sigma2, shape, nu):
    g = len(pi1)
    dens = sum(pi1[j] * dSS(x, mu[j], sigma2[j], shape[j], nu) for j in range(g))
    return dens

# Multivariate Skew-Normal (SN)
def dmvSN(y, mu, Sigma, lambda_):
    n, p = y.shape
    dens = 2 * multivariate_normal.pdf(y, mean=mu, cov=Sigma) * norm.cdf(
        np.sum((lambda_ @ np.linalg.inv(sqrtm(Sigma))) * (y - mu), axis=1)
    )
    return dens

def dmvt_ls(y, mu, Sigma, lambda_, nu):
    """
    Computes the density of the multivariate skew-t distribution with location-scale parameterization.

    Parameters:
    - y: (n, p) matrix where each row is a multivariate observation.
    - mu: (p,) mean vector.
    - Sigma: (p, p) covariance matrix.
    - lambda_: (p,) skewness parameter.
    - nu: degrees of freedom (scalar).

    Returns:
    - dens: Density values for each observation (n,)
    """
    n, p = y.shape

    # Mahalanobis distance
    diff = y - mu
    inv_Sigma = inv(Sigma)
    maha = np.sum((diff @ inv_Sigma) * diff, axis=1)

    # Compute density of Student-t
    denst = (
        (gamma((p + nu) / 2) / gamma(nu / 2)) *
        (np.pi ** (-p / 2)) * (nu ** (-p / 2)) *
        (np.linalg.det(Sigma) ** (-0.5)) *
        (1 + maha / nu) ** (-(p + nu) / 2)
    )

    # Compute skew correction factor
    root_Sigma = sqrtm(Sigma)
    skew_factor = (lambda_ @ inv(root_Sigma)).T
    skew_term = np.sum((skew_factor * diff), axis=1)
    
    # Compute final density
    dens = 2 * denst * t.cdf(
        np.sqrt((p + nu) / (maha + nu)) * skew_term,
        df=nu + p
    )

    return dens


# Mixtures of Multivariate Skew-Normal (SN)
def d_mixedmvSN(y, pi1, mu, Sigma, lambda_):
    g = len(pi1)
    dens = sum(pi1[j] * dmvSN(y, mu[j], Sigma[j], lambda_[j]) for j in range(g))
    return dens

# Mixtures of Multivariate Skew-t (ST)
def d_mixedmvST(y, pi1, mu, Sigma, lambda_, nu):
    g = len(pi1)
    dens = sum(pi1[j] * dmvt_ls(y, mu[j], Sigma[j], lambda_[j], nu) for j in range(g))
    return dens

def d_mixedmvST_custom(X,pis,alphas):
    dens = sum(pis[j] * dmvt_ls(X, *alphas[j] ) for j in range(len(pis)))
    return dens
    