"""
Source codes are from R package mixsmsn. 
Converted to python as closely as possible. Many thanks to the authors.
"""

import numpy as np
from scipy.stats import norm, gamma, multivariate_normal
from scipy.linalg import sqrtm

# ----------------------------------------------------------------
#        Auxiliary functions to replicate some R functionality
# ----------------------------------------------------------------

def find_interval(u, intervals):
    """
    Mimic R's findInterval(u, vec). 
    'intervals' should be a non-decreasing sequence, and 'u' is a scalar.
    This returns the index i such that intervals[i] <= u < intervals[i+1].
    """
    # np.searchsorted(intervals, u, side='right') gives the insertion point 
    # to the right. Subtract 1 to get a 0-based index.
    return np.searchsorted(intervals, u, side='right') - 1

def runif(n):
    """Replicate R's runif(n) -> uniform(0,1). Returns an array of length n."""
    return np.random.rand(n)

def rnorm(n, mean=0.0, sd=1.0):
    """Replicate R's rnorm(n, mean, sd)."""
    return np.random.normal(loc=mean, scale=sd, size=n)

def rgamma_r(n, shape, rate):
    """
    Replicate R's rgamma(n, shape, rate).
    In Python/scipy, Gamma is parameterized by shape (alpha) and scale (theta).
    R uses shape (alpha) and rate (beta). scale = 1/beta.
    """
    scale = 1.0 / rate
    return np.random.gamma(shape=shape, scale=scale, size=n)

# ----------------------------------------------------------------
#         Univariate SNI and mixture generation functions
# ----------------------------------------------------------------

def gen_Skew_normal(n, mu, sigma2, shape, nu=None):
    """
    Generate n random values from a Skew-Normal distribution.

    Parameters:
    -----------
    n : int
        Number of samples
    mu : float
        Location parameter
    sigma2 : float
        Scale^2 parameter
    shape : float
        Skewness parameter
    nu : ignored (for compatibility)
    """
    delta = shape / np.sqrt(1.0 + shape**2)
    # abs(rnorm(n)) + sqrt(1 - delta^2)*rnorm(n)
    z1 = np.abs(rnorm(n))
    z2 = rnorm(n)
    out = mu + np.sqrt(sigma2) * (delta * z1 + np.sqrt(1 - delta**2) * z2)
    return out

def gen_Skew_t(n, mu, sigma2, shape, nu):
    """
    Generate n random values from a Skew-t distribution.

    Parameters:
    -----------
    n : int
        Number of samples
    mu : float
        Location parameter
    sigma2 : float
        Scale^2 parameter
    shape : float
        Skewness parameter
    nu : float
        Degrees of freedom
    """
    # (rgamma(n, nu/2, nu/2))^(-1/2) in R means shape = nu/2, rate=nu/2
    # which is scale=2/nu in Python
    gam = rgamma_r(n, nu/2.0, nu/2.0)**(-0.5)  
    # Then multiply by a Skew Normal(0, sigma2, shape):
    sn = gen_Skew_normal(n, 0.0, sigma2, shape)
    return mu + gam * sn

def gen_Skew_cn(n, mu, sigma2, shape, nu1, nu2):
    """
    Generate n random values from a Skew Normal Contaminated (Skew-CN).

    This internally calls rmix(...) with mixture = nu1, 1-nu1.
    That is, with scaled variance sigma2/nu2 in the 'contaminated' component.
    """
    # We generate from the mixture:
    # rmix(n, [nu1, 1-nu1], "Skew.normal", list([mu, sigma2/nu2, shape], [mu, sigma2, shape]))
    out = rmix(n, [nu1, 1 - nu1],
               "Skew.normal",
               [[mu, sigma2/nu2, shape], [mu, sigma2, shape]],
               cluster=False)
    return out

def gen_Skew_slash(n, mu, sigma2, shape, nu):
    """
    Generate n random values from a Skew-Slash distribution.

    We use the fact that if U ~ Uniform(0,1), then U^(1/nu) inverts 
    the distribution for the slash's mixing variable.
    """
    u = runif(n)
    # u2 = u^(1/nu)
    u2 = u**(1.0/nu)
    # Then multiply a skew-normal(0, sigma2, shape) by (u2)^(-1/2).
    sn = gen_Skew_normal(n, 0.0, sigma2, shape)
    return mu + (u2**(-0.5))*sn

def rmix(n, pii, family, arg, cluster=False):
    """
    Generate n samples from a mixture of univariate distributions.

    Parameters:
    -----------
    n : int
        Number of samples
    pii : list or array
        Mixing proportions (length g)
    family : str
        One of ["Normal", "Skew.normal", "t", "Skew.t", "Skew.cn", "Skew.slash"].
    arg : list of lists
        Each element corresponds to the parameters for the chosen family.
        E.g., for a 2-component mixture of Skew.normal, 
        arg might be [[mu1, sigma2_1, shape1], [mu2, sigma2_2, shape2]]
    cluster : bool
        If True, returns also the cluster labels.

    Returns:
    --------
    x : 1D numpy array of length n
    or
    (x, clusters) if cluster=True
    """

    # Validate family
    valid_families = ["t", "Skew.t", "Skew.cn", "Skew.slash", "Skew.normal", "Normal"]
    if family not in valid_families:
        raise ValueError(f"Family {family} not recognized.")

    # Decide which generator to use
    if family in ["Normal", "Skew.normal"]:
        rF1 = gen_Skew_normal
        # If "Normal", forcibly set shape=0 for each arg
        if family == "Normal":
            for a in arg:
                # a is [mu, sigma2, shape]
                a[2] = 0.0

    elif family in ["t", "Skew.t"]:
        rF1 = gen_Skew_t
        # If "t", forcibly set shape=0 
        if family == "t":
            for a in arg:
                a[2] = 0.0

    elif family == "Skew.cn":
        rF1 = gen_Skew_cn

    elif family == "Skew.slash":
        rF1 = gen_Skew_slash

    # We'll generate intervals for mixture selection
    g = len(pii)
    interval = np.zeros(g+1)
    for j in range(1, g):
        interval[j] = interval[j-1] + pii[j-1]
    interval[g] = 1.0

    x1 = np.empty(n, dtype=float)
    clusters = np.empty(n, dtype=int)

    for i in range(n):
        u = np.random.rand()
        comp = find_interval(u, interval)
        clusters[i] = comp
        # param set = arg[comp]
        # E.g. if rF1 = gen_Skew_normal, we call 
        # gen_Skew_normal(1, mu, sigma2, shape, [nu if needed])
        # Because we said each arg[i] might be [mu, sigma2, shape, (nu...)].
        x1[i] = single_call_univariate(rF1, arg[comp])

    if cluster:
        return x1, clusters
    else:
        return x1

def single_call_univariate(fun, param_list):
    """
    Helper to call the univariate generator with a variable number of parameters.
    Param list might be 3, 4, or 5 items depending on the family.
    For example, for Skew.normal we expect [mu, sigma2, shape] (3 args).
    For Skew.t we expect [mu, sigma2, shape, nu] (4 args).
    etc.
    """
    return fun(1, *param_list)[0] if len(param_list) >= 3 else fun(1, *param_list)[0]


# ----------------------------------------------------------------
#        Multivariate SNI and mixture generation functions
# ----------------------------------------------------------------

def gen_SN_multi(n, mu, Sigma, shape, nu=None):
    """
    Generate n random values from a (multivariate) Skew-Normal distribution.

    mu : 1D array (length p)
    Sigma : (p x p) covariance matrix
    shape : 1D array (length p)
    nu : unused, included for API symmetry
    """
    mu = np.asarray(mu)
    shape = np.asarray(shape)
    p = mu.shape[0]
    # delta is a vector
    denom = np.sqrt(1.0 + np.dot(shape, shape))
    delta = shape / denom  # p-dimensional
    # sqrt(Sigma) using the symmetric square root
    sqrt_Sigma = sqrtm(Sigma)

    samples = np.zeros((n, p))
    I_p = np.eye(p)
    # A = sqrt(I - delta delta^T)
    M = I_p - np.outer(delta, delta)
    sqrt_M = sqrtm(M)  # p x p matrix

    for i in range(n):
        # Generate a single sample
        z1 = np.abs(rnorm(1))[0]   # scalar
        z2 = rnorm(p)              # p-dim standard normal
        # Combine
        vec = delta * z1 + sqrt_M @ z2
        # Multiply by sqrt_Sigma and shift by mu
        samples[i, :] = mu + (sqrt_Sigma @ vec)

    return samples

def gen_ST_multi(n, mu, Sigma, shape, nu):
    """
    Generate n random values from a (multivariate) Skew-t distribution.
    """
    mu = np.asarray(mu)
    p = mu.shape[0]
    # (rgamma(n, nu/2, nu/2))^(-1/2)
    gam = rgamma_r(n, nu/2.0, nu/2.0)**(-0.5)
    # We first generate Skew Normal(0, Sigma, shape)
    sn = gen_SN_multi(n, np.zeros(p), Sigma, shape)
    # Then scale each row by gam[i]
    for i in range(n):
        sn[i, :] *= gam[i]
    return (sn + mu)

def gen_SCN_multi(n, mu, Sigma, shape, nu):
    """
    Generate from the multivariate Skew Normal Contaminated (Skew-CN).
    nu is expected to be something like [nu1, nu2].
    """
    mu = np.asarray(mu)
    p = mu.shape[0]
    out = np.zeros((n, p))
    for i in range(n):
        u = np.random.rand()
        if u < nu[0]:
            # shape remains the same, but the covariance is scaled by 1/nu[1]
            out[i,:] = gen_SN_multi(1, mu, Sigma/nu[1], shape)[0]
        else:
            out[i,:] = gen_SN_multi(1, mu, Sigma, shape)[0]
    return out

def gen_SS_multi(n, mu, Sigma, shape, nu):
    """
    Generate from the multivariate Skew-Slash distribution.
    """
    mu = np.asarray(mu)
    p = mu.shape[0]
    u = runif(n)
    u2 = u**(1.0/nu)   # the mixing variable
    # Generate Skew-Normal(0, Sigma, shape)
    sn = gen_SN_multi(n, np.zeros(p), Sigma, shape)
    for i in range(n):
        sn[i, :] *= (u2[i]**-0.5)
    return (sn + mu)

def rmmix(n, pii, family, arg, cluster=False):
    """
    Generate n samples from a mixture of multivariate distributions.

    Parameters:
    -----------
    n : int
        Number of samples
    pii : list or array
        Mixing proportions (length g)
    family : str
        One of ["Normal","Skew.normal","t","Skew.t","Skew.cn","Skew.slash"].
    arg : list of dicts
        Each dict should have keys: mu, Sigma, shape, (nu)...
        Example (for Skew.normal):
           [
             {'mu': [...], 'Sigma': [...], 'shape': [...]},
             {'mu': [...], 'Sigma': [...], 'shape': [...]},
             ...
           ]
    cluster : bool
        If True, return (X, z) where z are mixture labels, else just X.

    Returns:
    --------
    X : (n x p) numpy array
    or 
    (X, clusters) if cluster=True
    """

    valid_families = ["Normal", "Skew.normal", "t", "Skew.t", "Skew.cn", "Skew.slash"]
    if family not in valid_families:
        raise ValueError(f"Family {family} not recognized.")

    # Decide generator
    if family == "Normal":
        rF1 = gen_SN_multi
        # forcibly set shape=0
        for a in arg:
            a['shape'] = np.zeros_like(a['shape'])

    elif family == "Skew.normal":
        rF1 = gen_SN_multi

    elif family in ["t", "Skew.t"]:
        rF1 = gen_ST_multi
        if family == "t":
            # forcibly set shape=0
            for a in arg:
                a['shape'] = np.zeros_like(a['shape'])

    elif family == "Skew.cn":
        rF1 = gen_SCN_multi

    elif family == "Skew.slash":
        rF1 = gen_SS_multi

    g = len(pii)
    interval = np.zeros(g+1)
    for j in range(1, g):
        interval[j] = interval[j-1] + pii[j-1]
    interval[g] = 1.0

    # dimension of the data
    p = len(arg[0]['mu']) 
    X = np.empty((n, p), dtype=float)
    clusters = np.empty(n, dtype=int)

    for i in range(n):
        u = np.random.rand()
        comp = find_interval(u, interval)
        clusters[i] = comp
        # Now we call rF1(1, **arg[comp])
        # for convenience, let's do a helper
        X[i, :] = single_call_multivariate(rF1, arg[comp])

    if cluster:
        return X, clusters
    else:
        return X

def single_call_multivariate(fun, param_dict):
    """
    Helper to call the multivariate generator with arguments in a dictionary.
    We expect something like:
        fun(1, mu, Sigma, shape, nu)
    or in general up to 4 or 5 parameters. We'll handle them carefully.
    """
    # We check keys: mu, Sigma, shape, maybe nu, maybe we ignore the rest
    if 'nu' in param_dict:
        samples = fun(1, param_dict['mu'], param_dict['Sigma'], param_dict['shape'], param_dict['nu'])
    else:
        samples = fun(1, param_dict['mu'], param_dict['Sigma'], param_dict['shape'])
    return samples[0, :]  # single sample row

