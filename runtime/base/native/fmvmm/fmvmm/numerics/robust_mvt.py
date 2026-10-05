"""Stable full-dimensional multivariate Student t arithmetic.

No degree-of-freedom clipping, pseudo-determinants, covariance repair or jitter.
The default relative eigenvalue floor is an explicit numerical admissibility
rule. Absolute covariance/mixture constraints belong to the fitting protocol.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, digamma

LOG_TWO_PI = float(np.log(2*np.pi))


class CovarianceDomainError(ValueError):
    """A covariance lies outside the declared full-rank numerical domain."""


@dataclass(frozen=True)
class SPDDecomposition:
    scale: float
    normalized_shape: np.ndarray
    cholesky: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    logdet: float
    relative_minimum_eigenvalue: float


def _dimension(p):
    if isinstance(p, (bool, np.bool_)) or int(p) != p or p < 1:
        raise ValueError("Dimension must be a positive integer")
    return int(p)


def _scalar(value, name):
    a = np.asarray(value, dtype=float)
    if a.size != 1:
        raise ValueError(name + " must be a scalar")
    return float(a.reshape(-1)[0])


def validate_spd(shape, *, min_relative_eigenvalue=1e-12, symmetry_rtol=1e-12):
    """Validate and factor a scale without silently dropping any dimension.

    A matrix is rescaled before eigendecomposition/Cholesky to avoid absolute
    overflow. Small accepted asymmetry is explicitly symmetrized. No eigenvalue
    is replaced; failures raise CovarianceDomainError.
    """
    a = np.asarray(shape, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.size:
        raise CovarianceDomainError("Scale must be a nonempty square matrix")
    if not np.isfinite(a).all():
        raise CovarianceDomainError("Scale entries must be finite")
    if not 0 <= min_relative_eigenvalue < 1:
        raise ValueError("Relative eigenvalue floor must lie in [0,1)")
    if not np.isfinite(symmetry_rtol) or symmetry_rtol < 0:
        raise ValueError("Symmetry tolerance must be finite and nonnegative")
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        raise CovarianceDomainError("Zero scale is singular")
    normalized = a/scale
    if np.max(np.abs(normalized-normalized.T)) > symmetry_rtol:
        raise CovarianceDomainError("Scale is not symmetric to the declared tolerance")
    normalized = .5*(normalized+normalized.T)
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    if eigenvalues[0] <= 0 or eigenvalues[-1] <= 0:
        raise CovarianceDomainError("Scale is not numerically positive definite")
    relative = float(eigenvalues[0]/eigenvalues[-1])
    if relative < min_relative_eigenvalue:
        raise CovarianceDomainError("Relative eigenvalue %.6g is below the declared floor %.6g" % (relative,min_relative_eigenvalue))
    try:
        chol = np.linalg.cholesky(normalized)
    except np.linalg.LinAlgError as exc:
        raise CovarianceDomainError("Full-rank Cholesky factorization failed") from exc
    logdet = len(a)*np.log(scale)+2*np.log(np.diag(chol)).sum()
    return SPDDecomposition(scale,normalized,chol,eigenvalues,eigenvectors,
                            float(logdet),relative)


def _log1p_ratio(numerator, log_denominator):
    return float(np.logaddexp(0.,np.log(float(numerator))-log_denominator))


def _half_gamma_correction(inv_df):
    """log Gamma(z+.5)/Gamma(z) - .5 log z, z=df/2, df>=64."""
    square = inv_df*inv_df
    return inv_df*(-.25+square*(1/24+square*(-1/20+square*(17/112+square*(-31/36)))))


def log_normalizer(df, p):
    """Student t log normalizer before minus one half log determinant.

    Exact integer recurrences handle even dimensions; odd dimensions use a
    stable half-gamma recurrence, with a high-order expansion only for df>=64.
    This preserves the normal boundary at df=+infinity without clipping df.
    """
    p = _dimension(p)
    df = _scalar(df,"df")
    if np.isnan(df) or df <= 0:
        raise ValueError("df must be positive, or positive infinity")
    if np.isposinf(df):
        return -.5*p*LOG_TWO_PI
    log_df = np.log(df)
    m = p//2
    if p % 2 == 0:
        return float(-m*LOG_TWO_PI+sum(_log1p_ratio(2*j,log_df) for j in range(1,m)))
    if df >= 64:
        half = -.5*LOG_TWO_PI+_half_gamma_correction(1/df)
    else:
        # Gamma(df/2) = Gamma(1+df/2)/(df/2); avoids underflow of df/2.
        half = gammaln((df+1)/2)-gammaln(1+df/2)+.5*log_df-np.log(2)-.5*np.log(np.pi)
    return float(half-m*LOG_TWO_PI+sum(_log1p_ratio(2*j+1,log_df) for j in range(m)))


def log_normalizer_eta(eta, p):
    """Same normalizer parameterized by eta=1/df, including eta=0."""
    eta = _scalar(eta,"eta")
    p = _dimension(p)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if eta == 0:
        return -.5*p*LOG_TWO_PI
    # Direct eta arithmetic also covers subnormal eta whose reciprocal overflows.
    m = p//2
    additions = sum(float(np.logaddexp(0.,np.log(2*j)+np.log(eta))) for j in range(1,m)) if p%2==0 else sum(float(np.logaddexp(0.,np.log(2*j+1)+np.log(eta))) for j in range(m))
    if p%2==0:
        return float(-m*LOG_TWO_PI+additions)
    if eta <= 1/64:
        return float(-.5*p*LOG_TWO_PI+_half_gamma_correction(eta)+additions)
    return log_normalizer(1/eta,p)


def derivative_log_normalizer_eta(eta, p):
    """Derivative in eta; at zero the exact one-sided value is p(p-2)/4."""
    eta = _scalar(eta,"eta")
    p = _dimension(p)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if eta == 0:
        return p*(p-2)/4
    m = p//2
    if p%2==0:
        return float(sum(1/(eta+1/(2*j)) for j in range(1,m)))
    additions = sum(1/(eta+1/(2*j+1)) for j in range(m))
    if eta <= 1/64:
        square = eta*eta
        half = -.25+square*(1/8+square*(-1/4+square*(17/16+square*(-31/4))))
    else:
        df = 1/eta
        half = -.5*df*df*(digamma((df+1)/2)-digamma(1+df/2))-.5*df
    return float(half+additions)


def _prepare(x, loc, shape, min_relative_eigenvalue):
    loc = np.asarray(loc,dtype=float)
    x = np.asarray(x,dtype=float)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    if loc.ndim != 1 or x.ndim != 2 or x.shape[1] != len(loc):
        raise ValueError("x and location dimensions differ")
    if not np.isfinite(x).all() or not np.isfinite(loc).all():
        raise ValueError("Observations and locations must be finite")
    decomposition = validate_spd(shape,min_relative_eigenvalue=min_relative_eigenvalue)
    if decomposition.cholesky.shape != (len(loc),len(loc)):
        raise ValueError("Scale and location dimensions differ")
    with np.errstate(over="ignore",invalid="ignore"):
        dev_scaled=(x-loc)/np.sqrt(decomposition.scale)
    if not np.isfinite(dev_scaled).all():
        raise FloatingPointError("Centered/scaled observations exceed floating-point range")
    whitened=solve_triangular(decomposition.cholesky,dev_scaled.T,lower=True).T
    if not np.isfinite(whitened).all():
        raise FloatingPointError('Whitened observations exceed floating-point range')
    maximum=np.max(np.abs(whitened),axis=1)
    log_maha=np.full(len(x),-np.inf)
    nonzero=maximum>0
    standardized=whitened[nonzero]/maximum[nonzero,None]
    log_maha[nonzero]=2*np.log(maximum[nonzero])+np.log(np.square(standardized).sum(axis=1))
    return x,loc,decomposition,dev_scaled,log_maha,single


def student_t_logpdf(x, loc, shape, df, *, min_relative_eigenvalue=1e-12,
                     return_diagnostics=False):
    """Full-dimensional Student t log density, including its Gaussian boundary.

    Return a scalar for one vector and an array for a matrix of observations.
    Extreme values below representable log-density range may return -infinity;
    covariance-domain and invalid-parameter errors are never converted to scores.
    """
    df=_scalar(df,"df")
    normalizer=log_normalizer(df,len(np.asarray(loc)))
    x,loc,decomp,dev,log_maha,single=_prepare(x,loc,shape,min_relative_eigenvalue)
    p=len(loc)
    with np.errstate(over="ignore",under="ignore",invalid="ignore"):
        if np.isposinf(df):
            tail=np.exp(log_maha-np.log(2))
        else:
            log_ratio=log_maha-np.log(df)
            log1pratio=np.logaddexp(0.,log_ratio)
            tail=np.empty(len(x))
            small=log_ratio < np.log(1e-5)
            ratio=np.exp(log_ratio[small])
            # log1p(r)/r, including r too tiny to form in float arithmetic.
            quotient=1-ratio/2+ratio*ratio/3-ratio**3/4
            tail[small]=np.exp(log_maha[small]-np.log(2))*quotient+.5*p*log1pratio[small]
            tail[~small]=(.5*df)*log1pratio[~small]+(.5*p)*log1pratio[~small]
        result=normalizer-.5*decomp.logdet-tail
    if np.isnan(result).any() or np.isposinf(result).any():
        raise FloatingPointError("Density arithmetic produced an invalid log value")
    answer=float(result[0]) if single else result
    if not return_diagnostics:
        return answer
    return answer,{"df":df,"dimension":p,"effective_rank":p,
                   "relative_minimum_eigenvalue":decomp.relative_minimum_eigenvalue,
                   "minimum_relative_eigenvalue_policy":min_relative_eigenvalue,
                   "log_determinant":decomp.logdet,"normalizer":normalizer,
                   "gaussian_boundary":bool(np.isposinf(df)),
                   "negative_infinite_outputs":int(np.isneginf(result).sum())}
