"""Stable log probability operations for unchanged FMVMM component families."""
import numpy as np
from scipy.special import log_ndtr
from .robust_mvt import _prepare, _scalar, LOG_TWO_PI


def weighted_log_likelihood(log_values, weights):
    """Sum only positive-weight terms, so 0 times -infinity is exactly zero.

    NaN/+infinity on a positive-weight observation is an error. A negative
    infinite positive-weight log density yields -infinity. All-zero weights
    yield zero; an optimizer must separately decide whether to fit that group.
    """
    values=np.asarray(log_values,dtype=float)
    weights=np.asarray(weights,dtype=float)
    if values.ndim!=1 or values.shape!=weights.shape:
        raise ValueError('Log densities and weights must be equally sized vectors')
    if not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Weights must be finite and nonnegative')
    keep=weights>0
    selected=values[keep]
    if np.isnan(selected).any() or np.isposinf(selected).any():
        raise FloatingPointError('Positive-weight log densities contain NaN or positive infinity')
    if np.isneginf(selected).any():
        return -np.inf
    return float(np.dot(weights[keep],selected))


def evaluate_weighted_log_likelihood(logpdf, x, weights, *args, **kwargs):
    """Evaluate a valid component on its positive-weight observations only."""
    x=np.asarray(x,dtype=float);weights=np.asarray(weights,dtype=float)
    if x.ndim!=2 or weights.shape!=(len(x),):
        raise ValueError('Observations and weights have inconsistent dimensions')
    if not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Weights must be finite and nonnegative')
    keep=weights>0
    if not keep.any():
        return 0.
    return weighted_log_likelihood(logpdf(x[keep],*args,**kwargs),weights[keep])


def _skew_argument(dev,decomp,lmbda):
    lmbda=np.asarray(lmbda,dtype=float)
    if lmbda.shape!=(dev.shape[1],) or not np.isfinite(lmbda).all():
        raise ValueError('Skewness must be a finite vector of the component dimension')
    # Symmetric inverse square root, exactly the parameter convention of FMVMM.
    with np.errstate(over='ignore',invalid='ignore'):
        whitened=(dev@decomp.eigenvectors)/np.sqrt(decomp.eigenvalues)
        argument=whitened@(decomp.eigenvectors.T@lmbda)
    if np.isnan(argument).any():
        raise FloatingPointError('Skewness projection exceeded floating-point range')
    return argument


def skew_normal_logpdf(x,loc,shape,lmbda,*,min_relative_eigenvalue=1e-12):
    """Log of 2 phi_p(x;loc,shape) Phi(lambda' shape^(-1/2)(x-loc))."""
    x,loc,decomp,dev,logq,single=_prepare(x,loc,shape,min_relative_eigenvalue)
    a=_skew_argument(dev,decomp,lmbda)
    with np.errstate(over='ignore',under='ignore'):
        result=np.log(2)-len(loc)*LOG_TWO_PI/2-decomp.logdet/2-np.exp(logq-np.log(2))+log_ndtr(a)
    if np.isnan(result).any() or np.isposinf(result).any():
        raise FloatingPointError('Invalid skew-normal log density')
    return float(result[0]) if single else result


def skew_contaminated_normal_logpdf(x,loc,shape,lmbda,nu,*,
                                   min_relative_eigenvalue=1e-12,
                                   return_diagnostics=False):
    """Log density of the exact FMVMM MSNC family, evaluated by log-sum-exp.

    nu=(contamination probability, inverse inflation). Endpoints of the first
    parameter are allowed; inverse inflation lies in (0,1]. No covariance
    inflation is materialized and no covariance jitter is introduced.
    """
    nu=np.asarray(nu,dtype=float)
    if nu.shape!=(2,) or not np.isfinite(nu).all():
        raise ValueError('Contamination parameters must be a finite length-two vector')
    mixing,inverse_inflation=nu
    if not 0<=mixing<=1 or not 0<inverse_inflation<=1:
        raise ValueError('MSNC requires mixing in [0,1] and inverse inflation in (0,1]')
    x,loc,decomp,dev,logq,single=_prepare(x,loc,shape,min_relative_eigenvalue)
    argument=_skew_argument(dev,decomp,lmbda)
    base=-len(loc)*LOG_TWO_PI/2-decomp.logdet/2
    clean=np.full(len(x),-np.inf);dirty=np.full(len(x),-np.inf)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        if mixing<1:
            clean=np.log1p(-mixing)+base-np.exp(logq-np.log(2))+log_ndtr(argument)
        if mixing>0:
            log_inflation=np.log(inverse_inflation)
            dirty=(np.log(mixing)+base+len(loc)*log_inflation/2
                   -np.exp(logq+log_inflation-np.log(2))
                   +log_ndtr(np.sqrt(inverse_inflation)*argument))
        result=np.log(2)+np.logaddexp(clean,dirty)
    if np.isnan(result).any() or np.isposinf(result).any():
        raise FloatingPointError('Invalid contaminated skew-normal log density')
    answer=float(result[0]) if single else result
    if not return_diagnostics:
        return answer
    return answer,{'dimension':len(loc),'effective_rank':len(loc),
                   'relative_minimum_eigenvalue':decomp.relative_minimum_eigenvalue,
                   'mixture_boundary':bool(mixing in (0,1)),
                   'negative_infinite_outputs':int(np.isneginf(result).sum())}
