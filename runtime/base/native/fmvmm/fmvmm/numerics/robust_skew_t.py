"""SMSN skew-Student log density in FMVMM's symmetric-root convention.

The density is 2 t_p(x; mu, Sigma, nu) T_(nu+p)(a sqrt((nu+p)/(nu+Q))).
No finite degree of freedom is replaced by infinity. The exact infinity
boundary is the skew-normal. Rare Student-CDF underflow or subnormal probability is evaluated by a
scaled one-dimensional tail integral instead of clipping probabilities.
"""
import numpy as np
from scipy.special import stdtr, log_ndtr
from scipy.integrate import quad
from .robust_mvt import _prepare, _scalar, student_t_logpdf, log_normalizer
from .robust_probability import _skew_argument, skew_normal_logpdf


def student_logcdf(argument, df):
    """Stable scalar/vector Student CDF, with a checked underflow fallback."""
    df = _scalar(df, 'df')
    if np.isnan(df) or df <= 0:
        raise ValueError('Student CDF df must be positive or positive infinity')
    z = np.asarray(argument, dtype=float)
    if np.isnan(z).any():
        raise ValueError('Student CDF argument contains NaN')
    if np.isposinf(df):
        return log_ndtr(z)
    flat = z.reshape(-1)
    prob = stdtr(df, -np.abs(flat))
    with np.errstate(divide='ignore'):
        logs = np.log(prob)
    # Nonzero subnormal CDF probabilities are quantized too coarsely for log
    # likelihoods and their derivatives. Use the same exact tail integral
    # before that quantization becomes significant; no finite df is truncated.
    for index in np.flatnonzero((prob < np.finfo(float).tiny) & np.isfinite(flat)):
        a = abs(float(flat[index]))
        log_ratio = 2*np.log(a)-np.log(df)
        # r=a^2/(df+a^2), without squaring a.
        r = np.exp(-np.logaddexp(0., -log_ratio))
        rate = df*r-(1-r)
        if not np.isfinite(rate) or rate <= 0:
            raise FloatingPointError('Student CDF tail rescaling failed')
        log_peak = (log_normalizer(df, 1)+np.log(a)
                    -.5*(df+1)*np.logaddexp(0., log_ratio))
        def integrand(v):
            u = v/rate
            if 2*u < 700:
                difference = np.log1p(r*np.expm1(2*u))
            else:
                difference = np.logaddexp(np.log1p(-r), np.log(r)+2*u)
            return np.exp(u-.5*(df+1)*difference)
        integral, error = quad(integrand, 0., np.inf, epsabs=1e-11,
                               epsrel=1e-11, limit=200)
        if not np.isfinite(integral) or integral <= 0 or error > 1e-8*integral:
            raise FloatingPointError('Student CDF tail integral failed accuracy check')
        logs[index] = log_peak + np.log(integral)-np.log(rate)
    positive = flat > 0
    logs[positive] = np.log1p(-np.exp(logs[positive]))
    answer = logs.reshape(z.shape)
    return float(answer) if z.ndim == 0 else answer


def skew_t_logpdf(x, loc, shape, lmbda, df, *, min_relative_eigenvalue=1e-12):
    df = _scalar(df, 'df')
    if np.isposinf(df):
        return skew_normal_logpdf(x, loc, shape, lmbda,
                                 min_relative_eigenvalue=min_relative_eigenvalue)
    if not np.isfinite(df) or df <= 0:
        raise ValueError('Skew-t df must be positive or positive infinity')
    x, loc, decomp, dev, logq, single = _prepare(x, loc, shape, min_relative_eigenvalue)
    a = _skew_argument(dev, decomp, lmbda)
    if not np.isfinite(a).all():
        raise FloatingPointError('Skew-t projection exceeds numerical range')
    # Multiplication in log space prevents both overflowing Q and 0*infinity.
    arg = np.zeros_like(a)
    nz = a != 0
    log_df_p = np.logaddexp(np.log(df), np.log(len(loc)))
    log_den = np.logaddexp(np.log(df), logq[nz])
    arg[nz] = np.sign(a[nz])*np.exp(np.log(np.abs(a[nz]))+.5*(log_df_p-log_den))
    answer = (np.log(2)+student_t_logpdf(x, loc, shape, df,
                 min_relative_eigenvalue=min_relative_eigenvalue)
              +student_logcdf(arg, df+len(loc)))
    if np.isnan(answer).any() or np.isposinf(answer).any():
        raise FloatingPointError('Invalid skew-t log density')
    return float(answer[0]) if single else answer
