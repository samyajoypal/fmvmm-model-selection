"""Generalized-hyperbolic log density from its normal/GIG mixture.

This evaluator does not alter chi, psi, distances, or the statistical family.
In particular the variance-gamma center is genuinely infinite for lambda<=p/2.
An optimizer requiring a finite bounded likelihood must exclude that domain.
"""
import numpy as np
from scipy.special import gammaln
from .robust_bessel import log_bessel_k, log_gig_integral_scaled
from .robust_mvt import _prepare, _scalar, LOG_TWO_PI, log_normalizer


def gh_logpdf(x, lmbda, chi, psi, loc, shape, gamma, *, min_relative_eigenvalue=1e-12):
    lmbda, chi, psi = (_scalar(v,n) for v,n in [(lmbda,'lambda'),(chi,'chi'),(psi,'psi')])
    if not np.isfinite([lmbda,chi,psi]).all() or chi < 0 or psi < 0:
        raise ValueError('GH parameters must be finite, chi>=0 and psi>=0')
    if (chi == 0 and (psi == 0 or lmbda <= 0)) or (psi == 0 and lmbda >= 0):
        raise ValueError('Invalid GIG mixing-distribution parameters')
    x,loc,decomp,dev,logq,single = _prepare(x,np.asarray(loc).reshape(-1),shape,min_relative_eigenvalue)
    gamma = np.asarray(gamma,dtype=float).reshape(-1)
    if gamma.shape != loc.shape or not np.isfinite(gamma).all():
        raise ValueError('GH skewness must be a finite vector of the data dimension')
    whitened = np.linalg.solve(decomp.cholesky,dev.T).T
    whitened_gamma = np.linalg.solve(decomp.cholesky,gamma/np.sqrt(decomp.scale))
    with np.errstate(over='ignore',invalid='ignore'):
        linear = whitened@whitened_gamma
    if not np.isfinite(linear).all() or not np.isfinite(whitened_gamma).all():
        raise FloatingPointError('GH skewness quadratic form exceeds numerical range')
    gamma_max=float(np.max(np.abs(whitened_gamma)))
    lognorm=(-np.inf if gamma_max==0 else
             2*np.log(gamma_max)+np.log(np.sum((whitened_gamma/gamma_max)**2)))
    p=len(loc);v=lmbda-p/2
    base=np.full(len(x),-p*LOG_TWO_PI/2-decomp.logdet/2)
    if psi==0 and gamma_max==0:
        # The symmetric inverse-gamma mixture is an exactly rescaled Student t.
        df=-2*lmbda
        logratio=logq-np.log(chi)
        answer=(log_normalizer(df,p)+p/2*(np.log(df)-np.log(chi))
                -decomp.logdet/2+v*np.logaddexp(0.,logratio))
        return float(answer[0]) if single else answer
    with np.errstate(divide='ignore'):
        loga=np.logaddexp(np.log(chi),logq)
        logb=np.logaddexp(np.log(psi),lognorm)
    if chi>0 and psi>0:
        logmix=(lmbda/2*(np.log(psi)-np.log(chi))-np.log(2)
                -log_bessel_k(lmbda,np.exp(.5*(np.log(chi)+np.log(psi)))))
    elif chi==0:
        logmix=lmbda*(np.log(psi)-np.log(2))-gammaln(lmbda)
    else:
        logmix=-lmbda*(np.log(chi)-np.log(2))-gammaln(-lmbda)
    answer=np.empty(len(x));center=np.isneginf(loga)
    if center.any():
        answer[center]=(np.inf if v<=0 else
                       base[center]+logmix+gammaln(v)+v*(np.log(2)-logb))
    keep=~center
    logz=.5*(loga[keep]+logb)
    loss=np.exp(logz)-linear[keep]
    positive=linear[keep]>0
    if positive.any():
        # sqrt((chi+Q)(psi+G))-u'v can catastrophically cancel for strong
        # skewness. Rationalize it; compute QG-(u'v)^2 as an orthogonal
        # residual norm, so every numerator term is nonnegative.
        u=whitened[keep][positive];direction=whitened_gamma/gamma_max
        residual=u-(u@direction/(direction@direction))[:,None]*direction
        maximum=np.max(np.abs(residual),axis=1);log_residual=np.full(len(u),-np.inf)
        nonzero=maximum>0
        log_residual[nonzero]=2*np.log(maximum[nonzero])+np.log(np.sum((residual[nonzero]/maximum[nonzero,None])**2,axis=1))
        with np.errstate(divide='ignore'):
            logchi=np.log(chi);logpsi=np.log(psi)
        log_numerator=np.logaddexp(np.logaddexp(logchi+logpsi,logchi+lognorm),
            np.logaddexp(logq[keep][positive]+logpsi,lognorm+log_residual))
        log_denominator=np.logaddexp(logz[positive],np.log(linear[keep][positive]))
        loss[positive]=np.exp(log_numerator-log_denominator)
    answer[keep]=base[keep]+logmix+log_gig_integral_scaled(v,loga[keep],logb)-loss
    if np.isnan(answer).any():
        raise FloatingPointError('Invalid generalized-hyperbolic log density')
    return float(answer[0]) if single else answer
