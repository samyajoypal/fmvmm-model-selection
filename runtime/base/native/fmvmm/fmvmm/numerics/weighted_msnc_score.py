"""Analytic MSNC weighted-component score in constrained native-EM coordinates.

Weighted component evaluation for constrained EM. This does not optimize full
mixtures or change the parameter domain. Coordinates are mu, log diagonal of L, strict
lower triangle of L, b, contamination probability, inverse inflation, where
Sigma=L L.T+covariance_floor I and lambda=Sigma^(1/2)b.
"""
import numpy as np
from scipy.linalg import cho_solve
from scipy.special import log_ndtr, erfcx, ndtr

LOG2PI=np.log(2*np.pi)


def _mills(z):
    out=np.empty_like(z);negative=z<0
    out[negative]=np.sqrt(2/np.pi)/erfcx(-z[negative]/np.sqrt(2))
    out[~negative]=np.exp(-.5*z[~negative]**2-.5*LOG2PI)/ndtr(z[~negative])
    return out


def values_gradient(x,u,policy,weights):
    """Return log densities and d/du sum_i weights_i log f(x_i;u).

    The caller enforces the declared coordinate bounds. Positive finite scales,
    contamination probability strictly between zero and one, and inverse
    inflation in (0,1] are required. Zero-weight observations are best removed
    by the caller, as in the existing constrained component M-step.
    """
    x=np.asarray(x,float);u=np.asarray(u,float);w=np.asarray(weights,float)
    if x.ndim!=2 or w.shape!=(len(x),):raise ValueError('Observation/weight dimensions differ')
    n,p=x.shape;m=p*(p-1)//2;start=2*p+m
    if u.shape!=(start+p+2,) or not np.isfinite(u).all():raise ValueError('Invalid MSNC coordinates')
    if not np.isfinite(w).all() or np.any(w<0):raise ValueError('Weights must be finite nonnegative')
    mu=u[:p];L=np.diag(np.exp(u[p:2*p]));L[np.tril_indices(p,-1)]=u[2*p:start]
    sigma=L@L.T+policy.covariance_floor*np.eye(p)
    b=u[start:start+p];a,gamma=u[-2:]
    if not 0<a<1 or not 0<gamma<=1:raise ValueError('Invalid contamination parameters')
    C=np.linalg.cholesky(sigma);inv=cho_solve((C,True),np.eye(p))
    d=x-mu;v=d@inv;q=np.einsum('ij,ij->i',d,v);z=d@b;sg=np.sqrt(gamma)
    base=np.log(2)-p*LOG2PI/2-np.log(np.diag(C)).sum()
    clean=base-q/2+log_ndtr(z)
    dirty=base+p*np.log(gamma)/2-gamma*q/2+log_ndtr(sg*z)
    values=np.logaddexp(np.log1p(-a)+clean,np.log(a)+dirty)
    clean_share=np.exp(np.log1p(-a)+clean-values)
    dirty_share=np.exp(np.log(a)+dirty-values)
    r=clean_share+gamma*dirty_share
    clean_mills=_mills(z);dirty_mills=_mills(sg*z)
    skew=clean_share*clean_mills+dirty_share*sg*dirty_mills
    mu_score=((w*r)[:,None]*v).sum(0)-np.dot(w,skew)*b
    G=.5*(v.T@((w*r)[:,None]*v)-w.sum()*inv)
    dL=2*G@L
    delta=dirty-clean
    # (f_dirty-f_clean)/f. This is exactly zero when gamma=1, and avoids
    # dividing cancellation in rho-a by a small a or 1-a near the box faces.
    mixing_score=np.empty(n);positive=delta>=0
    mixing_score[positive]=-np.expm1(-delta[positive])/(a+(1-a)*np.exp(-delta[positive]))
    mixing_score[~positive]=np.expm1(delta[~positive])/((1-a)+a*np.exp(delta[~positive]))
    inverse_inflation_score=dirty_share*(p/(2*gamma)-q/2+z*dirty_mills/(2*sg))
    gradient=np.r_[mu_score,np.diag(dL)*np.diag(L),dL[np.tril_indices(p,-1)],
                   d.T@(w*skew),w@mixing_score,w@inverse_inflation_score]
    if not np.isfinite(values).all() or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite MSNC density or score')
    return values,gradient
