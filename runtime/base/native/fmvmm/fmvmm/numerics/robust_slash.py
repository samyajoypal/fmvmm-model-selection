"""Stable slash densities; vectorized experimental skew-slash quadrature.

These evaluate the integrals in the reference SMSN slash implementations.
They do not change their statistical families or fit any model. Quadrature
refinement is an error diagnostic, not a mathematical proof of quadrature error.
"""
from functools import lru_cache
import numpy as np
from scipy.special import (gammainc,gammaincc,gammainccinv,gammaln,
                           roots_jacobi,roots_laguerre,logsumexp,log_ndtr)
from .robust_mvt import _prepare,_scalar,LOG_TWO_PI,student_t_logpdf
from .robust_probability import _skew_argument,skew_normal_logpdf

class NumericalIntegrationError(FloatingPointError):pass

def _log_slash_integral(log_b,a):
    """log integral_0^1 u^(a-1) exp(-b*u) du, with positive series at small b."""
    log_b=np.asarray(log_b,dtype=float)
    out=np.empty_like(log_b)
    zero=np.isneginf(log_b);out[zero]=-np.log(a)
    small=(~zero)&(log_b<np.log(a+1))
    b=np.exp(log_b[small]);term=np.ones_like(b);total=term.copy()
    for k in range(1,100001):
        term*=b/(a+k);total+=term
        if np.all(term<=np.finfo(float).eps*total):break
    else:raise NumericalIntegrationError('Slash positive series did not converge')
    out[small]=-b-np.log(a)+np.log(total)
    large=(~zero)&(~small)
    with np.errstate(over='ignore',under='ignore'):
        b=np.exp(log_b[large]);prob=gammainc(a,b)
        out[large]=gammaln(a)-a*log_b[large]+np.log(prob)
    if np.isnan(out).any() or np.isposinf(out).any():
        raise NumericalIntegrationError('Incomplete-gamma slash arithmetic failed')
    return out

def slash_logpdf(x,loc,shape,nu,*,min_relative_eigenvalue=1e-12):
    """Symmetric slash log density with exact incomplete-gamma representation."""
    nu=_scalar(nu,'nu')
    if np.isposinf(nu):return student_t_logpdf(x,loc,shape,np.inf,min_relative_eigenvalue=min_relative_eigenvalue)
    if not np.isfinite(nu) or nu<=0:raise ValueError('Slash nu must be positive or positive infinity')
    x,loc,decomp,dev,logq,single=_prepare(x,loc,shape,min_relative_eigenvalue)
    a=nu+len(loc)/2
    result=np.log(nu)-len(loc)*LOG_TWO_PI/2-decomp.logdet/2+_log_slash_integral(logq-np.log(2),a)
    return float(result[0]) if single else result

@lru_cache(maxsize=32)
def _jacobi_rule(order,a):
    nodes,weights=roots_jacobi(order,0,a-1)
    t=(nodes+1)/2
    with np.errstate(divide='ignore'):
        log_weights=np.log(weights)-a*np.log(2)
    if not np.isfinite(t).all() or np.isnan(log_weights).any() or np.isposinf(log_weights).any():
        raise NumericalIntegrationError('Gauss-Jacobi rule is outside numerical range')
    return t,log_weights

@lru_cache(maxsize=16)
def _laguerre_rule(order):
    nodes,weights=roots_laguerre(order)
    with np.errstate(divide='ignore'):
        logs=np.log(weights)
    if np.isnan(logs).any() or not np.isfinite(nodes).all():
        raise NumericalIntegrationError('Gauss-Laguerre rule is outside numerical range')
    return nodes,logs

def _large_nu_skew(logb,A,nu,a,half_dimension,base,log_tolerance,initial_order,max_order):
    """Endpoint representation u=exp(-w/a), valid also at very large finite nu.

    Numerical support is deliberately restricted to S/a<=1/4; outside that
    domain the endpoint integrand concentrates too far from the Laguerre nodes.
    The caller raises instead of treating an unsupported finite nu as infinity.
    """
    b=np.exp(logb);logPhi=log_ndtr(A)
    boundary=base+np.log(2)-b+logPhi
    previous=None;consecutive=0;order=initial_order;error=np.inf
    while order<=min(max_order,256):
        nodes,logweights=_laguerre_rule(order)
        ratio=nodes/a
        delta=(b[:,None]*(-np.expm1(-ratio))[None,:]
               +log_ndtr(A[:,None]*np.exp(-ratio/2)[None,:])-logPhi[:,None])
        result=boundary-np.log1p(half_dimension/nu)+logsumexp(logweights[None,:]+delta,axis=1)
        if not np.isfinite(result).all():raise NumericalIntegrationError('Large-nu endpoint arithmetic failed')
        if previous is not None:
            error=float(np.max(np.abs(result-previous)))
            consecutive=consecutive+1 if error<=log_tolerance else 0
            if consecutive>=2:return result,order,error
        previous=result;order*=2
    raise NumericalIntegrationError('Large-nu endpoint refinement failed: log difference %.3g'%error)

def skew_slash_logpdf(x,loc,shape,lmbda,nu,*,min_relative_eigenvalue=1e-12,
                     log_tolerance=1e-8,initial_order=32,max_order=2048,
                     return_diagnostics=False):
    """Vectorized scaled quadrature with refinement checks and an analytic tail bound.

    Scaling includes both the normal and negative-skew exponential tails.
    Two consecutive refinements must satisfy the requested log tolerance.
    Failure raises NumericalIntegrationError, never an approximate low score.
    The finite Gauss-Jacobi discretization error is empirically checked, not
    rigorously bounded; the optional truncated tail has an analytic upper bound.
    """
    nu=_scalar(nu,'nu')
    if np.isposinf(nu):
        answer=skew_normal_logpdf(x,loc,shape,lmbda,min_relative_eigenvalue=min_relative_eigenvalue)
        return (answer,{'gaussian_boundary':True,'quadrature_order':0,'tail_relative_bound':0.}) if return_diagnostics else answer
    if not np.isfinite(nu) or nu<=0:raise ValueError('Slash nu must be positive or positive infinity')
    if log_tolerance<=0 or not np.isfinite(log_tolerance):raise ValueError('Positive finite tolerance required')
    if initial_order<8 or max_order<4*initial_order:raise ValueError('At least two quadrature refinements required')
    x,loc,decomp,dev,logq,single=_prepare(x,loc,shape,min_relative_eigenvalue)
    lmbda=np.asarray(lmbda,dtype=float)
    A=_skew_argument(dev,decomp,lmbda)
    if not np.isfinite(A).all():raise NumericalIntegrationError('Skew argument exceeds numerical range')
    if np.all(lmbda==0):
        answer=slash_logpdf(x,loc,shape,nu,min_relative_eigenvalue=min_relative_eigenvalue)
        answer=float(answer[0]) if single else answer
        return (answer,{'symmetric_exact':True,'quadrature_order':0,'tail_relative_bound':0.}) if return_diagnostics else answer
    a=nu+len(loc)/2
    logb=logq-np.log(2)
    with np.errstate(divide='ignore'):
        log_skew=np.where(A<0,2*np.log(np.maximum(-A,0.))-np.log(2),-np.inf)
    logS=np.logaddexp(logb,log_skew)
    # This is a computational dispatch threshold, not a model boundary.
    # A log/exp coordinate round trip can put nominal a=500 a few ulps above
    # 500. The same quadrature safely covers that small numerical envelope;
    # retain the actual a and nu in every density calculation.
    general_limit=500*(1+8*np.finfo(float).eps)
    if a>general_limit:
        if np.any(logS-np.log(a)>np.log(.25)):
            raise NumericalIntegrationError('Unsupported skew-slash domain: a>500 and scaled tail concentration S/a>1/4')
        result,order,error=_large_nu_skew(logb,A,nu,a,len(loc)/2,-len(loc)*LOG_TWO_PI/2-decomp.logdet/2,
                                          log_tolerance,initial_order,max_order)
        answer=float(result[0]) if single else result
        diagnostics={'method':'large_nu_endpoint','quadrature_order':order,
                     'last_refinement_max_log_difference':error,'tail_relative_bound':0.,
                     'quadrature_error_certified':False,'tolerance':log_tolerance}
        return (answer,diagnostics) if return_diagnostics else answer
    origin=np.isneginf(logS)
    # P(Gamma(a,1)>cut)=1e-15; the remaining relative bound is checked below.
    cutoff=float(gammainccinv(a,1e-15))
    if not np.isfinite(cutoff):raise NumericalIntegrationError('Gamma tail cutoff failed')
    logT=np.minimum(logS,np.log(cutoff));T=np.exp(logT)
    scaled_A=np.zeros(len(A));negative=A<0
    keep=~origin
    nonzero_A=keep&(A!=0)
    with np.errstate(over='ignore'):
        scaled_A[nonzero_A]=np.sign(A[nonzero_A])*np.exp(np.log(np.abs(A[nonzero_A]))-.5*logS[nonzero_A])
    fraction=np.zeros(len(A))
    fraction[negative]=np.exp(log_skew[negative]-logS[negative])
    prefactor=np.log(2)+np.log(nu)-len(loc)*LOG_TWO_PI/2-decomp.logdet/2
    previous=None;consecutive=0;error=np.inf;bound=np.inf
    order=int(initial_order)
    while order<=max_order:
        # v=s^2 removes the square-root CDF endpoint singularity. The remaining
        # power is exactly integrated by the Jacobi weight t^(2a-1).
        nodes,logweights=_jacobi_rule(order,2*a)
        v=T[:,None]*nodes**2
        with np.errstate(invalid='ignore',under='ignore'):
            logh=-v+log_ndtr(scaled_A[:,None]*np.sqrt(v))+fraction[:,None]*v
            logs=logsumexp(logweights[None,:]+logh,axis=1)
            result=prefactor+np.log(2)+a*(logT-logS)+logs
        result[origin]=prefactor-np.log(2)-np.log(a)
        if not np.isfinite(result).all():raise NumericalIntegrationError('Nonfinite skew-slash quadrature')
        truncated=(logS>np.log(cutoff))
        if truncated.any():
            # Phi(-z)<=exp(-z²/2)/2, while Phi(z)<=1 for positive z.
            logtail=gammaln(a)+np.log(gammaincc(a,cutoff))
            log_relative=logtail-np.log(2)-a*logT[truncated]-logs[truncated]
            log_relative+=np.where(negative[truncated],-np.log(2),0.)
            bound=float(np.exp(log_relative).max())
        else:bound=0.
        if previous is not None:
            error=float(np.max(np.abs(result-previous)))
            consecutive=consecutive+1 if error<=log_tolerance and bound<=log_tolerance/10 else 0
            if consecutive>=2:
                answer=float(result[0]) if single else result
                diagnostics={'symmetric_exact':False,'quadrature_order':order,
                    'last_refinement_max_log_difference':error,'tail_relative_bound':bound,
                    'quadrature_error_certified':False,'tolerance':log_tolerance}
                return (answer,diagnostics) if return_diagnostics else answer
        previous=result;order*=2
    raise NumericalIntegrationError('Skew-slash refinement failed: last log difference %.3g, tail bound %.3g' % (error,bound))
