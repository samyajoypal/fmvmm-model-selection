"""Semi-analytic SMSN skew-t component score; only inverse df is differenced.

This helper evaluates one weighted component inside the FMVMM EM workflow.
It matches the constrained component coordinates mu, log-diag(L), lower(L),
b, eta with Sigma=L L.T+floor I, lambda=Sigma^(1/2)b, eta=1/df.
"""
import numpy as np
from scipy.linalg import cho_solve
from scipy.special import erfcx, ndtr
from fmvmm.numerics.robust_mvt import log_normalizer_eta
from fmvmm.numerics.robust_skew_t import student_logcdf, skew_t_logpdf

LOG2PI=np.log(2*np.pi)


def _normal_mills(z):
    result=np.empty_like(z);negative=z<0
    result[negative]=np.sqrt(2/np.pi)/erfcx(-z[negative]/np.sqrt(2))
    result[~negative]=np.exp(-z[~negative]**2/2-LOG2PI/2)/ndtr(z[~negative])
    return result


def _eta_difference(eta,eta_max,delta,y,eta_v,values,evaluate,step_factor):
    # Effective curvature scales remain bounded when a large delta is already
    # damped by eta. Close to eta=0 they prevent a fixed finite-difference step
    # from making a large change in the actual log density.
    effective_delta=np.max(delta/(1+eta*delta))
    effective_y=np.max(y*y/(1+eta_v*y*y))
    curvature=max(1.,effective_delta,effective_y)
    step=step_factor*np.cbrt(np.finfo(float).eps)*max(1.,eta)/curvature
    step=min(step,eta_max/4)
    if step<=0 or eta+step==eta: raise FloatingPointError('Skew-t eta difference step is not representable')
    side='central' if eta>=step and eta+step<=eta_max else 'forward' if eta<step else 'backward'
    def derivative(h):
        if side=='central':return (evaluate(eta+h)-evaluate(eta-h))/(2*h)
        if side=='forward':return (4*(evaluate(eta+h)-values)-(evaluate(eta+2*h)-values))/(2*h)
        return (4*(values-evaluate(eta-h))-(values-evaluate(eta-2*h)))/(2*h)
    coarse=derivative(step);fine=derivative(step/2)
    answer=fine+(fine-coarse)/3
    return answer,{'eta_step':step,'eta_difference_side':side,
                   'eta_refinement_max_scaled_change':float(np.max(np.abs(fine-coarse)/(1+np.abs(answer))))}


def values_gradient(x,u,policy,weights,*,eta_step_factor=1.,return_diagnostics=False):
    """Return native log densities and gradient of sum(weights*logdensity).

    Location, scale and b scores are analytic. The eta score uses a bounded,
    curvature-scaled three-point difference with a second step for refinement.
    The eta finite differences evaluate the exact same stable native density,
    including eta=0 as the skew-normal limit. No finite df is approximated as
    infinity. The refinement diagnostic is not a mathematical error bound.
    """
    x=np.asarray(x,float);u=np.asarray(u,float);w=np.asarray(weights,float)
    if x.ndim!=2 or w.shape!=(len(x),) or len(x)==0: raise ValueError('Invalid observation/weight dimensions')
    n,p=x.shape;m=p*(p-1)//2;start=2*p+m
    if u.shape!=(start+p+1,) or not np.isfinite(u).all(): raise ValueError('Invalid skew-t coordinates')
    if not np.isfinite(w).all() or np.any(w<0): raise ValueError('Weights must be finite and nonnegative')
    mu=u[:p];L=np.diag(np.exp(u[p:2*p]));L[np.tril_indices(p,-1)]=u[2*p:start]
    sigma=L@L.T+policy.covariance_floor*np.eye(p);b=u[start:start+p];eta=float(u[-1])
    if not 0<=eta<=policy.inverse_df_max: raise ValueError('Inverse df outside declared interval')
    C=np.linalg.cholesky(sigma);inverse=cho_solve((C,True),np.eye(p))
    z=x-mu;s=z@inverse;delta=np.einsum('ij,ij->i',z,s);a=z@b
    denominator=1+eta*delta;c=np.sqrt((1+p*eta)/denominator);y=a*c
    eta_v=eta/(1+p*eta)
    if eta==0:
        ratio=_normal_mills(y)
    else:
        log_pdf=(log_normalizer_eta(eta_v,1)
                 -.5*(1+eta_v)*np.log1p(eta_v*y*y)/eta_v)
        ratio=np.exp(log_pdf-student_logcdf(y,1/eta_v))
    vals,vecs=np.linalg.eigh(sigma);lmbda=((vecs*np.sqrt(vals))@vecs.T)@b
    def evaluate(inverse_df):
        return np.asarray(skew_t_logpdf(x,mu,sigma,lmbda,
            np.inf if inverse_df==0 else 1/inverse_df),float).reshape(-1)
    values=evaluate(eta)
    r=c*c+ratio*y*eta/denominator
    mu_score=((w*r)[:,None]*s).sum(0)-np.dot(w,ratio*c)*b
    G=.5*(s.T@((w*r)[:,None]*s)-w.sum()*inverse);dL=2*G@L
    if eta==0:
        # Exact one-sided derivative at the SN boundary. The first-order
        # Student-CDF expansion is Phi(a)-(a**3+a)*phi(a)*eta/4.
        eta_scores=(p*(p-2)/4-p*delta/2+delta**2/4
                    +ratio*(a*(p-delta)/2-(a**3+a)/4))
        diagnostics={'eta_difference_side':'exact_eta_zero_derivative','eta_step':0.,
                     'eta_refinement_max_scaled_change':0.}
    else:
        eta_scores,diagnostics=_eta_difference(eta,policy.inverse_df_max,delta,y,eta_v,values,evaluate,eta_step_factor)
    gradient=np.r_[mu_score,np.diag(dL)*np.diag(L),dL[np.tril_indices(p,-1)],
                   z.T@(w*ratio*c),w@eta_scores]
    if not np.isfinite(values).all() or not np.isfinite(gradient).all(): raise FloatingPointError('Nonfinite skew-t score')
    if return_diagnostics:return values,gradient,diagnostics
    return values,gradient
