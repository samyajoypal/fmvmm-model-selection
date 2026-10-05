"""Weighted slash scores in the constrained component coordinates.

Spatial scores use exact latent-moment identities. The primary skew-slash path
computes the tail score from E[log U] in the same quadrature. The preceding
native-density method remains an explicit fallback for unsupported joint-score
quadrature; it also remains the inexpensive symmetric-slash method.
"""
import numpy as np
from scipy.linalg import cho_solve
from .robust_slash import slash_logpdf, skew_slash_logpdf, _log_slash_integral

LOG2PI=np.log(2*np.pi)


def _native_values_gradient(x,u,name,policy,weights,*,return_diagnostics=False):
    if name not in ('msl','mssl'): raise ValueError('Expected a slash family.')
    x=np.asarray(x,float);u=np.asarray(u,float);w=np.asarray(weights,float)
    if x.ndim!=2 or w.shape!=(len(x),) or not len(x): raise ValueError('Invalid data/weight shapes.')
    n,p=x.shape;spatial=p+p*(p+1)//2
    if u.shape!=(spatial+(p if name=='mssl' else 0)+1,): raise ValueError('Invalid slash coordinate vector.')
    if not np.isfinite(x).all() or not np.isfinite(u).all() or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Nonfinite observations/coordinates or invalid weights.')
    mu=u[:p];L=np.diag(np.exp(u[p:2*p]));L[np.tril_indices(p,-1)]=u[2*p:spatial]
    sigma=L@L.T+policy.covariance_floor*np.eye(p)
    inverse=cho_solve((np.linalg.cholesky(sigma),True),np.eye(p))
    z=x-mu;s=z@inverse;delta=np.einsum('ij,ij->i',z,s)
    b=u[spatial:spatial+p] if name=='mssl' else np.zeros(p)
    A=z@b;v=float(u[-1]);nu=float(np.exp(v));a=nu+p/2
    if not policy.slash_df_min<=nu<=policy.slash_df_max*(1+1e-14): raise ValueError('Slash tail outside working interval.')
    eigenvalues,eigenvectors=np.linalg.eigh(sigma)
    lmbda=((eigenvectors*np.sqrt(eigenvalues))@eigenvectors.T)@b
    def evaluate(log_nu):
        tail=np.exp(log_nu)
        return np.asarray(skew_slash_logpdf(x,mu,sigma,lmbda,tail) if name=='mssl'
                          else slash_logpdf(x,mu,sigma,tail),float).reshape(-1)
    values=evaluate(v)
    if name=='mssl' and a+1>500:
        # The native general quadrature supports a<=500. A moment identity
        # must not require a different, unsupported density at a+1. Preserve
        # the original optimizer domain and use its former native score at
        # this narrow edge rather than clipping nu or tightening its bounds.
        from scipy.optimize._numdiff import approx_derivative
        from .constrained_mstep import _unpack,_bounds
        limits=np.asarray(_bounds(name,p,policy))
        def native_objective(z):
            alpha,_=_unpack(z,name,p,policy)
            return float(w@np.asarray(skew_slash_logpdf(x,*alpha)))
        gradient=approx_derivative(native_objective,u,method='3-point',bounds=limits.T).reshape(-1)
        if not np.isfinite(gradient).all(): raise FloatingPointError('Nonfinite slash edge score.')
        diagnostic=dict(gradient_method='native_finite_difference_at_shifted_moment_domain_edge',
                        score_error_certified=False)
        return (values,gradient,diagnostic) if return_diagnostics else (values,gradient)
    logdet=np.linalg.slogdet(sigma)[1]
    with np.errstate(divide='ignore'):
        logb=np.log(delta/2)
    if name=='msl':
        denominator=_log_slash_integral(logb,a)
        EU=np.exp(_log_slash_integral(logb,a+1)-denominator)
        H=np.zeros(n)
    else:
        # This evaluation is a moment identity, not an optimizer tail proposal.
        shifted=np.asarray(skew_slash_logpdf(x,mu,sigma,lmbda,nu+1),float).reshape(-1)
        EU=np.exp(shifted-values+np.log(nu)-np.log(nu+1))
        denominator=values-(np.log(2*nu)-p*LOG2PI/2-logdet/2)
        with np.errstate(divide='ignore'):
            log_rate=np.logaddexp(logb,2*np.log(np.abs(A))-np.log(2))
        H=np.exp(_log_slash_integral(log_rate,a+.5)-LOG2PI/2-denominator)
    location=(w*EU)@s-np.dot(w,H)*b
    G=.5*(s.T@((w*EU)[:,None]*s)-w.sum()*inverse);dL=2*G@L
    gradient=np.r_[location,np.diag(dL)*np.diag(L),dL[np.tril_indices(p,-1)]]
    if name=='mssl': gradient=np.r_[gradient,z.T@(w*H)]
    lower,upper=np.log(policy.slash_df_min),np.log(policy.slash_df_max)
    step=min(1e-3,(upper-lower)/4)
    side='central' if v-step>=lower and v+step<=upper else 'forward' if v-step<lower else 'backward'
    def difference(h):
        if side=='central': return (evaluate(v+h)-evaluate(v-h))/(2*h)
        if side=='forward': return (4*(evaluate(v+h)-values)-(evaluate(v+2*h)-values))/(2*h)
        return (4*(values-evaluate(v-h))-(values-evaluate(v-2*h)))/(2*h)
    coarse=difference(step);fine=difference(step/2);tail=fine+(fine-coarse)/3
    gradient=np.r_[gradient,w@tail]
    if not np.isfinite(values).all() or not np.isfinite(gradient).all(): raise FloatingPointError('Nonfinite slash value/score.')
    diagnostics=dict(log_nu_difference_side=side,log_nu_step=step,
                     gradient_method='analytic_spatial_latent_moments_native_log_nu_difference',
                     log_nu_refinement_max_scaled_change=float(np.max(np.abs(coarse-fine)/(1+np.abs(tail)))),
                     latent_mean_u_min=float(EU.min()),latent_mean_u_max=float(EU.max()),
                     score_error_certified=False)
    return (values,gradient,diagnostics) if return_diagnostics else (values,gradient)


def values_gradient(x,u,name,policy,weights,*,return_diagnostics=False):
    """Use joint skew-slash moments, with the original native method as fallback."""
    if name!='mssl':
        return _native_values_gradient(x,u,name,policy,weights,return_diagnostics=return_diagnostics)
    from .weighted_slash_joint_score import values_gradient as joint
    from .robust_slash import NumericalIntegrationError
    try:
        values,gradient,diagnostics=joint(x,u,policy,weights)
    except NumericalIntegrationError as error:
        # A failed joint-moment refinement does not alter the statistical
        # model or turn an unsupported density into a small likelihood.
        # The preceding native method must itself succeed and stay finite.
        values,gradient,diagnostics=_native_values_gradient(x,u,name,policy,weights,return_diagnostics=True)
        diagnostics=dict(diagnostics,joint_score_fallback_reason=str(error))
    return (values,gradient,diagnostics) if return_diagnostics else (values,gradient)
