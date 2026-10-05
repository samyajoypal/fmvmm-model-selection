"""Joint skew-slash quadrature for native densities and latent-moment scores."""
import numpy as np
from scipy.linalg import cho_solve
from scipy.special import gammainccinv,gammaincc,gammaln,log_ndtr,logsumexp
from .robust_slash import _jacobi_rule,_log_slash_integral,NumericalIntegrationError


def values_gradient(x,u,policy,w,*,tolerance=1e-7):
    """Return native log densities, the weighted score, and error diagnostics.

    The scalar mixing variable U has posterior density proportional to
    u**(nu+p/2-1)*exp(-Q*u/2)*Phi(A*sqrt(u)) on (0,1).
    Location/scale use E[U]; the log-nu derivative is 1+nu*E[log U].
    Two consecutive whole-score and density refinements are required.
    Omitted-tail bounds exclude quadrature and floating-point discretization
    error; refinement is numerical evidence, not a rigorous total error bound.
    """
    from .constrained_mstep import _unpack
    x=np.asarray(x,float);u=np.asarray(u,float);w=np.asarray(w,float)
    if x.ndim!=2 or not len(x) or w.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('Invalid skew-slash observations or coordinates.')
    if not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Need positive finite skew-slash membership mass.')
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive finite score tolerance required.')
    expected=2*x.shape[1]+x.shape[1]*(x.shape[1]+1)//2+1
    if u.shape!=(expected,):raise ValueError('Invalid skew-slash coordinate vector.')
    p=x.shape[1];alpha,L=_unpack(u,'mssl',p,policy);mu,sigma,lmbda,nu=alpha
    nu=float(np.asarray(nu).ravel()[0]);a=nu+p/2
    if not policy.slash_df_min<=nu<=policy.slash_df_max*(1+1e-14):raise ValueError('Slash tail outside working interval.')
    if a>500*(1+8*np.finfo(float).eps): raise NumericalIntegrationError('Joint skew-slash score exceeds the general quadrature range.')
    spatial=p+p*(p+1)//2;b=u[spatial:spatial+p]
    inv=cho_solve((np.linalg.cholesky(sigma),True),np.eye(p));z=x-mu;s=z@inv
    delta=np.einsum('ij,ij->i',z,s);A=z@b
    with np.errstate(divide='ignore'):
        logb=np.log(delta/2);logskew=np.where(A<0,2*np.log(np.maximum(-A,0))-np.log(2),-np.inf)
    logS=np.logaddexp(logb,logskew);origin=np.isneginf(logS)
    cutoff=float(gammainccinv(a,1e-15))
    if not np.isfinite(cutoff):raise NumericalIntegrationError('Invalid joint skew-slash Gamma tail cutoff.')
    logT=np.minimum(logS,np.log(cutoff));T=np.exp(logT)
    scaled=np.zeros(len(x));nonzero=(~origin)&(A!=0)
    scaled[nonzero]=np.sign(A[nonzero])*np.exp(np.log(np.abs(A[nonzero]))-.5*logS[nonzero])
    fraction=np.zeros(len(x));negative=A<0;fraction[negative]=np.exp(logskew[negative]-logS[negative])
    prefactor=np.log(2*nu)-p*np.log(2*np.pi)/2-np.linalg.slogdet(sigma)[1]/2
    previous=None;previous_values=None;consecutive=0
    for order in (32,64,128,256,512,1024,2048):
        try:nodes,lw=_jacobi_rule(order,2*a)
        except (ValueError,FloatingPointError,OverflowError) as error:
            # At this point a and the quadrature order are valid. A SciPy
            # rule-construction range failure is a numerical support issue,
            # for which the wrapper retains the preceding native method.
            raise NumericalIntegrationError('Joint skew-slash quadrature rule construction failed.') from error
        v=T[:,None]*nodes**2
        terms=lw[None,:]-v+log_ndtr(scaled[:,None]*np.sqrt(v))+fraction[:,None]*v
        logs=logsumexp(terms,axis=1)
        with np.errstate(invalid='ignore'):
            logu=(logT-logS)[:,None]+2*np.log(nodes)[None,:]
            values=prefactor+np.log(2)+a*(logT-logS)+logs
        # The origin is handled exactly, without evaluating inf-inf moments.
        logu[origin]=0.
        posterior=np.exp(terms-logs[:,None])
        EU=np.sum(posterior*np.exp(logu),axis=1)
        Elog=np.sum(posterior*logu,axis=1)
        values[origin]=prefactor-np.log(2)-np.log(a);EU[origin]=a/(a+1);Elog[origin]=-1/a
        logI=values-prefactor
        with np.errstate(divide='ignore'):
            rate=np.logaddexp(logb,2*np.log(np.abs(A))-np.log(2))
        H=np.exp(_log_slash_integral(rate,a+.5)-.5*np.log(2*np.pi)-logI)
        location=(w*EU)@s-(w@H)*b
        G=.5*(s.T@((w*EU)[:,None]*s)-w.sum()*inv);dL=2*G@L
        gradient=np.r_[location,np.diag(dL)*np.diag(L),dL[np.tril_indices(p,-1)],z.T@(w*H),w@(1+nu*Elog)]
        if not np.isfinite(values).all() or not np.isfinite(gradient).all(): raise NumericalIntegrationError('Nonfinite joint skew-slash value or score.')
        truncated=logS>np.log(cutoff);tail_ratio=np.zeros(len(x))
        if np.any(truncated):
            logtail=gammaln(a)+np.log(gammaincc(a,cutoff))
            relative=logtail-np.log(2)-a*logT[truncated]-logs[truncated]
            relative+=np.where(negative[truncated],-np.log(2),0.)
            tail_ratio[truncated]=np.exp(relative)
        # The omitted mass changes E[U] by at most r. A sharper bound is
        # r*retained_EU + omitted_first_moment/retained_mass, where the latter
        # has the Gamma(a+1,T)/S bound. This avoids an unnecessarily large
        # bound when all relevant U values are small under a tight scale.
        Uerror=tail_ratio.copy()
        if np.any(truncated):
            log_first=gammaln(a+1)+np.log(gammaincc(a+1,cutoff))
            tail_first_over_mass=tail_ratio[truncated]*np.exp(log_first-logtail-logS[truncated])
            Uerror[truncated]=np.minimum(Uerror[truncated],tail_ratio[truncated]*EU[truncated]+tail_first_over_mass)
        # The omitted mass changes H by at most r*H,
        # and E[log U] by at most r*(|retained ElogU|+log(S/cutoff)).
        # These yield an absolute bound for each weighted score coordinate.
        Herror=H*tail_ratio;logerror=np.zeros(len(x))
        logerror[truncated]=tail_ratio[truncated]*(np.abs(Elog[truncated])+logS[truncated]-np.log(cutoff))
        mu_error=(w*Uerror)@np.abs(s)+(w@Herror)*np.abs(b)
        Gerror=.5*(np.abs(s).T@((w*Uerror)[:,None]*np.abs(s)));dLerror=2*Gerror@np.abs(L)
        errors=np.r_[mu_error,np.diag(dLerror)*np.diag(L),dLerror[np.tril_indices(p,-1)],np.abs(z).T@(w*Herror),nu*(w@logerror)]
        gradient_tail_bound=float(errors.max()/w.sum())
        if gradient_tail_bound>tolerance/10: raise NumericalIntegrationError('Joint skew-slash omitted-tail score bound exceeds the arithmetic tolerance.')
        if previous is not None:
            err=float(np.max(np.abs(gradient-previous))/w.sum());density_error=float(np.max(np.abs(values-previous_values)))
            consecutive=consecutive+1 if err<=tolerance and density_error<=1e-8 and tail_ratio.max()<=1e-9 else 0
            if consecutive>=2:return values,gradient,dict(order=order,gradient_refinement=err,density_refinement=density_error,
                gradient_tail_bound=gradient_tail_bound,density_tail_bound=float(tail_ratio.max()),
                gradient_method='joint_quadrature_latent_moments_including_log_tail',
                latent_mean_u_min=float(EU.min()),latent_mean_u_max=float(EU.max()),
                score_tolerance=tolerance,score_error_certified=False,
                tail_bound_meaning='Omitted-integral tail only; discretization and roundoff checked numerically.')
        previous=gradient;previous_values=values
    raise NumericalIntegrationError('Joint skew-slash density/score refinement failed.')
