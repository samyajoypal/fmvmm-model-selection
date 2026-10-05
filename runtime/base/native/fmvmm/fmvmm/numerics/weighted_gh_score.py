"""GH-family weighted score, with explicit nonsmooth-center handling.

Spatial derivatives follow posterior GIG moments. Canonical tail derivatives
are computed separately from native log densities. At genuine location cusps,
only an explicitly location-fixed score can be requested.
"""
import numpy as np
from scipy.linalg import cho_solve
from .robust_bessel import log_gig_moment
from fmvmm.numerics.constrained_mstep import _unpack,_bounds
from fmvmm.numerics.robust_gh import gh_logpdf

FAMILIES=('mghp','mvhb','mnig','mgst','mvvg','mvsl')

class NonsmoothLocationError(ValueError):
    """The weighted location objective has no ordinary score at this center."""


def _full(name,alpha,p):
    if name=='mghp':return alpha
    if name=='mgst':
        lam,chi,mu,sigma,gamma=alpha;return lam,chi,0.,mu,sigma,gamma
    if name=='mvvg':
        lam,psi,mu,sigma,gamma=alpha;return lam,0.,psi,mu,sigma,gamma
    if name in ('mvhb','mnig'):
        chi,psi,mu,sigma,gamma=alpha;return (p+1)/2 if name=='mvhb' else -.5,chi,psi,mu,sigma,gamma
    if name=='mvsl':
        mu,sigma,gamma=alpha;return (p+1)/2,0.,1.,mu,sigma,gamma
    raise ValueError('Unsupported GH-family score')


def values_gradient(x,u,name,policy,weights,*,fixed_location=False,return_diagnostics=False,tail_step=3e-5):
    """Return native log densities and weighted component-coordinate score.

    At a VG/Laplace cusp, an ordinary location derivative does not exist.
    Default evaluation therefore raises NonsmoothLocationError. For an
    explicitly location-fixed optimization, fixed_location=True omits that
    derivative, puts zeros in its unused slots, and records the restriction.
    Callers must optimize only the nonlocation coordinates in that mode.
    """
    if name not in FAMILIES:raise ValueError('Unsupported GH-family score')
    x=np.asarray(x,float);u=np.asarray(u,float);w=np.asarray(weights,float)
    if x.ndim!=2 or not len(x) or w.shape!=(len(x),) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Invalid weighted observations')
    p=x.shape[1];native,L=_unpack(u,name,p,policy);full=_full(name,native,p)
    lam,chi,psi,mu,sigma,gamma=full
    ld=np.asarray(gh_logpdf(x,*full)).reshape(-1)
    if not np.isfinite(ld[w>0]).all():raise FloatingPointError('Native GH density is unsupported at these positive-weight observations')
    C=np.linalg.cholesky(sigma);inverse=cho_solve((C,True),np.eye(p))
    y=x-mu;s=y@inverse;v=inverse@gamma
    a=chi+np.einsum('ij,ij->i',y,s);b=psi+float(gamma@v);order=lam-p/2
    centers=(chi==0)&np.all(y==0,axis=1)
    if np.any((a==0)&(~centers)):
        raise FloatingPointError('A nonzero spatial residual underflowed; it is not an exact location cusp.')
    positive_center=np.flatnonzero(centers & (w>0))
    nonsmooth=bool(len(positive_center) and order<=.5)
    if nonsmooth and not fixed_location:
        raise NonsmoothLocationError(f'{name}: ordinary location score unavailable at positive-weight center(s) {positive_center.tolist()}, lambda-p/2={order}.')
    A=np.zeros(len(x));B=np.zeros(len(x));interior=~centers
    if b==0:
        # Symmetric inverse-gamma/GST boundary: exact posterior moments.
        if np.any(a<=0) or order>=-1:raise FloatingPointError('Posterior inverse-gamma first moment unavailable')
        A=a/(-2*(order+1));B=-2*order/a
    else:
        if np.any(centers):
            if order<=0:raise FloatingPointError('Unbounded GH center is outside the declared model')
            A[centers]=2*order/b
            # B may be infinite for order<=1, but B*s and B*s*s' have
            # exact zero limits here when the spatial score is available.
            # The unavailable location case was explicitly handled above.
            B[centers]=0.
        if np.any(interior):
            loga=np.log(a[interior]);logb=np.log(b)
            A[interior]=np.exp(log_gig_moment(order,loga,logb,1))
            B[interior]=np.exp(log_gig_moment(order,loga,logb,-1))
    mu_score=(w[:,None]*(B[:,None]*s-v)).sum(0)
    gamma_score=(w[:,None]*(s-A[:,None]*v)).sum(0)
    ws=w@s
    G=.5*(-w.sum()*inverse+s.T@((w*B)[:,None]*s)-np.outer(ws,v)-np.outer(v,ws)+float(w@A)*np.outer(v,v))
    # The skew-Laplace native tuple has fixed E[W]=p+1, while the bounded
    # optimizer coordinates use canonical (p+1)*Sigma and (p+1)*gamma.
    if name=='mvsl':G/=(p+1);gamma_score/=(p+1)
    dL=2*G@L
    if fixed_location:mu_score=np.zeros(p)
    gradient=np.r_[mu_score,np.diag(dL)*np.diag(L),dL[np.tril_indices(p,-1)],gamma_score]
    spatial_size=len(gradient);limits=np.asarray(_bounds(name,p,policy));tails=[]
    for i in range(spatial_size,len(u)):
        h=tail_step*max(1,abs(u[i]))
        if name=='mvvg':h*=min(1.,order/lam)
        lower,upper=limits[i]
        def value(step):
            z=u.copy();z[i]+=step;aa,_=_unpack(z,name,p,policy)
            return np.asarray(gh_logpdf(x,*_full(name,aa,p)))
        if u[i]-h>=lower and u[i]+h<=upper:derivative=(value(h)-value(-h))/(2*h)
        elif u[i]-h<lower:derivative=(4*(value(h)-ld)-(value(2*h)-ld))/(2*h)
        else:derivative=(4*(ld-value(-h))-(ld-value(-2*h)))/(2*h)
        tails.append(float(w@derivative))
    gradient=np.r_[gradient,tails]
    if not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite GH-family component score')
    diagnostic={'family':name,'location_fixed':bool(fixed_location),'positive_weight_center_indices':positive_center.tolist(),
                'nonsmooth_location_center':nonsmooth,'radial_power':float(2*order) if np.any(centers) else None,
                'gradient_method':'analytic_spatial_and_native_tail_differences'}
    return (ld,gradient,diagnostic) if return_diagnostics else (ld,gradient)
