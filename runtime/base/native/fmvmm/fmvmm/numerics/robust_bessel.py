"""Logarithmic Bessel K and normal/GIG mixture arithmetic.

Positive order recurrences avoid overflow of K at small arguments. Rare
low-order SciPy range failures use scaled integral representations, not a
replacement parameter. Small-product Gamma expansions are used only with
an explicit bound on their omitted term.
"""
import numpy as np
from scipy.special import kve,gammaln
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

LOG2=np.log(2.)


def _base_log_scaled(order,logz):
    """Base order in [0,1], with an integral fallback outside SciPy's range."""
    z=np.exp(logz) if logz>=np.log(np.nextafter(0.,1.)) else 0.
    value=kve(order,z) if z>0 else np.inf
    if np.isfinite(value) and value>0:return float(np.log(value))
    if logz>np.log(1e6):
        root=np.exp(logz/2)
        def logf(s):return -2*(root*np.sinh(s/(2*root)))**2+np.logaddexp(order*s/root,-order*s/root)-LOG2
        upper=12.
        integral,error=quad(lambda s:np.exp(logf(s)),0.,upper,epsabs=1e-13,epsrel=1e-13)
        tail=np.exp(logf(upper))/(upper-order/root)
        if integral<=0 or error+tail>1e-11*integral:raise FloatingPointError('Large-argument Bessel integral failed')
        return -.5*logz+np.log(integral)
    # K_v(z)=integral exp(-z cosh(t))*cosh(vt)dt. Truncate where
    # the logarithmic derivative is at most -50; bound the remaining tail.
    upper=max(2.,np.log(2*(order+60))-logz)
    def logf(t):
        exponent=np.exp(logz+np.logaddexp(t,-t)-LOG2)
        return -exponent+np.logaddexp(order*t,-order*t)-LOG2+z
    opt=minimize_scalar(lambda t:-logf(t),bounds=(0,upper),method='bounded')
    peak=max(logf(0),logf(float(opt.x)))
    integral,error=quad(lambda t:np.exp(logf(t)-peak),0.,upper,epsabs=1e-12,epsrel=1e-12,limit=300)
    zsinh=np.exp(logz+upper-LOG2)*(-np.expm1(-2*upper))
    tail=np.exp(logf(upper)-peak)/(zsinh-order)
    if integral<=0 or error+tail>1e-10*integral:raise FloatingPointError('Small-argument Bessel integral failed')
    return peak+np.log(integral)


def _log_scaled_from_log_argument(order,log_argument):
    order_array=np.asarray(order,float)
    if order_array.size!=1:raise ValueError('One Bessel order required internally')
    order=float(order_array.reshape(-1)[0])
    logs=np.asarray(log_argument,float)
    if not np.isfinite(order) or not np.isfinite(logs).all():raise ValueError('Finite Bessel order and log argument required')
    nu=abs(order);n=int(np.floor(nu));r=nu-n
    if n>10000:raise FloatingPointError('Bessel order exceeds supported recurrence budget')
    z=np.exp(logs)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        direct=kve(nu,z)
    output=np.empty_like(logs);good=np.isfinite(direct)&(direct>0)
    output[good]=np.log(direct[good])
    for index in zip(*np.where(~good)) if logs.ndim else ([()] if not good else []):
        lz=float(logs[index]);previous=_base_log_scaled(r,lz)
        if n==0:output[index]=previous;continue
        reflected=_base_log_scaled(1-r,lz)
        current=reflected if r==0 else np.logaddexp(reflected,LOG2+np.log(r)-lz+previous)
        for j in range(1,n):
            following=np.logaddexp(previous,LOG2+np.log(r+j)-lz+current)
            previous,current=current,following
        output[index]=current
    return output


def log_scaled_bessel_k(order,argument):
    """Log(exp(z) K_order(z)); the result need not exponentiate to a finite float."""
    z=np.asarray(argument,float)
    if not np.isfinite(z).all() or np.any(z<=0):raise ValueError('Bessel argument must be finite and positive')
    orders=np.asarray(order,float)
    if orders.size==1:return _log_scaled_from_log_argument(orders,np.log(z))
    orders,z=np.broadcast_arrays(orders,z)
    if not np.isfinite(orders).all():raise ValueError('Finite Bessel orders required')
    result=np.empty_like(z)
    for value in np.unique(orders):
        mask=orders==value;result[mask]=_log_scaled_from_log_argument(value,np.log(z[mask]))
    return result


def log_bessel_k(order,argument):
    z=np.asarray(argument,float)
    return log_scaled_bessel_k(order,z)-z


def log_bessel_k_ratio(numerator_order,denominator_order,argument):
    """Log K_num(z)/K_den(z), cancelling the exponential tail before subtraction."""
    return log_scaled_bessel_k(numerator_order,argument)-log_scaled_bessel_k(denominator_order,argument)


def _gig_integral(order,log_a,log_b,scaled=False):
    """Log integral w^(order-1) exp[-(a/w+b*w)/2]dw, a,b strictly positive.

    For nu=abs(order)>2, E[exp(-c/T)], T~Gamma(nu,1), c=a*b/4,
    lies between 1-c/(nu-1) and that value+c^2/[2(nu-1)(nu-2)].
    A first-order evaluation is used only if this rigorous relative remainder
    is <=2e-15. This is arithmetic error control, not a zero-skew substitution.
    """
    order_array=np.asarray(order,float)
    if order_array.size!=1:raise ValueError('One GIG order required')
    order=float(order_array.reshape(-1)[0]);la,lb=np.broadcast_arrays(np.asarray(log_a,float),np.asarray(log_b,float))
    if not np.isfinite(order) or not np.isfinite(la).all() or not np.isfinite(lb).all():
        raise ValueError('Finite order and positive finite-log GIG arguments required')
    logz=.5*(la+lb);nu=abs(order);out=np.empty_like(logz);series=np.zeros(logz.shape,dtype=bool)
    if nu>2:
        log_first=2*logz-2*LOG2-np.log(nu-1)
        log_remainder=2*(2*logz-2*LOG2)-LOG2-np.log(nu-1)-np.log(nu-2)
        with np.errstate(over='ignore',under='ignore'):
            first=np.exp(log_first)
        allowed=first<.5
        series=allowed&(log_remainder-np.log1p(-np.minimum(first,.5))<=np.log(2e-15))
        if np.any(series):
            lead=lb if order>0 else la
            out[series]=gammaln(nu)+nu*(LOG2-lead[series])+np.log1p(-first[series])
            if scaled:out[series]+=np.exp(logz[series])
    remaining=~series
    if np.any(remaining):
        lz=logz[remaining]
        logk=_log_scaled_from_log_argument(order,lz)
        if not scaled:logk-=np.exp(lz)
        out[remaining]=LOG2+.5*order*(la[remaining]-lb[remaining])+logk
    return float(out) if out.ndim==0 else out


def log_gig_integral(order,log_a,log_b):
    """Log normal/GIG integral, including bounded-error small-product arithmetic."""
    return _gig_integral(order,log_a,log_b)


def log_gig_integral_scaled(order,log_a,log_b):
    """log integral + sqrt(a*b), retaining cancellation-free exponential tails."""
    return _gig_integral(order,log_a,log_b,scaled=True)


def log_gig_moment(order,log_a,log_b,power):
    """Log conditional GIG moment E[W**power], with strictly positive a and b."""
    return _gig_integral(order+power,log_a,log_b,scaled=True)-_gig_integral(order,log_a,log_b,scaled=True)
