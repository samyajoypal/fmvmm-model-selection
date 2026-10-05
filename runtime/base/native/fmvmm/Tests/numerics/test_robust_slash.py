import math
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln,log_ndtr
from fmvmm.numerics.robust_slash import slash_logpdf,skew_slash_logpdf,NumericalIntegrationError
from fmvmm.numerics.robust_probability import skew_normal_logpdf

def scalar_oracle(x,lmbda,nu):
    """Independent adaptive scalar quadrature, scaled by the gamma normalizer."""
    x=np.asarray(x);lmbda=np.asarray(lmbda)
    p=len(x);b=float(x@x)/2;A=float(x@lmbda);a=nu+p/2
    if b==0:return math.log(nu/a)-p*math.log(2*math.pi)/2
    S=b+min(A,0)**2/2
    if S<1:
        def direct(t):
            if t==0:return 0.
            return 2*math.exp((2*a-1)*math.log(t)-b*t*t+log_ndtr(A*t))
        value,error=quad(direct,0,1,epsabs=0,epsrel=1e-12,limit=400)
        return math.log(2*nu)-p*math.log(2*math.pi)/2+math.log(value)
    coefficient=A/math.sqrt(S);fraction=min(A,0)**2/(2*S)
    shift=gammaln(a)
    def integrand(t):
        if t==0:return 0.
        log_value=(2*a-1)*math.log(t)-t*t+log_ndtr(coefficient*t)+fraction*t*t-shift
        return 2*math.exp(log_value)
    value,error=quad(integrand,0,math.sqrt(S) if S<1e4 else np.inf,epsabs=0,epsrel=1e-12,limit=400)
    if error>1e-9*value:raise AssertionError('Independent adaptive reference did not converge')
    return math.log(2*nu)-p*math.log(2*math.pi)/2-a*math.log(S)+shift+math.log(value)

class SlashTests(unittest.TestCase):
    def test_large_and_tiny_nu_domains(self):
        for p in [1,2,5]:
            lam=np.ones(p);x=np.array([np.zeros(p),np.ones(p),-np.ones(p)])
            limit=skew_normal_logpdf(x,np.zeros(p),np.eye(p),lam)
            for nu in [1e4,1e8,1e16]:
                actual=skew_slash_logpdf(x,np.zeros(p),np.eye(p),lam,nu)
                expected=[]
                for row in x:
                    b=float(row@row)/2;A=float(row@lam);a=nu+p/2
                    original=-p*math.log(2*math.pi)/2+math.log(2)-b+log_ndtr(A)
                    def integrand(w):
                        d=b*(-np.expm1(-w/a))+log_ndtr(A*np.exp(-w/(2*a)))-log_ndtr(A)
                        return math.exp(-w+d)
                    val,err=quad(integrand,0,np.inf,epsabs=0,epsrel=1e-12)
                    expected.append(original-math.log1p((p/2)/nu)+math.log(val))
                np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=1e-12)
                if nu==1e16:np.testing.assert_allclose(actual,limit,atol=1e-13,rtol=1e-13)
                normal=-p*math.log(2*math.pi)/2-(x*x).sum(1)/2
                if nu==1e16:np.testing.assert_allclose(slash_logpdf(x,np.zeros(p),np.eye(p),nu),normal,atol=1e-13,rtol=1e-13)
            np.testing.assert_array_equal(skew_slash_logpdf(x,np.zeros(p),np.eye(p),lam,np.inf),limit)
            for nu in [np.nextafter(0.,1.),1e-250]:
                actual=skew_slash_logpdf(x,np.zeros(p),np.eye(p),lam,nu)
                # nu is below representable significance in a: f/nu has the nu=0 integral limit.
                control=skew_slash_logpdf(x,np.zeros(p),np.eye(p),lam,1e-12)
                np.testing.assert_allclose(actual-np.log(nu),control-np.log(1e-12),atol=5e-12,rtol=1e-12)
        with self.assertRaises(NumericalIntegrationError):
            skew_slash_logpdf([100.,100.],[0,0],np.eye(2),[2,1],1000.)

    def test_symmetric_integrals(self):
        for p in [1,2,5]:
            x=np.array([np.zeros(p),np.ones(p)*.01,np.ones(p),np.ones(p)*30,np.ones(p)*1e4])
            for nu in [.001,.1,1,5,100]:
                actual=slash_logpdf(x,np.zeros(p),np.eye(p),nu)
                expected=np.array([scalar_oracle(row,np.zeros(p),nu) for row in x])
                np.testing.assert_allclose(actual,expected,atol=3e-10,rtol=1e-12)

    def test_skew_integrals(self):
        for p in [1,2,5]:
            lmbda=np.arange(p)+2.
            x=np.array([np.zeros(p),np.ones(p)*.01,np.ones(p),-np.ones(p),
                        np.ones(p)*30,-np.ones(p)*30,np.ones(p)*1e4,-np.ones(p)*1e4])
            for nu in [.001,.1,1,5,100]:
                actual,diagnostic=skew_slash_logpdf(x,np.zeros(p),np.eye(p),lmbda,nu,
                                                   log_tolerance=1e-8,return_diagnostics=True)
                expected=np.array([scalar_oracle(row,lmbda,nu) for row in x])
                np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=1e-12,
                                           err_msg=str((p,nu,diagnostic)))

    def test_symmetry_identity_and_refusal(self):
        x=np.array([[1.,2.],[10.,-30.],[-50.,20.]])
        symmetric=slash_logpdf(x,[0,0],np.eye(2),.1)
        a=skew_slash_logpdf(x,[0,0],np.eye(2),[2,3],.1)
        b=skew_slash_logpdf(-x,[0,0],np.eye(2),[2,3],.1)
        np.testing.assert_allclose(np.logaddexp(a,b),symmetric+np.log(2),atol=2e-8)
        with self.assertRaises(NumericalIntegrationError):
            skew_slash_logpdf(x,[0,0],np.eye(2),[2,3],.001,
                             log_tolerance=1e-14,initial_order=8,max_order=32)

if __name__=='__main__':unittest.main()
