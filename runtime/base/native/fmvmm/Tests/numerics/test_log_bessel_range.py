"""Portable range checks with independent exact and mixture-integral references."""
import math,unittest
from decimal import Decimal as D,localcontext
import numpy as np
from scipy.special import kve
from scipy.integrate import quad
from scipy.stats import multivariate_normal,invgamma
from fmvmm.numerics.robust_bessel import log_scaled_bessel_k,log_bessel_k,log_bessel_k_ratio,log_gig_integral,log_gig_moment
from fmvmm.numerics.robust_gh import gh_logpdf
from fmvmm.numerics.robust_slash import slash_logpdf,skew_slash_logpdf
from fmvmm.distributions.multivariate_genhyperbolic import _alphabar2chipsi
PI=D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')

def half_integer_log_scaled(m,z):
    with localcontext() as context:
        context.prec=100;z=D(str(z));series=D(0)
        for k in range(m+1):
            series+=D(math.factorial(m+k))/D(math.factorial(k)*math.factorial(m-k))/(2*z)**k
        return float((PI/(2*z)).ln()/2+series.ln())

def inverse_gamma_mixture(x,lam,chi,mu,sigma,gamma):
    def density(u):
        w=np.exp(u)
        return np.exp(multivariate_normal.logpdf(x,mu+w*gamma,w*sigma)+invgamma.logpdf(w,a=-lam,scale=chi/2)+u)
    value,error=quad(density,-8,8,epsabs=1e-13,epsrel=1e-11,limit=300)
    if value<=0 or error>1e-8*value:raise AssertionError('Independent mixture integral did not converge')
    return np.log(value)

class LogBesselRangeTests(unittest.TestCase):
    def test_half_integer_high_precision(self):
        for m in [0,1,10,51,53]:
            for z in [1e-310,1e-100,1e-8,.01,1.,100.,1e8,1e12]:
                expected=half_integer_log_scaled(m,z);actual=float(log_scaled_bessel_k(m+.5,z))
                self.assertLess(abs(actual-expected),max(2e-11,abs(expected)*2e-15),(m,z,actual,expected))
                self.assertAlmostEqual(float(log_scaled_bessel_k(-m-.5,z)),actual,places=12)
    def test_recurrence_and_safe_scipy_range(self):
        for nu in [0.,.001,.2,.99,1.,2.3,10.,51.,53.5]:
            for z in [1e-8,.01,1,100,1e4]:
                ref=kve(nu,z)
                if np.isfinite(ref):self.assertLess(abs(float(log_scaled_bessel_k(nu,z))-np.log(ref)),1e-10)
                lhs=float(log_bessel_k_ratio(nu+1,nu,z))
                rhs=np.logaddexp(float(log_bessel_k_ratio(abs(nu-1),nu,z)),np.log(2*nu/z) if nu else -np.inf)
                self.assertLess(abs(lhs-rhs),2e-11)
        with self.assertRaises(ValueError):log_bessel_k(1,0)
        with self.assertRaises(ValueError):log_bessel_k(1,np.inf)
    def test_array_orders_and_large_argument_moments(self):
        orders=np.array([-.5,1.5,51.5]);z=np.array([1e-300,1.,1e12])
        expected=np.array([half_integer_log_scaled(m,zz) for m,zz in zip([0,1,51],z)])
        np.testing.assert_allclose(log_scaled_bessel_k(orders,z),expected,rtol=2e-15,atol=2e-11)
        for z in [1e2,1e8,1e12]:
            # At order -1/2, the first and inverse moments follow exactly
            # from the finite half-integer Bessel polynomials.
            a=2*z;b=z/2
            self.assertAlmostEqual(log_gig_moment(-.5,np.log(a),np.log(b),1),np.log(2),places=12)
            self.assertAlmostEqual(log_gig_moment(-.5,np.log(a),np.log(b),-1),-np.log(2)+np.log1p(1/z),places=12)
    def test_laplace_strong_skew_cancellation(self):
        # Exact half-order density, with a 100-digit evaluation of the small
        # difference between its two large spatial terms.
        from scipy.special import gammaln
        for p in [2,3,4,5]:
            with localcontext() as context:
                context.prec=100;s=D('0.0001');g=D(50);t=D(30)
                q=D(p)*t*t/s;norm=D(p)*g*g/s;linear=D(p)*t*g/s
                loss=float(((1+norm)*q).sqrt()-linear)
                center=float(-D(p)*((2*PI).ln()+s.ln())/2
                    -D(p+1)/2*D(2).ln()+PI.ln()/2+(D(2)/(1+norm)).ln()/2)
            expected=center-gammaln((p+1)/2)-loss
            actual=gh_logpdf(np.full(p,30.),(p+1)/2,0.,1.,np.zeros(p),np.eye(p)*1e-4,np.full(p,50.))
            self.assertLess(abs(actual-expected),2e-10)
    def test_gst_failure_against_independent_mixture(self):
        for p in [2,3,4,5]:
            mu=np.zeros(p);sigma=np.eye(p)
            for small in [0.,1e-8,1e-4,.05]:
                gamma=np.full(p,small)
                for x in [mu,np.linspace(-.4,.7,p)]:
                    expected=inverse_gamma_mixture(x,-51.,100.,mu,sigma,gamma)
                    actual=gh_logpdf(x,-51.,100.,0.,mu,sigma,gamma)
                    self.assertLess(abs(actual-expected),2e-10,(p,small,actual,expected))
            for small in [1e-50,1e-100,1e-200]:
                actual=gh_logpdf(mu,-51.,100.,0.,mu,sigma,np.full(p,small))
                normal=gh_logpdf(mu,-51.,100.,0.,mu,sigma,np.zeros(p))
                self.assertLess(abs(actual-normal),2e-10)
    def test_gig_integral_moments(self):
        for order in [-52.,-53.5,47.5,49.]:
            for c in [1e-18,1e-8,1e-5,1e-3,.1]:
                a=100.;b=4*c/a;actual=log_gig_integral(order,np.log(a),np.log(b))
                ref=np.log(2)+order/2*np.log(a/b)+float(log_scaled_bessel_k(order,np.sqrt(a*b)))-np.sqrt(a*b)
                self.assertLess(abs(actual-ref),5e-11)
                for power in [-1.,1.]:
                    moment=log_gig_moment(order,np.log(a),np.log(b),power)
                    if c==1e-18:
                        nu=abs(order)
                        expected=(np.log(a/(2*(nu-1))) if power==1 else np.log(2*nu/a)) if order<0 else (np.log(2*nu/b) if power==1 else np.log(b/(2*(nu-1))))
                        self.assertLess(abs(moment-expected),2e-11)
    def test_declared_gh_domain_p2_through_p5(self):
        for p in [2,3,4,5]:
            mu=np.zeros(p);L=np.tril(np.full((p,p),20.));np.fill_diagonal(L,1e-5)
            scales=[np.eye(p),np.eye(p)*1e-4,L@L.T+np.eye(p)*1e-4]
            x=np.vstack([mu,np.full(p,1e-10),np.ones(p),-np.ones(p)*30,np.ones(p)*30])
            params=[]
            for lam in [-10.,0.,10.]:
                for abar in [.001,200.]:params.append((lam,*_alphabar2chipsi(abar,lam)))
            params += [(-51.,100.,0.),(-1.05,.1,0.)]
            params += [(lam,0.,2*lam) for lam in [p/2+.05,p/2+.49,p/2+.5,50.]]
            params += [((p+1)/2,0.,1.)]
            for sigma in scales:
                for gamma in [np.zeros(p),np.full(p,1e-8),np.full(p,50.)]:
                    for lam,chi,psi in params:
                        result=gh_logpdf(x,lam,chi,psi,mu,sigma,gamma)
                        self.assertTrue(np.isfinite(result).all(),(p,lam,chi,psi,gamma,result))
    def test_slash_study_dimensions_upper_bound(self):
        for p in [2,3,4,5]:
            x=np.vstack([np.zeros(p),np.ones(p)*30,-np.ones(p)*30,np.ones(p)*1e5,-np.ones(p)*1e5])
            for nu in [.001,.1,5.,450.]:
                symmetric=slash_logpdf(x,np.zeros(p),np.eye(p)*1e-4,nu)
                skew,diag=skew_slash_logpdf(x,np.zeros(p),np.eye(p)*1e-4,np.full(p,50.),nu,return_diagnostics=True)
                self.assertTrue(np.isfinite(symmetric).all());self.assertTrue(np.isfinite(skew).all())
                self.assertLessEqual(diag['last_refinement_max_log_difference'],1e-8)
                self.assertLessEqual(diag['tail_relative_bound'],1e-9)
    def test_skew_slash_dispatch_roundtrip_at_500(self):
        for p in [2,3,4,5]:
            nu=500-p/2;roundtrip=np.exp(np.log(nu))
            x=np.zeros((2,p));x[:,:2]=[[-10.,5.],[10.,-5.]]
            shape=np.zeros(p);shape[:2]=[2.,-.5]
            exact=skew_slash_logpdf(x,np.zeros(p),np.eye(p),shape,nu)
            actual,diagnostic=skew_slash_logpdf(x,np.zeros(p),np.eye(p),shape,roundtrip,return_diagnostics=True)
            np.testing.assert_allclose(actual,exact,rtol=0,atol=2e-10)
            self.assertNotEqual(diagnostic.get('method'),'large_nu_endpoint')
if __name__=='__main__':unittest.main()
