import unittest
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.special import betaln, gammaln, log_ndtr, logsumexp
from scipy.stats import multivariate_t, multivariate_normal, t, norm, genhyperbolic, geninvgauss, gamma, invgamma
from fmvmm.distributions import multivariate_t as mvt
from fmvmm.distributions import multivariate_skew_t_smsn as mvst
from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.distributions import multivariate_genhyperbolic as mvgh
from fmvmm.distributions import multivariate_slash as mvs
from fmvmm.numerics.robust_skew_t import student_logcdf
from fmvmm.numerics.robust_probability import skew_normal_logpdf
from fmvmm.numerics.robust_gh import gh_logpdf,log_bessel_k


def beta_tail_oracle(x, df):
    """Positive incomplete-beta series, separate from the production integral."""
    a=df/2;z=df/(df+x*x)
    assert z<.5
    # B_z(a,1/2)=z^a/a sum (1/2)_k z^k/k! * a/(a+k).
    term=1.;total=1.
    for k in range(1,10000):
        term *= (k-.5)/k*z
        change=term*a/(a+k);total+=change
        if change<1e-16*total:break
    return -np.log(2)+a*np.log(z)-np.log(a)-betaln(a,.5)+np.log(total)


def mixture_oracle(x,lmbda,chi,psi,mu,sigma,gam):
    """Independent normal/GIG mixing integral in log(w), with Scipy densities."""
    if chi==0:
        mix=lambda w: gamma.logpdf(w,a=lmbda,scale=2/psi)
    elif psi==0:
        mix=lambda w: invgamma.logpdf(w,a=-lmbda,scale=chi/2)
    else:
        mix=lambda w:geninvgauss.logpdf(w,p=lmbda,b=np.sqrt(chi*psi),scale=np.sqrt(chi/psi))
    def integrand(u):
        w=np.exp(u)
        return np.exp(multivariate_normal.logpdf(x,mu+w*gam,w*sigma)+mix(w)+u)
    lower = -350 if chi==0 and np.array_equal(x,mu) else -35
    value,error=quad(integrand,lower,35,epsabs=1e-13,epsrel=1e-10,limit=300)
    assert value>0 and error<1e-8*value,(value,error)
    return np.log(value)


class IntegrationTests(unittest.TestCase):
    def test_skew_t_reference(self):
        rng=np.random.default_rng(93174)
        for p in [1,2,5]:
            matrix=rng.normal(size=(p,p));sigma=matrix@matrix.T+np.eye(p)
            x=rng.normal(size=(30,p));mu=rng.normal(size=p);shape=rng.normal(size=p)
            y=x-mu;invs=np.linalg.inv(sigma)
            arg=y@np.linalg.inv(sqrtm(sigma))@shape
            q=np.sum(y@invs*y,axis=1)
            for df in [.2,1,5,30,100,10000]:
                expected=np.log(2)+multivariate_t.logpdf(x,mu,sigma,df)+t.logcdf(arg*np.sqrt((df+p)/(df+q)),df+p)
                actual=mvst.logpdf(x,mu,sigma,shape,np.array([df]))
                np.testing.assert_allclose(actual,expected,atol=3e-11,rtol=3e-12)

    def test_skew_t_limits_and_underflow(self):
        for p in [1,2,5]:
            x=np.array([np.zeros(p),np.ones(p),-np.ones(p),-1000*np.ones(p)])
            mu=np.zeros(p);sigma=np.eye(p);shape=np.arange(p)+1.
            np.testing.assert_allclose(mvst.logpdf(x,mu,sigma,np.zeros(p),1e16),mvt.logpdf(x,mu,sigma,1e16),atol=1e-10)
            np.testing.assert_allclose(mvst.logpdf(x,mu,sigma,shape,np.array([np.inf])),skew_normal_logpdf(x,mu,sigma,shape),atol=1e-10)
            self.assertTrue(np.isfinite(mvst.logpdf(x,mu,sigma,shape,10000)).all())
            np.testing.assert_allclose(mvst.pdf(x,mu,sigma,shape,5),np.exp(mvst.logpdf(x,mu,sigma,shape,5)),atol=0)

    def test_student_logcdf_independent_tail(self):
        for df in [.1,1,5,100,100000]:
            for x in [10000,1e8]:
                expected=beta_tail_oracle(x,df)
                actual=student_logcdf(-x,df)
                self.assertLess(abs(actual-expected),max(1e-9,abs(expected)*2e-13),(df,x,actual,expected))
        for x in [-1,-10,-40,-100]:
            self.assertAlmostEqual(student_logcdf(x,1e16),log_ndtr(x),places=7)
        self.assertEqual(student_logcdf(-np.inf,4),-np.inf)
        self.assertEqual(student_logcdf(np.inf,4),0)
        self.assertAlmostEqual(student_logcdf(0,4),-np.log(2))

    def test_gh_against_scipy_univariate(self):
        x=np.array([[-10.],[-1.],[0.],[2.],[1000.]])
        for lmbda in [-4,-.5,0,2,8]:
            for chi,psi in [(.01,.03),(1.,2.),(200.,200.)]:
                sigma=np.array([[1.3]]);gam=np.array([.2]);mu=np.array([.4])
                scale=np.sqrt(chi*sigma[0,0]);a=np.sqrt(chi*(psi+gam[0]**2/sigma[0,0]));b=gam[0]*np.sqrt(chi/sigma[0,0])
                expected=genhyperbolic.logpdf(x[:,0],lmbda,a,b,loc=mu[0],scale=scale)
                actual=gh_logpdf(x,lmbda,chi,psi,mu,sigma,gam)
                np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=2e-12)

    def test_gh_against_normal_mixture_integral(self):
        for p in [1,2,5]:
            sigma=np.eye(p)+.1;mu=np.zeros(p);gam=np.ones(p)*.2
            for lmbda,chi,psi in [(-.5,1.,2.),(2.,.2,1.),(p/2+.1,0.,3.),(-3.,4.,0.)]:
                for x in [np.zeros(p),np.linspace(-.5,.9,p)]:
                    expected=mixture_oracle(x,lmbda,chi,psi,mu,sigma,gam)
                    actual=gh_logpdf(x,lmbda,chi,psi,mu,sigma,gam)
                    self.assertLess(abs(actual-expected),1e-8,(p,lmbda,chi,psi,x,actual,expected))

    def test_gh_boundaries_and_centers(self):
        for p in [1,2,5]:
            sigma=np.eye(p);mu=np.zeros(p);x=np.array([mu,np.ones(p),10*np.ones(p)])
            for df in [.1,1.,5.,100.,1e16]:
                np.testing.assert_allclose(gh_logpdf(x,-df/2,df,0,mu,sigma,mu),mvt.logpdf(x,mu,sigma,df),atol=1e-8,rtol=2e-12)
            self.assertEqual(gh_logpdf(mu,p/2,0,2,mu,sigma,mu),np.inf)
            self.assertEqual(gh_logpdf(mu,p/2-.1,0,2,mu,sigma,mu),np.inf)
            self.assertTrue(np.isfinite(gh_logpdf(mu,p/2+.05,0,2,mu,sigma,mu)))
            # Continuity to the exact finite variance-gamma center.
            center=gh_logpdf(mu,p/2+2,0,2,mu,sigma,mu)
            near=gh_logpdf(np.full(p,1e-8),p/2+2,0,2,mu,sigma,mu)
            self.assertLess(abs(center-near),1e-12)
        self.assertTrue(np.isfinite(log_bessel_k(3,1000)))
        with self.assertRaises(ValueError):gh_logpdf([[0]],1,1,-1,[0],[[1]],[0])

    def test_gh_identifiable_mapping(self):
        from fmvmm.distributions.multivariate_genhyperbolic import _alphabar2chipsi
        from scipy.special import kve
        for lmbda in [-10,-2,-1,-.5,0,2,10]:
            for abar in [.001,.1,1.,200.,1000.,1e6]:
                chi,psi=_alphabar2chipsi(abar,lmbda)
                self.assertLess(abs(np.sqrt(chi*psi)/abar-1),1e-13)
                expected_mean=np.sqrt(chi/psi)*kve(lmbda+1,abar)/kve(lmbda,abar)
                self.assertLess(abs(expected_mean-1),1e-13)
        self.assertEqual(_alphabar2chipsi(0,2),(0.,4.))
        self.assertEqual(_alphabar2chipsi(0,-2),(2.,0.))
        for lmbda in [-1,-.5,0]:
            with self.assertRaises(ValueError):_alphabar2chipsi(0,lmbda)
        self.assertGreater(_alphabar2chipsi(1e-16,2)[0],0)

    def test_skew_laplace_normalizer_center(self):
        for p in [1,2,5]:
            sigma=np.eye(p)+.1;mu=np.zeros(p);gam=np.ones(p)*.3
            invs=np.linalg.inv(sigma);a=np.sqrt(1+gam@invs@gam)
            const=(-.5*np.linalg.slogdet(sigma)[1]-p*np.log(2)-(p-1)/2*np.log(np.pi)-np.log(a)-gammaln((p+1)/2))
            x=np.array([mu,np.ones(p),-np.ones(p)*1000]);q=np.sum(x@invs*x,axis=1)
            expected=const-a*np.sqrt(q)+x@invs@gam
            np.testing.assert_allclose(mvsl.logpdf(x,mu,sigma,gam),expected,atol=1e-9,rtol=2e-12)
            self.assertAlmostEqual(mvsl.logpdf(mu,mu,sigma,gam),const,places=12)

    def test_wrappers_reject_invalid_full_dimensional_density(self):
        for fun,args in [(mvst.logpdf,([0,0],np.diag([1,0]),[1,1],5)),(mvgh.logpdf,(1,1,1,[0,0],np.diag([1,0]),[0,0])),(mvsl.logpdf,([0,0],np.diag([1,0]),[0,0]))]:
            with self.assertRaises(ValueError):fun([[0,0]],*args)

if __name__=='__main__':unittest.main()
