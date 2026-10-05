"""Joint moments checked with a separate adaptive full-integral oracle."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import _pack,_unpack,prepare_component
from fmvmm.numerics.weighted_slash_score import values_gradient
from fmvmm.numerics.weighted_slash_joint_score import values_gradient as joint
from fmvmm.numerics.robust_slash import NumericalIntegrationError

LOG2PI=np.log(2*np.pi)


def adaptive_moments(a,rate,shape):
    """Integrate the full latent domain using t=-log U, no Jacobi rule/tail cut."""
    total_rate=rate+.5*min(shape,0.)**2
    upper=max(2.,np.log(max(total_rate,1.)/a)+max(5.,20/a))
    def logf(t):return -a*t-rate*np.exp(-t)+log_ndtr(shape*np.exp(-t/2))
    opt=minimize_scalar(lambda t:-logf(t),bounds=(0,upper),method='bounded')
    mode=float(opt.x) if logf(opt.x)>logf(0.) else 0.;peak=logf(mode)
    scale=a if mode<.1/a else max(1.,np.sqrt(a))
    lower=-mode*scale
    def integrand(v,index):
        t=mode+v/scale
        if index==3:
            return np.exp(-(a+.5)*t-(rate+.5*shape**2)*np.exp(-t)-.5*LOG2PI-peak)/scale
        base=np.exp(logf(t)-peak)/scale
        return base*(1. if index==0 else np.exp(-t) if index==1 else -t)
    integrals=[]
    for index in range(4):
        first=(0.,0.) if lower==0 else quad(lambda t:integrand(t,index),lower,0.,epsabs=1e-13,epsrel=3e-12,limit=300)
        second=quad(lambda t:integrand(t,index),0.,np.inf,epsabs=1e-13,epsrel=3e-12,limit=300)
        value=first[0]+second[0];error=first[1]+second[1]
        if error>1e-9*max(abs(value),integrals[0] if integrals else abs(value)):
            raise AssertionError('Independent adaptive moment integration did not converge.')
        integrals.append(value)
    I=integrals[0]
    return peak+np.log(I),integrals[1]/I,integrals[2]/I,integrals[3]/I


def oracle(x,u,policy,w):
    p=x.shape[1];alpha,L=_unpack(u,'mssl',p,policy);mu,sigma,_,tail=alpha
    nu=float(tail[0]);spatial=p+p*(p+1)//2;b=u[spatial:spatial+p]
    z=x-mu;inverse=np.linalg.inv(sigma);s=z@inverse
    moments=np.array([adaptive_moments(nu+p/2,.5*d@inverse@d,float(d@b)) for d in z])
    logI,EU,Elog,H=moments.T
    values=np.log(2*nu)-p*LOG2PI/2-np.linalg.slogdet(sigma)[1]/2+logI
    G=.5*(s.T@((w*EU)[:,None]*s)-w.sum()*inverse);dL=2*G@L
    gradient=np.r_[(w*EU)@s-(w@H)*b,np.diag(dL)*np.diag(L),dL[np.tril_indices(p,-1)],z.T@(w*H),w@(1+nu*Elog)]
    return values,gradient


class JointSlashTests(unittest.TestCase):
    def make_case(self,p,nu,*,floor=False,zero_skew=False):
        policy=replace(EMPolicy(),slash_df_max=max(450.,nu));sigma=np.eye(p)*(1.000001e-4 if floor else 1.)
        shape=np.zeros(p) if zero_skew else np.r_[2.,-.5,np.zeros(p-2)]
        alpha,_=prepare_component('mssl',(np.zeros(p),sigma,shape,np.array([nu])),policy)
        u,_=_pack('mssl',alpha,policy,False)
        x=np.array([np.zeros(p),np.linspace(-2.,1.,p),np.linspace(1.,-2.,p)])
        if floor:x[1:]*=15
        return policy,u,x,np.array([.2,.9,.7])

    def check_oracle(self,p,nu,**kwargs):
        policy,u,x,w=self.make_case(p,nu,**kwargs)
        ld,g,d=joint(x,u,policy,w);ref,score=oracle(x,u,policy,w)
        np.testing.assert_allclose(ld,ref,rtol=0,atol=3e-8)
        self.assertLess(np.max(abs(g-score)/(1+abs(score))),2e-6)
        self.assertLessEqual(d['gradient_tail_bound'],1e-8)
        self.assertLessEqual(d['density_tail_bound'],1e-9)
        self.assertLessEqual(d['gradient_refinement'],1e-7)

    def test_independent_adaptive_oracle_p2_through_p5(self):
        for p in (2,3,4,5):
            for nu in (.001,.5,450.,500-p/2):
                with self.subTest(p=p,nu=nu):self.check_oracle(p,nu)

    def test_adaptive_oracle_covariance_floor_and_zero_skew(self):
        for p in (2,5):
            for nu in (.001,450.):
                with self.subTest(p=p,nu=nu):self.check_oracle(p,nu,floor=True)
                with self.subTest(p=p,nu=nu,zero_skew=True):self.check_oracle(p,nu,zero_skew=True)

    def test_exact_origin_tail_and_shape_scores(self):
        for p in (2,5):
            for nu in (.001,450.,500-p/2):
                policy,u,x,w=self.make_case(p,nu);x[:]=0
                _,g,d=joint(x,u,policy,w)
                self.assertAlmostEqual(g[-1],w.sum()*p/(2*(nu+p/2)),places=11)
                np.testing.assert_array_equal(g[-p-1:-1],np.zeros(p))
                self.assertAlmostEqual(d['latent_mean_u_min'],(nu+p/2)/(nu+p/2+1),places=14)

    def test_explicit_original_method_fallback(self):
        policy,u,x,w=self.make_case(2,499.)
        with patch('fmvmm.numerics.weighted_slash_joint_score.values_gradient',side_effect=NumericalIntegrationError('deliberate unsupported refinement')):
            ld,g,d=values_gradient(x,u,'mssl',policy,w,return_diagnostics=True)
        self.assertTrue(np.isfinite(ld).all());self.assertTrue(np.isfinite(g).all())
        self.assertIn('joint_score_fallback_reason',d)
        self.assertEqual(d['gradient_method'],'native_finite_difference_at_shifted_moment_domain_edge')
        policy,u,x,w=self.make_case(2,4.)
        with patch('fmvmm.numerics.weighted_slash_joint_score._jacobi_rule',side_effect=ValueError('deliberate SciPy rule range error')):
            _,g,d=values_gradient(x,u,'mssl',policy,w,return_diagnostics=True)
        self.assertTrue(np.isfinite(g).all())
        self.assertIn('quadrature rule construction failed',d['joint_score_fallback_reason'])

    def test_invalid_memberships_are_not_a_fallback(self):
        policy,u,x,w=self.make_case(2,4.)
        for bad in (-w,np.zeros_like(w)):
            with self.assertRaises(ValueError):values_gradient(x,u,'mssl',policy,bad)

if __name__=='__main__':unittest.main()
