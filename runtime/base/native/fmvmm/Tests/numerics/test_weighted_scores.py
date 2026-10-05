"""Independent native-density checks for analytic and semi-analytic SMSN scores."""
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import prepare_component, _pack, _unpack, _bounds, _analytic
from fmvmm.numerics.weighted_msnc_score import values_gradient as msnc_score
from fmvmm.numerics.weighted_skew_t_score import values_gradient as skew_t_score
from fmvmm.mixtures.FMVMM import dist_map


class TestWeightedScores(unittest.TestCase):
    def test_native_objective_derivatives(self):
        rng=np.random.default_rng(7123);policy=EMPolicy()
        for name,helper in [('msnc',msnc_score),('mvst',skew_t_score)]:
            for p in (2,5):
                for parameter in ((.2,.3),(.01,1.)) if name=='msnc' else (0.,.2,5.):
                    with self.subTest(name=name,p=p,parameter=parameter):
                        x=rng.normal(size=(40,p));w=rng.uniform(.1,1,size=40)
                        tail=np.array(parameter) if name=='msnc' else np.array([np.inf if parameter==0 else 1/parameter])
                        alpha=(np.zeros(p),np.eye(p),np.linspace(-.3,.4,p),tail)
                        alpha,_=prepare_component(name,alpha,policy);u,_=_pack(name,alpha,policy,False)
                        ld,g=helper(x,u,policy,w)
                        def objective(v):
                            native,_=_unpack(v,name,p,policy)
                            return float(w@dist_map[name].logpdf(x,*native))
                        native,_=_unpack(u,name,p,policy)
                        np.testing.assert_allclose(ld,dist_map[name].logpdf(x,*native),rtol=2e-10,atol=2e-9)
                        numeric=approx_derivative(objective,u,method='3-point',bounds=np.asarray(_bounds(name,p,policy)).T).reshape(-1)
                        np.testing.assert_allclose(g,numeric,rtol=2e-5,atol=2e-5)

    def test_skew_t_zero_skew_reduces_to_analytic_student_score(self):
        rng=np.random.default_rng(8181);policy=EMPolicy()
        for p in (2,5):
            for eta in (0.,1e-10,.2,5.):
                with self.subTest(p=p,eta=eta):
                    x=rng.normal(size=(40,p))*10;w=rng.uniform(.1,1,size=40)
                    alpha=(np.zeros(p),np.eye(p),np.zeros(p),np.array([np.inf if eta==0 else 1/eta]))
                    alpha,_=prepare_component('mvst',alpha,policy);u,_=_pack('mvst',alpha,policy,False)
                    ld,g=skew_t_score(x,u,policy,w)
                    t_u=np.r_[u[:-(p+1)],eta]
                    t_ld,t_g=_analytic(x,t_u,'mvt',policy,w)
                    np.testing.assert_allclose(ld,t_ld,rtol=2e-10,atol=2e-9)
                    np.testing.assert_allclose(np.r_[g[:-(p+1)],g[-1]],t_g,rtol=2e-5,atol=2e-5)


if __name__=='__main__':unittest.main()
