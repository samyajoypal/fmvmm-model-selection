import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import prepare_component,_pack,_unpack,_bounds
from fmvmm.numerics.weighted_slash_score import values_gradient
from fmvmm.mixtures.FMVMM import dist_map


class SlashScoreTests(unittest.TestCase):
    def test_native_gradient_and_latent_bounds(self):
        policy=EMPolicy();rng=np.random.default_rng(53011)
        for p in (2,3,4,5):
            for name in ('msl','mssl'):
                for nu in (.001,.5,4.,450.):
                    with self.subTest(p=p,name=name,nu=nu):
                        x=rng.normal(size=(12,p))*3;w=rng.uniform(.1,1,len(x))
                        shape=np.zeros(p);shape[0]=2.
                        alpha=(np.zeros(p),np.eye(p),np.array([nu])) if name=='msl' else (np.zeros(p),np.eye(p),shape,np.array([nu]))
                        alpha,_=prepare_component(name,alpha,policy);u,_=_pack(name,alpha,policy,False)
                        bounds=np.array(_bounds(name,p,policy))
                        values,gradient,diagnostic=values_gradient(x,u,name,policy,w,return_diagnostics=True)
                        def objective(z):
                            a,_=_unpack(z,name,p,policy)
                            return float(w@dist_map[name].logpdf(x,*a))
                        fd=approx_derivative(objective,u,method='3-point',bounds=(bounds[:,0],bounds[:,1])).reshape(-1)
                        self.assertLess(np.max(np.abs(gradient-fd)/(1+np.abs(fd))),2e-6)
                        self.assertAlmostEqual(float(w@values),objective(u),places=9)
                        self.assertGreaterEqual(diagnostic['latent_mean_u_min'],0.)
                        self.assertLessEqual(diagnostic['latent_mean_u_max'],1.+1e-8)

    def test_zero_skew_spatial_reduction_and_nonzero_shape_derivative(self):
        policy=EMPolicy();p=3;x=np.array([[.3,-.5,2.],[2.,.7,-.1],[-.5,1.,.7]])
        w=np.array([.2,.5,.3]);nu=np.array([2.]);base=(np.zeros(p),np.eye(p))
        a,_=prepare_component('msl',(*base,nu),policy);u,_=_pack('msl',a,policy,False)
        b,_=prepare_component('mssl',(*base,np.zeros(p),nu),policy);v,_=_pack('mssl',b,policy,False)
        values,g=values_gradient(x,u,'msl',policy,w);skew,h=values_gradient(x,v,'mssl',policy,w)
        np.testing.assert_allclose(skew,values,atol=1e-12)
        np.testing.assert_allclose(h[:len(g)-1],g[:-1],atol=1e-10)
        self.assertGreater(np.linalg.norm(h[len(g)-1:-1]),.01)


if __name__=='__main__': unittest.main()
