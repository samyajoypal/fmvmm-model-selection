"""Native-density regression checks for proposed GH-family component scores."""
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import _pack,_unpack,_bounds,prepare_component
from fmvmm.numerics.weighted_gh_score import values_gradient,NonsmoothLocationError
from fmvmm.mixtures.FMVMM import dist_map


def starting_parameters(name,p,order=.7,zero_gamma=False):
    mu=np.zeros(p);sigma=np.diag(np.linspace(.7,1.3,p))
    gamma=np.zeros(p) if zero_gamma else np.linspace(-.2,.3,p)
    if name=='mghp':return order,2.,3.,mu,sigma,gamma
    if name=='mgst':return -3.,4.,mu,sigma,gamma
    if name in ('mvhb','mnig'):return 2.,3.,mu,sigma,gamma
    if name=='mvvg':return p/2+order,2*(p/2+order),mu,sigma,gamma
    if name=='mvsl':return mu,sigma,gamma
    raise ValueError(name)


class TestGHWeightedScores(unittest.TestCase):
    def native_comparison(self,name,p,alpha,x,w,fixed=False):
        policy=EMPolicy();native,_=prepare_component(name,alpha,policy)
        u,_=_pack(name,native,policy,False);limits=np.asarray(_bounds(name,p,policy))
        ld,g,meta=values_gradient(x,u,name,policy,w,fixed_location=fixed,return_diagnostics=True)
        first=p if fixed else 0
        def objective(v):
            native,_=_unpack(np.r_[u[:first],v],name,p,policy)
            return float(w@dist_map[name].logpdf(x,*native))
        numeric=approx_derivative(objective,u[first:],method='3-point',bounds=limits[first:].T).reshape(-1)
        np.testing.assert_allclose(g[first:],numeric,rtol=2e-5,atol=2e-5)
        native,_=_unpack(u,name,p,policy)
        np.testing.assert_allclose(ld,dist_map[name].logpdf(x,*native),rtol=2e-10,atol=2e-9)
        if fixed:
            self.assertTrue(meta['location_fixed'])
            np.testing.assert_equal(g[:p],np.zeros(p))
        return u

    def test_smooth_native_density_scores(self):
        rng=np.random.default_rng(22048)
        for p in (2,5):
            for name in ('mghp','mvhb','mnig','mgst','mvvg','mvsl'):
                for zero in (False,True):
                    with self.subTest(p=p,name=name,zero_gamma=zero):
                        self.native_comparison(name,p,starting_parameters(name,p,zero_gamma=zero),
                            rng.normal(size=(30,p)),rng.uniform(.1,1,size=30))

    def test_center_limits_and_explicit_location_restriction(self):
        rng=np.random.default_rng(63324);policy=EMPolicy()
        for p in (2,5):
            for name,order in [('mvvg',.05),('mvvg',.25),('mvvg',.5),('mvvg',.75),('mvvg',1.),('mvvg',1.5),('mvsl',.5)]:
                with self.subTest(p=p,name=name,order=order):
                    x=np.r_[np.zeros((1,p)),rng.normal(size=(10,p))];w=rng.uniform(.1,1,size=len(x))
                    u=self.native_comparison(name,p,starting_parameters(name,p,order),x,w,True)
                    native,_=_unpack(u,name,p,policy)
                    actual_order=float(native[0])-p/2 if name=='mvvg' else .5
                    if actual_order<=.5:
                        with self.assertRaises(NonsmoothLocationError):values_gradient(x,u,name,policy,w)
                    else:
                        _,_,metadata=values_gradient(x,u,name,policy,w,return_diagnostics=True)
                        self.assertFalse(metadata['nonsmooth_location_center'])


if __name__=='__main__':unittest.main()
