"""Regression tests for exactly tied float objectives; no scientific model fits."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.component_budget import Budget,solve
from fmvmm.numerics.constrained_mstep import fit_component,prepare_component,_pack,_unpack
from fmvmm.numerics.cusp_stationarity import anchor_continuation
from fmvmm.mixtures.FMVMM import dist_map

HESSIAN=1e8
START=2e-12

def flat_float_objective(value):return 5.+.5*HESSIAN*value**2


class ExactTieTests(unittest.TestCase):
    def test_solver_keeps_equally_valued_stationary_endpoint(self):
        result=solve(lambda u:(flat_float_objective(u[0]),np.array([HESSIAN*u[0]])),
            np.array([START]),[(-1.,1.)],EMPolicy(),Budget(100,1000),restart_ftol=0.)
        self.assertEqual(result['value'],5.)
        self.assertTrue(result['qualified'])
        self.assertLess(result['projected_gradient'],1e-6)
        self.assertEqual(result['total_iterations'],1)

    def test_component_tracker_does_not_discard_the_solver_endpoint(self):
        x=np.zeros((4,2));w=np.ones(4);policy=EMPolicy()
        alpha,_=prepare_component('mvn',(np.array([START,0.]),np.eye(2)),policy)
        def native(x,mu,sigma):return np.full(len(x),-flat_float_objective(mu[0]))
        def analytic(x,u,name,policy,w):
            g=np.zeros(len(u));g[0]=-w.sum()*HESSIAN*u[0]
            return np.full(len(x),-flat_float_objective(u[0])),g
        with patch.object(dist_map['mvn'],'logpdf',native),patch('fmvmm.numerics.constrained_mstep._analytic',analytic):
            candidate,record=fit_component('mvn',x,w,alpha,policy,maxiter=100)
        self.assertTrue(record['gradient_acceptable']);self.assertTrue(record['accepted'])
        self.assertEqual(record['weighted_loglik'],record['weighted_loglik_before'])
        self.assertLess(abs(candidate[0][0]),1e-14)

    def test_location_fixed_tracker_uses_its_conditional_score(self):
        p=2;x=np.zeros((4,p));w=np.ones(4);policy=EMPolicy();index=p+p*(p+1)//2
        alpha,_=prepare_component('mvsl',(np.zeros(p),np.eye(p),np.zeros(p)),policy)
        u,_=_pack('mvsl',alpha,policy,False);u[index]=START
        def native(x,mu,sigma,gamma):return np.full(len(x),-flat_float_objective(gamma[0]*(p+1)))
        def score(x,z,name,policy,w,*,fixed_location=False):
            g=np.zeros(len(z));g[index]=-w.sum()*HESSIAN*z[index]
            return np.full(len(x),-flat_float_objective(z[index])),g
        def objective(z):return flat_float_objective(z[index])
        with patch.object(dist_map['mvsl'],'logpdf',native),patch('fmvmm.numerics.cusp_stationarity.values_gradient',score):
            candidate,records=anchor_continuation(x,w,u,'mvsl',policy,100,objective)
        self.assertEqual(objective(candidate),objective(u))
        self.assertLess(abs(candidate[index]),1e-14)
        self.assertTrue(records[0]['stationarity']['stationarity_acceptable'])

    def test_one_ulp_worse_objective_is_never_promoted(self):
        def objective(u):
            return (1.,np.array([2e-4])) if u[0]==1. else (np.nextafter(1.,np.inf),np.array([0.]))
        def artificial_minimize(fun,start,*,callback,**kwargs):
            fun(np.array([1.]));value,g=fun(np.array([0.]));callback(np.array([0.]))
            return SimpleNamespace(success=True,message='Synthetic endpoint',fun=value,jac=g,x=np.array([0.]))
        with patch('fmvmm.numerics.component_budget.minimize',artificial_minimize):
            result=solve(objective,np.array([1.]),[(-2.,2.)],EMPolicy(),Budget(1,10),restart_ftol=0.)
        self.assertEqual(result['value'],1.);self.assertEqual(result['u'][0],1.)
        self.assertFalse(result['qualified'])

if __name__=='__main__':unittest.main()
