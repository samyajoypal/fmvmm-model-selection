"""Independent convex objectives exercise the generic stalled-solver rescue."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.component_budget import Budget,solve
from fmvmm.numerics.weighted_gh_score import NonsmoothLocationError


def stalled_minimize(fun,start,**kwargs):
    value,g=fun(start)
    return SimpleNamespace(success=False,message='Synthetic unresolved smooth stall',fun=value,jac=g,x=start)

class CurvatureRescueTests(unittest.TestCase):
    def run_stalled(self,fn,start,bounds,iterations=10,evaluations=1000,**kwargs):
        calls=[]
        def observed(u):
            self.assertTrue(np.all(np.asarray(u)>=np.asarray(bounds)[:,0]))
            self.assertTrue(np.all(np.asarray(u)<=np.asarray(bounds)[:,1]))
            calls.append(np.array(u));return fn(u)
        with patch('fmvmm.numerics.component_budget.minimize',stalled_minimize):
            result=solve(observed,np.asarray(start,float),bounds,EMPolicy(),Budget(iterations,evaluations),restart_ftol=0.,**kwargs)
        self.assertEqual(result['total_evaluations'],len(calls))
        self.assertLessEqual(result['total_iterations'],iterations)
        self.assertLessEqual(len(calls),evaluations)
        self.assertFalse(result['global_optimum_certified'])
        return result,calls

    def test_ill_conditioned_positive_quadratic(self):
        for ratio in [1.,1e5,1e9]:
            with self.subTest(ratio=ratio):
                A=np.diag([ratio,1.])
                fn=lambda z:(float(.5*z@A@z),A@z)
                start=np.array([1e-4/ratio,.01])
                result,calls=self.run_stalled(fn,start,[(-1,1)]*2)
                self.assertTrue(result['qualified']);self.assertLess(result['value'],fn(start)[0])
                self.assertEqual(len(result['curvature_rescues']),1)
                self.assertEqual(result['total_iterations'],1)
                self.assertLess(np.max(abs(result['u'])),1e-8)

    def test_outward_active_gradient_stays_fixed(self):
        def fn(z):return ((z[0]+1)**2+(z[1]-.5)**2,np.array([2*(z[0]+1),2*(z[1]-.5)]))
        result,calls=self.run_stalled(fn,[0.,.1],[(0,1),(-1,1)])
        self.assertTrue(result['qualified']);self.assertEqual(result['u'][0],0.)
        self.assertTrue(all(z[0]==0. for z in calls))
        self.assertEqual(result['curvature_rescues'][0]['free_coordinates'],[1])

    def test_inward_bound_uses_feasible_one_sided_differences(self):
        fn=lambda z:((z[0]-.1)**2,np.array([2*(z[0]-.1)]))
        result,calls=self.run_stalled(fn,[0.],[(0,1)])
        self.assertTrue(result['qualified']);self.assertAlmostEqual(result['u'][0],.1,places=7)
        self.assertEqual(result['curvature_rescues'][0]['difference_schemes'][0]['scheme'],'forward')

    def test_no_fresh_budget_for_hessian_or_line_search(self):
        fn=lambda z:(float(z@z),2*z)
        result,calls=self.run_stalled(fn,[.5,.5],[(-1,1)]*2,iterations=1,evaluations=5)
        self.assertEqual(result['total_evaluations'],5)
        self.assertEqual(result['total_iterations'],1)
        self.assertFalse(result['qualified'])
        self.assertEqual(result['curvature_rescues'][0]['stop_reason'],'shared_evaluation_budget')
        self.assertEqual(result['curvature_rescues'][0]['completed_hessian_columns'],1)
        self.assertFalse(result['curvature_rescues'][0]['accepted'])
        self.assertTrue(result['curvature_rescues'][0]['best_point_changed'])
        self.assertEqual(result['best_point_source'],'curvature_probe')
        self.assertEqual(result['value'],min(fn(z)[0] for z in calls))
        self.assertEqual(result['value'],fn(result['u'])[0])
        np.testing.assert_array_equal(result['gradient'],fn(result['u'])[1])

    def test_cusp_control_return_does_not_launch_rescue(self):
        fn=lambda z:(float(z@z),2*z)
        result,calls=self.run_stalled(fn,[.5],[(-1,1)],max_restarts=0)
        self.assertEqual(result['curvature_rescues'],[])
        self.assertEqual(result['stop_reason'],'return_control_for_conditional_update')

    def test_recorded_native_gh_stall_is_repaired_without_objective_loss(self):
        from fmvmm.numerics.constrained_mstep import _unpack,fit_component
        fixture=json.loads((Path(__file__).with_name('frozen_gh_stall.json')).read_text())
        x=np.asarray(fixture['X']);w=np.asarray(fixture['weights']);u=np.asarray(fixture['u'])
        policy=EMPolicy(**fixture['policy']);alpha,_=_unpack(u,'mghp',x.shape[1],policy)
        candidate,record=fit_component('mghp',x,w,alpha,policy,coordinates_prev=u)
        self.assertTrue(record['gradient_acceptable']);self.assertTrue(record['accepted'])
        self.assertGreater(record['weighted_loglik'],record['weighted_loglik_before'])
        self.assertLessEqual(record['iterations'],policy.component_maxiter)
        self.assertLessEqual(record['evaluations'],policy.component_maxfun)
        self.assertTrue(any(r.get('method')=='curvature_rescue' for r in record['optimizer_runs']))
        self.assertFalse(record['global_optimum_certified'])

    def test_exact_cusp_exception_is_not_swallowed(self):
        calls=[0]
        def fn(z):
            calls[0]+=1
            if calls[0]>2:raise NonsmoothLocationError('No ordinary location derivative')
            return float(z@z),2*z
        with self.assertRaises(NonsmoothLocationError):
            self.run_stalled(fn,[.5],[(-1,1)])

if __name__=='__main__':unittest.main()
