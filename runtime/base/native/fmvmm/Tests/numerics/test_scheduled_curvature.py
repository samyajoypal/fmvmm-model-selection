"""Budget/dispatch regressions for periodic smooth curvature directions."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from fmvmm.numerics.component_budget import Budget,solve
from fmvmm.numerics.em_policy import EMPolicy

class ScheduledCurvatureTests(unittest.TestCase):
    def run_slow(self,maxiter,maxfun=10000,*,max_restarts=None):
        calls=[];limits=[]
        def objective(z):calls.append(z.copy());return float(.5*z@z),z.copy()
        def slow_minimize(fun,start,*,callback,options,**kwargs):
            limits.append(options['maxiter']);u=start.copy();f,g=fun(u)
            for _ in range(options['maxiter']):
                u*=.999;f,g=fun(u);callback(u)
            return SimpleNamespace(success=False,message='STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT',x=u,fun=f,jac=g)
        with patch('fmvmm.numerics.component_budget.minimize',slow_minimize):
            r=solve(objective,np.array([1.]),[(-2,2)],EMPolicy(),Budget(maxiter,maxfun),restart_ftol=0.,max_restarts=max_restarts)
        self.assertEqual(r['total_evaluations'],len(calls));self.assertLessEqual(r['total_iterations'],maxiter)
        self.assertLessEqual(r['total_evaluations'],maxfun)
        self.assertEqual(r['value'],objective(r['u'])[0]);np.testing.assert_array_equal(r['gradient'],r['u'])
        self.assertFalse(r['global_optimum_certified'])
        return r,limits

    def test_progressing_large_solve_uses_curvature_before_total_cap(self):
        r,limits=self.run_slow(2000)
        self.assertEqual(limits,[1000]);self.assertTrue(r['qualified'])
        self.assertEqual(r['total_iterations'],1001)
        self.assertEqual(r['ordinary_chunk_iterations'],100)
        self.assertEqual(r['runs'][0]['stop_scope'],'ordinary_chunk_iteration_limit')
        self.assertEqual(r['runs'][0]['remaining_iterations'],1000)
        self.assertGreater(r['runs'][0]['projected_gradient'],.1)
        self.assertLess(r['projected_gradient'],1e-6)
        self.assertEqual(r['curvature_rescues'][0]['trigger'],'initial_ordinary_stop')
        self.assertEqual(r['initial_ordinary_iteration_limit'],1000)
        self.assertEqual(r['runs'][0]['ordinary_phase'],'initial_ordinary')
        self.assertEqual(sum(x['iterations'] for x in r['runs']),r['total_iterations'])
        self.assertEqual(sum(x['evaluations'] for x in r['runs']),r['total_evaluations'])

    def test_component_sixty_schedule_unchanged(self):
        r,limits=self.run_slow(60)
        self.assertEqual(limits,[60]);self.assertFalse(r['qualified'])
        self.assertIsNone(r['ordinary_chunk_iterations']);self.assertEqual(r['curvature_rescues'],[])
        self.assertEqual(r['total_iterations'],60)
        self.assertEqual(r['runs'][0]['stop_scope'],'shared_budget_cap')

    def test_budget_at_most_one_thousand_keeps_unsplit_ordinary_schedule(self):
        r,limits=self.run_slow(1000)
        self.assertEqual(limits,[1000]);self.assertIsNone(r['ordinary_chunk_iterations'])
        self.assertEqual(r['curvature_rescues'],[])
        self.assertEqual(r['total_iterations'],1000)
        self.assertFalse(r['qualified'])

    def test_explicit_cusp_control_keeps_unsplit_solve(self):
        observed=[]
        def short_stop(fun,start,*,options,**kwargs):
            observed.append(options['maxiter']);f,g=fun(start)
            return SimpleNamespace(success=True,message='Controlled early ordinary stop',x=start,fun=f,jac=g)
        with patch('fmvmm.numerics.component_budget.minimize',short_stop):
            r=solve(lambda z:(float(z@z),2*z),np.array([.5]),[(-1,1)],EMPolicy(),Budget(2000,1000),restart_ftol=0.,max_restarts=0)
        self.assertEqual(observed,[2000]);self.assertEqual(r['stop_reason'],'return_control_for_conditional_update')
        self.assertIsNone(r['ordinary_chunk_iterations']);self.assertEqual(r['curvature_rescues'],[])

    def test_repeated_chunks_and_rejected_rescues_share_one_total_allowance(self):
        def unchanged_rescue(evaluate,u,value,g,bounds,budget,projected):
            budget.iterate()
            return {'accepted':False,'stop_reason':'controlled_no_curvature_progress',
                    'iterations':1,'evaluations':0,'global_optimum_certified':False}
        with patch('fmvmm.numerics.smooth_rescue.curvature_rescue',unchanged_rescue):
            r,limits=self.run_slow(1105)
        self.assertEqual(limits,[1000,100,3]);self.assertEqual(r['total_iterations'],1105)
        self.assertEqual(len(r['curvature_rescues']),2)
        self.assertTrue(all(not a['best_point_changed'] for a in r['curvature_rescues']))
        self.assertEqual(sum(a['iterations'] for a in r['runs']),1105)
        self.assertEqual(sum(a['evaluations'] for a in r['runs']),r['total_evaluations'])
        self.assertEqual(r['runs'][-1]['stop_scope'],'shared_budget_cap')
        self.assertEqual(r['stop_reason'],'shared_budget_exhausted')
        self.assertFalse(r['qualified'])

    def test_recorded_native_gh_profiles_from_declared_initializations(self):
        from fmvmm.numerics.constrained_mstep import _unpack,fit_component
        fixtures=json.loads(Path(__file__).with_name('scheduled_gh_profiles.json').read_text())
        for fixture in fixtures:
            with self.subTest(case=fixture['name']):
                x=np.asarray(fixture['X']);w=np.asarray(fixture['weights']);u=np.asarray(fixture['initial_u'])
                policy=EMPolicy(**fixture['policy']);alpha,_=_unpack(u,'mghp',x.shape[1],policy)
                candidate,record=fit_component('mghp',x,w,alpha,policy,maxiter=2000,coordinates_prev=u)
                self.assertTrue(record['gradient_acceptable']);self.assertTrue(record['accepted'])
                self.assertLess(record['iterations'],2000);self.assertLessEqual(record['evaluations'],10000)
                self.assertGreater(record['weighted_loglik'],fixture['frozen_unqualified_weighted_loglik'])
                self.assertTrue(any(a.get('method')=='curvature_rescue' for a in record['optimizer_runs']))
                self.assertFalse(record['global_optimum_certified'])

    def test_partial_hessian_cap_is_not_reset_or_misreported(self):
        # 1001 ordinary requests + one Hessian probe, then no evaluation remains.
        r,limits=self.run_slow(2000,1002)
        self.assertFalse(r['qualified']);self.assertEqual(r['total_evaluations'],1002)
        self.assertEqual(r['total_iterations'],1001)
        rescue=r['curvature_rescues'][0]
        self.assertEqual(rescue['stop_reason'],'shared_evaluation_budget')
        self.assertEqual(rescue['completed_hessian_columns'],0)
        self.assertEqual(rescue['evaluations'],1)
        self.assertFalse(rescue['best_point_changed'])
        self.assertFalse(rescue['accepted'])
        self.assertEqual(r['stop_reason'],'shared_budget_exhausted')

if __name__=='__main__':unittest.main()
