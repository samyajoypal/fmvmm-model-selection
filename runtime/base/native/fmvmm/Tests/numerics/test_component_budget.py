"""Budget sharing, stricter restart stopping, and retained-coordinate regressions."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import rosen,rosen_der
from fmvmm.numerics.component_budget import Budget,solve
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import component_stationarity,fit_component,prepare_component,_pack
from fmvmm.mixtures.FMVMM import fmvmm


class ComponentBudgetTests(unittest.TestCase):
    def test_restart_tightens_objective_stop_with_one_total_budget(self):
        policy=EMPolicy(ftol=1e-2)
        budget=Budget(100,300)
        result=solve(lambda x:(rosen(x),rosen_der(x)),np.array([-1.2,1.]),[(-3.,3.)]*2,policy,budget,restart_ftol=policy.restart_ftol)
        self.assertGreater(result['restarts'],0);self.assertTrue(result['qualified'])
        self.assertEqual(result['runs'][0]['ftol'],policy.ftol)
        self.assertTrue(all(r['ftol']==0 for r in result['runs'][1:]))
        self.assertEqual(sum(r['iterations'] for r in result['runs']),budget.iterations)
        self.assertLessEqual(budget.iterations,100);self.assertLessEqual(budget.evaluations,300)

    def test_iteration_and_evaluation_budgets_are_shared_between_solves(self):
        policy=EMPolicy();budget=Budget(4,200)
        first=solve(lambda x:(rosen(x),rosen_der(x)),np.array([-1.2,1.]),[(-3.,3.)]*2,policy,budget,restart_ftol=0.)
        self.assertEqual(budget.iterations,4);before=budget.evaluations
        second=solve(lambda x:(_ for _ in ()).throw(AssertionError('No iterations remain')),first['u'],[(-3.,3.)]*2,policy,budget,restart_ftol=0.)
        self.assertEqual(budget.evaluations,before);self.assertFalse(first['qualified'])
        budget=Budget(100,2);calls=[]
        def counted(x):calls.append(x.copy());return rosen(x),rosen_der(x)
        result=solve(counted,np.array([-1.2,1.]),[(-3.,3.)]*2,policy,budget,restart_ftol=0.)
        self.assertEqual(len(calls),2);self.assertEqual(budget.evaluations,2);self.assertFalse(result['qualified'])

    def test_cusp_optimizations_share_original_iteration_budget(self):
        x=np.array([[0.,0.],[.7,.2],[-.5,-.4],[.1,.8]])
        policy=EMPolicy(component_maxiter=3)
        alpha,_=prepare_component('mvvg',(1.3,2.6,np.zeros(2),np.eye(2),np.zeros(2)),policy)
        _,record=fit_component('mvvg',x,np.ones(4),alpha,policy)
        self.assertLessEqual(record['iterations'],3)
        self.assertLessEqual(record['evaluations'],policy.component_maxfun)
        self.assertGreaterEqual(record['weighted_loglik'],record['weighted_loglik_before'])
        self.assertEqual(record['iteration_budget'],3)

    def test_native_nig_roundtrip_uses_exact_internal_coordinates(self):
        data=json.loads((Path(__file__).parent/'fixtures/coordinate_roundtrips.json').read_text())['nig']
        alpha=data['alpha'];u=np.array(data['original_u']);p=data['p']
        x=np.asarray(alpha[-3])+np.random.default_rng(14092).normal(size=(20,p))
        with patch('fmvmm.numerics.constrained_mstep._pack',side_effect=AssertionError('Exact coordinates must bypass re-encoding')):
            state=component_stationarity('mnig',x,np.ones(len(x)),alpha,EMPolicy(),coordinates=u)
        self.assertEqual(state['coordinate_recovery']['source'],'retained_optimizer_coordinates')
        self.assertTrue(np.isfinite(state['projected_gradient']))

    def test_public_em_retains_coordinates_for_final_q_and_next_mstep(self):
        rng=np.random.default_rng(61756);x=np.r_[rng.normal([-2,0],.6,(30,2)),rng.normal([2,0],.8,(30,2))]
        alpha=[(np.array([-1.,0.]),np.eye(2)),(np.array([1.,0.]),np.eye(2))]
        with patch('fmvmm.numerics.constrained_mstep._pack',wraps=_pack) as encoded:
            model=fmvmm(2,list_of_dist=['mvn'],candidate_combinations=[('mvn','mvn')],
                candidate_initializations={('mvn','mvn'):([.5,.5],alpha)},numerical_policy={},max_iter=3,verbose=False)
            model.fit(x)
        self.assertFalse(model.candidate_failures)
        self.assertEqual(len(encoded.call_args_list),2)
        self.assertTrue(all(call.args[3] is True for call in encoded.call_args_list))
        for item in model.em_diagnostics[0]['final_component_stationarity']:
            self.assertEqual(item['coordinate_recovery']['source'],'retained_optimizer_coordinates')

    def test_retained_coordinates_cannot_change_native_likelihood(self):
        x=np.zeros((5,2));policy=EMPolicy();alpha,info=prepare_component('mvn',(np.zeros(2),np.eye(2)),policy)
        coordinates=np.array(info['optimizer_coordinates_vector']);coordinates[0]=1.
        with self.assertRaises(FloatingPointError):component_stationarity('mvn',x,np.ones(5),alpha,policy,coordinates=coordinates)


if __name__=='__main__':unittest.main()
