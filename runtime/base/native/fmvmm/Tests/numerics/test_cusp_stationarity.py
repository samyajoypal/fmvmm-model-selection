"""Independent conditional-cusp checks and native-likelihood acceptance tests."""
import unittest
from dataclasses import replace
import numpy as np
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import prepare_component,_pack,_unpack,component_stationarity,fit_component
from fmvmm.numerics.cusp_stationarity import (location_classification,anchor_continuation,
    norm_cusp_location_diagnostic,leave_failed_norm_anchor)
from fmvmm.mixtures.FMVMM import dist_map


class TestCuspStationarity(unittest.TestCase):
    def setup_component(self,name='mvvg',order=.3,p=2,mu=None):
        mu=np.zeros(p) if mu is None else mu
        alpha=(p/2+order,2*(p/2+order),mu,np.eye(p),np.zeros(p)) if name=='mvvg' else (mu,np.eye(p),np.zeros(p))
        policy=EMPolicy();alpha,_=prepare_component(name,alpha,policy)
        u,_=_pack(name,alpha,policy,False)
        return alpha,u,policy

    def native_objective(self,name,x,w,policy):
        def value(u):
            alpha,_=_unpack(u,name,x.shape[1],policy)
            return -float(w@dist_map[name].logpdf(x,*alpha))/w.sum()
        return value

    def test_vg_strict_cusp_duplicates_and_exact_equality(self):
        alpha,u,policy=self.setup_component()
        x=np.array([[0.,0.],[0.,0.],[1.,.2],[-.3,.8]])
        w=np.array([1.,2.,.8,.6])
        state=location_classification(x,w,u,'mvvg',policy)
        self.assertTrue(state['strict_conditional_location_local_maximum'])
        self.assertEqual(state['center_mass'],3.)
        state2=component_stationarity('mvvg',x,w,alpha,policy)
        self.assertFalse(state2['ordinary_location_gradient_exists'])
        objective=self.native_objective('mvvg',x,w,policy)
        for step in (1e-5,1e-7,1e-9):
            for direction in (np.array([1.,0.]),np.array([0.,1.]),np.ones(2)):
                for sign in (-1,1):
                    v=u.copy();v[:2]+=sign*step*direction
                    self.assertGreater(objective(v),objective(u))
        near=u.copy();near[0]=1e-12
        self.assertFalse(location_classification(x,w,near,'mvvg',policy)['location_nonsmooth'])

    def test_no_equality_band_around_half(self):
        x=np.array([[0.,0.],[1.,0.],[-1.,0.]])
        for order in (.5-1e-7,.5+1e-7):
            alpha,u,policy=self.setup_component(order=order)
            state=component_stationarity('mvvg',x,np.ones(3),alpha,policy)
            self.assertEqual(state['ordinary_location_gradient_exists'],order>.5)

    def test_new_memberships_change_norm_subgradient_condition(self):
        alpha,u,policy=self.setup_component('mvsl')
        x=np.array([[0.,0.],[1.,0.],[-1.,0.]])
        old=component_stationarity('mvsl',x,[1.,1.,1.],alpha,policy)
        new=component_stationarity('mvsl',x,[.1,2.,.1],alpha,policy)
        self.assertTrue(old['location_condition_satisfied'])
        self.assertFalse(new['location_condition_satisfied'])
        self.assertGreater(new['location_residual'],policy.gradient_tolerance)
        w=np.array([.1,2.,.1]);objective=self.native_objective('mvsl',x,w,policy)
        escaped,diag=leave_failed_norm_anchor(x,w,u,'mvsl',policy,objective)
        self.assertTrue(diag['accepted'])
        self.assertGreater(escaped[0],u[0]);self.assertLess(objective(escaped),objective(u))

    def test_box_outward_normal_cone_sign(self):
        mu=np.array([1.]);sigma=np.eye(1);gamma=np.zeros(1)
        for weights in ([1.,1.],[.2,3.]):
            unconstrained=norm_cusp_location_diagnostic([[2.],[3.]],weights,mu,sigma,gamma)
            constrained=norm_cusp_location_diagnostic([[2.],[3.]],weights,mu,sigma,gamma,location_bound=1.)
            self.assertFalse(unconstrained['condition_satisfied'])
            self.assertTrue(constrained['condition_satisfied'])
            self.assertGreater(constrained['closest_box_normal'][0],0.)

    def test_lower_likelihood_and_outside_domain_anchors_are_rejected(self):
        alpha,u,policy=self.setup_component(order=2.)
        policy=replace(policy,cusp_proposal_radius=1e-3)
        x=np.array([[1e-4,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        w=np.array([1e-6,1.,1.,1.,1.]);objective=self.native_objective('mvvg',x,w,policy)
        selected,attempts=anchor_continuation(x,w,u,'mvvg',policy,10,objective)
        self.assertEqual(attempts[0]['status'],'rejected_lower_native_likelihood')
        np.testing.assert_array_equal(selected,u)
        alpha,u,policy=self.setup_component(mu=np.array([30.,0.]))
        x=np.array([[30.+1e-5,0.],[28.,1.],[29.,-1.]])
        w=np.ones(3);objective=self.native_objective('mvvg',x,w,policy)
        selected,attempts=anchor_continuation(x,w,u,'mvvg',policy,10,objective)
        self.assertEqual(attempts[0]['status'],'outside_location_domain')
        np.testing.assert_array_equal(selected,u)

    def test_zero_weight_anchor_does_not_create_cusp(self):
        alpha,u,policy=self.setup_component()
        x=np.array([[0.,0.],[1.,0.],[-1.,0.]])
        state=component_stationarity('mvvg',x,[0.,1.,1.],alpha,policy)
        self.assertFalse(state['location_nonsmooth'])

    def test_capped_component_update_remains_honest_and_monotone(self):
        alpha,u,policy=self.setup_component()
        policy=replace(policy,cusp_max_rounds=1)
        x=np.array([[0.,0.],[.7,.2],[-.5,-.4],[.1,.8]])
        out,diag=fit_component('mvvg',x,np.ones(len(x)),alpha,policy,maxiter=1)
        self.assertTrue(diag['hit_cap']);self.assertFalse(diag['gradient_acceptable'])
        self.assertGreaterEqual(diag['weighted_loglik'],diag['weighted_loglik_before'])
        self.assertFalse(diag['global_optimum_certified'])
        self.assertEqual(diag['stationarity']['ordinary_location_gradient_exists'],False)

    def test_tiny_skew_gst_spatial_score_is_finite(self):
        from fmvmm.numerics.weighted_gh_score import values_gradient
        for p in (2,3,4,5):
            alpha=(-51.,100.,np.zeros(p),np.eye(p),np.full(p,1e-150))
            native,_=prepare_component('mgst',alpha,EMPolicy())
            u,_=_pack('mgst',native,EMPolicy(),False)
            x=np.r_[np.zeros((1,p)),np.eye(p),-np.eye(p)]
            ld,g=values_gradient(x,u,'mgst',EMPolicy(),np.ones(len(x)))
            self.assertTrue(np.isfinite(ld).all());self.assertTrue(np.isfinite(g).all())
            np.testing.assert_allclose(g[:p],0,atol=1e-8)


if __name__=='__main__':unittest.main()
