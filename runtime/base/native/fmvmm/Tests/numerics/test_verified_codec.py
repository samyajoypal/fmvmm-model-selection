import json,unittest
from pathlib import Path
import numpy as np
from fmvmm.mixtures.FMVMM import dist_map
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.gh_policy import gh_tail_bounds,unpack_gh_tail,mean_gig
from fmvmm.numerics.constrained_mstep import _unpack
from fmvmm.numerics.verified_codec import encode_gh_tail,encode_native_scale,verify_native_likelihood

FIXTURES=json.loads((Path(__file__).parent/'fixtures/coordinate_roundtrips.json').read_text())

class VerifiedCodecTests(unittest.TestCase):
    def test_exact_gh_tail_failure_and_density_preservation(self):
        f=FIXTURES['gh'];alpha=f['alpha'];p=f['p'];policy=EMPolicy()
        tail,record=encode_gh_tail('mghp',alpha,p,policy)
        self.assertTrue(record['encoding_reconciled']);np.testing.assert_array_equal(tail,f['original_u'])
        candidate=(*unpack_gh_tail('mghp',tail,p,policy),*alpha[-3:])
        x=np.array(alpha[-3])+np.random.default_rng(911).normal(size=(20,p))
        checked=verify_native_likelihood(dist_map['mghp'],x,np.ones(len(x)),alpha,candidate)
        self.assertLess(checked['max_absolute_log_density_change'],1e-10)

    def test_tail_boundary_grid_and_material_rejection(self):
        policy=EMPolicy()
        for p in (2,3,4,5):
            for name in ('mghp','mvhb','mnig','mgst','mvvg'):
                bounds=np.array(gh_tail_bounds(name,p,policy))
                for corner in np.ndindex(*([2]*len(bounds))):
                    u=bounds[np.arange(len(bounds)),corner]
                    alpha=(*unpack_gh_tail(name,u,p,policy),np.zeros(p),np.eye(p),np.zeros(p))
                    encoded,record=encode_gh_tail(name,alpha,p,policy)
                    self.assertTrue(np.all(encoded>=bounds[:,0]) and np.all(encoded<=bounds[:,1]))
                    np.testing.assert_allclose(encoded,u,rtol=0,atol=3e-14)
        lam=.7;abar=policy.gh_alpha_bar_min*np.exp(-1e-8);ratio=mean_gig(lam,abar,abar)
        invalid=(lam,abar/ratio,abar*ratio,np.zeros(2),np.eye(2),np.zeros(2))
        with self.assertRaises(ValueError):encode_gh_tail('mghp',invalid,2,policy)

    def test_exact_nig_covariance_failure(self):
        f=FIXTURES['nig'];alpha=f['alpha'];p=f['p'];policy=EMPolicy()
        coords,L,sigma,record=encode_native_scale(alpha[-2],policy)
        self.assertTrue(record['encoding_reconciled'])
        self.assertGreaterEqual(L[-1,-1],policy.cholesky_min)
        self.assertLessEqual(record['backward_error_inf'],record['backward_error_budget'])
        candidate=(*alpha[:-2],sigma,alpha[-1])
        x=np.array(alpha[-3])+np.array(alpha[-1])+np.random.default_rng(912).normal(size=(30,p))
        result=verify_native_likelihood(dist_map['mnig'],x,np.ones(len(x)),alpha,candidate)
        self.assertLess(result['max_absolute_log_density_change'],2e-9)
        # Retaining the original legal optimizer coordinates avoids re-encoding.
        exact,_=_unpack(np.array(f['original_u']),'mnig',p,policy)
        np.testing.assert_array_equal(exact[-2],alpha[-2])

    def test_ordinary_scales_and_near_floor_factors_p2_through_p5(self):
        policy=EMPolicy();rng=np.random.default_rng(914)
        for p in (2,3,4,5):
            for near_floor in (False,True):
                L=np.eye(p)*.8;L[np.tril_indices(p,-1)]=rng.uniform(-.2,.2,p*(p-1)//2)
                if near_floor:L[-1,-1]=policy.cholesky_min
                sigma=L@L.T+policy.covariance_floor*np.eye(p)
                _,encoded,reconstructed,record=encode_native_scale(sigma,policy)
                self.assertLessEqual(record['backward_error_inf'],record['backward_error_budget'])
                self.assertTrue(np.all(np.diag(encoded)>=policy.cholesky_min))
                np.testing.assert_allclose(reconstructed,sigma,rtol=1e-13,atol=1e-15)

    def test_substantively_invalid_covariances_are_not_repaired(self):
        policy=EMPolicy()
        for factor in (.5*policy.cholesky_min,policy.cholesky_max+1e-4):
            sigma=(policy.covariance_floor+factor**2)*np.eye(2)
            with self.assertRaises(ValueError):encode_native_scale(sigma,policy)
        with self.assertRaises(np.linalg.LinAlgError):encode_native_scale(.5*policy.covariance_floor*np.eye(2),policy)

    def test_native_likelihood_guard_is_not_an_acceptance_override(self):
        a=(np.zeros(2),np.eye(2));b=(np.ones(2),np.eye(2));x=np.zeros((2,2))
        with self.assertRaises(FloatingPointError):verify_native_likelihood(dist_map['mvn'],x,np.ones(2),a,b)

if __name__=='__main__':unittest.main()
