"""Public soft-EM integration across all families, p=2..5 and K=3..5.

These deliberately bounded tests establish finite, domain-consistent monotone
updates and honest stopping diagnostics; they do not assert global optimization.
"""
import unittest
import numpy as np
from fmvmm.mixtures.FMVMM import fmvmm
from fmvmm.numerics.constrained_mstep import component_stationarity

CASES=[(2,('mghp','mvvg','mvsl')),(3,('mvhb','mnig','mgst','mghp')),
       (4,('mvn','mvt','mvsn','mvst','msnc')),(5,('msl','mssl','mvsl'))]


def scenario(p,names,seed=66153,n_per_component=18):
    rng=np.random.default_rng(seed+p);blocks=[];alpha=[]
    for j,name in enumerate(names):
        mu=np.zeros(p);mu[0]=5*(j-(len(names)-1)/2)
        block=rng.normal(mu,.65,size=(n_per_component,p));blocks.append(block)
        sigma=np.eye(p)*.6;gamma=np.linspace(-.1,.1,p)
        if name=='mghp':a=(1.,2.,3.,mu,sigma,gamma)
        elif name in ('mvhb','mnig'):a=(2.,3.,mu,sigma,gamma)
        elif name=='mgst':a=(-3.,4.,mu,sigma,gamma)
        elif name=='mvvg':a=(p/2+.3,2*(p/2+.3),mu,sigma,gamma)
        elif name=='mvsl':a=(mu,sigma/(p+1),gamma/(p+1))
        else:
            a=(mu,sigma)
            if name in ('mvsn','mvst','msnc','mssl'):a+=(gamma,)
            if name in ('mvt','mvst','msl','mssl'):a+=(np.array([5.]),)
            if name=='msnc':a+=(np.array([.08,.3]),)
        alpha.append(a)
    return np.concatenate(blocks),alpha


class TestAllFamilyEM(unittest.TestCase):
    def test_public_path_all_families_dimensions_and_components(self):
        for p,names in CASES:
            with self.subTest(p=p,names=names):
                x,alpha=scenario(p,names)
                model=fmvmm(len(names),list_of_dist=list(names),candidate_combinations=[names],
                    candidate_initializations={names:(np.ones(len(names))/len(names),alpha)},
                    numerical_policy={'component_maxiter':20},max_iter=2,tol=1e-8,verbose=False)
                model.fit(x)
                self.assertFalse(model.candidate_failures,model.candidate_failures)
                self.assertEqual(len(model.list_bic),1)
                self.assertTrue(np.isfinite(model.list_bic[0]))
                self.assertGreaterEqual(np.min(np.diff(model.list_all_log_likelihood[0])),-1e-8)
                diag=model.em_diagnostics[0]
                self.assertEqual(diag['outer_loop'],'BaseMixture._fit')
                self.assertFalse(diag['global_optimum_certified'])
                if diag['converged']:self.assertTrue(diag['final_stationarity_acceptable'])
                for iteration in diag['component_iterations']:
                    for component in iteration:
                        self.assertTrue(component['finite']);self.assertTrue(component['accepted'])
                        self.assertGreaterEqual(component['weighted_loglik'],component['weighted_loglik_before']-1e-8)
                        self.assertIn('stationarity_kind',component['stationarity'])


if __name__=='__main__':unittest.main()
