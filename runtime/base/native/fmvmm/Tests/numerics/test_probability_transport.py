import json
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal
from fmvmm.numerics.robust_probability import (weighted_log_likelihood,
    evaluate_weighted_log_likelihood, skew_normal_logpdf,
    skew_contaminated_normal_logpdf)
from fmvmm.numerics import lossless_json

class ProbabilityTests(unittest.TestCase):
    def test_weighted_zero_terms(self):
        self.assertEqual(weighted_log_likelihood([-2,-np.inf,np.nan],[1,0,0]),-2)
        self.assertEqual(weighted_log_likelihood([np.nan,-np.inf],[0,0]),0)
        self.assertEqual(weighted_log_likelihood([-2,-np.inf],[1,1]),-np.inf)
        with self.assertRaises(FloatingPointError):weighted_log_likelihood([np.nan],[1])
        with self.assertRaises(ValueError):weighted_log_likelihood([0],[-1])
        def restricted_logpdf(x):
            if np.any(x!=0):raise AssertionError('Zero-weight data evaluated')
            return np.full(len(x),3.)
        self.assertEqual(evaluate_weighted_log_likelihood(restricted_logpdf,[[0],[10]],[2,0]),6.)

    def test_msnc_moderate_identity(self):
        rng=np.random.default_rng(41)
        for p in [2,5]:
            a=rng.normal(size=(p,p));sigma=a@a.T+np.eye(p)
            x=rng.normal(size=(20,p));loc=rng.normal(size=p);lmbda=rng.normal(size=p)
            ev,vec=np.linalg.eigh(sigma);symmetric_inverse=(vec/np.sqrt(ev))@vec.T
            skew=(x-loc)@symmetric_inverse@lmbda
            for mixing,inv_inflation in [(0.,.1),(.2,.2),(.5,1.),(1.,.1)]:
                dirty=multivariate_normal.logpdf(x,loc,sigma/inv_inflation)+log_ndtr(np.sqrt(inv_inflation)*skew)
                clean=multivariate_normal.logpdf(x,loc,sigma)+log_ndtr(skew)
                with np.errstate(divide='ignore'):
                    expected=np.log(2)+np.logaddexp(np.log(mixing)+dirty,np.log1p(-mixing)+clean)
                actual=skew_contaminated_normal_logpdf(x,loc,sigma,lmbda,[mixing,inv_inflation])
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(skew_contaminated_normal_logpdf(x,loc,sigma,lmbda,[.4,1.]),
                                       skew_normal_logpdf(x,loc,sigma,lmbda),atol=1e-12,rtol=1e-12)

    def test_msnc_collapsed_pilot_underflow(self):
        p=5;x=np.stack([np.zeros(p),np.full(p,10.),np.full(p,-10.)])
        sigma=1e-6*np.eye(p);lmbda=np.array([2.,0,0,0,0])
        values=skew_contaminated_normal_logpdf(x,np.zeros(p),sigma,lmbda,[.1,.2])
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(values[1],-1e6)
        self.assertTrue(np.exp(values[1])==0.)
        self.assertEqual(weighted_log_likelihood(values,[1,0,0]),values[0])

    def test_round_trip(self):
        original={'df':np.inf,'missing':None,'scores':np.array([1.,-np.inf,np.nan]),'count':np.int64(3)}
        text=lossless_json.dumps(original,sort_keys=True)
        json.loads(text,parse_constant=lambda token: self.fail('Invalid JSON constant '+token))
        result=lossless_json.loads(text)
        self.assertEqual(result['df'],np.inf)
        self.assertEqual(result['missing'],None)
        self.assertEqual(result['scores'][:2],[1.,-np.inf])
        self.assertTrue(np.isnan(result['scores'][2]))
        for text in ['{"a": Infinity}','{"a": NaN}']:
            with self.assertRaises(ValueError):lossless_json.loads(text)
        with self.assertRaises(ValueError):lossless_json.dumps({lossless_json.TAG:'anything'})

if __name__=='__main__':unittest.main()
