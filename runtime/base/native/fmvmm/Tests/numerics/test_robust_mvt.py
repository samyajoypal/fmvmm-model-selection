"""Independent high precision checks for the isolated Student t kernel."""
import json
import math
import unittest
from decimal import Decimal as D, localcontext
import numpy as np
from scipy.stats import multivariate_t, multivariate_normal
from fmvmm.numerics.robust_mvt import (log_normalizer, log_normalizer_eta,
    derivative_log_normalizer_eta, student_t_logpdf, validate_spd,
    CovarianceDomainError)

PI=D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
STIRLING=((1,12),(-1,360),(1,1260),(-1,1680),(1,1188),
          (-691,360360),(1,156),(-3617,122400),(43867,244188),(-174611,125400))

def decimal_loggamma(value):
    """Independent Stirling oracle, shifted to >=64; omitted error <1e-37."""
    x=D(value)
    sub=D(0)
    while x<64:
        sub+=x.ln()
        x+=1
    result=(x-D('.5'))*x.ln()-x+(2*PI).ln()/2
    for j,(num,den) in enumerate(STIRLING,1):
        result+=D(num)/D(den)/x**(2*j-1)
    return result-sub

def decimal_normalizer(df,p):
    df=D(df)
    return decimal_loggamma((df+p)/2)-decimal_loggamma(df/2)-D(p)/2*(df*PI).ln()

class RobustTTests(unittest.TestCase):
    def test_normalizer_high_precision(self):
        with localcontext() as context:
            context.prec=100
            for p in [1,2,3,5,6,9,12]:
                for df in ['1e-250','1e-12','.01','.2','1','5','63.999','64','64.001','100','1e5','1e16','1e30']:
                    expected=float(decimal_normalizer(df,p))
                    actual=log_normalizer(float(df),p)
                    self.assertLess(abs(actual-expected),1e-11, (p,df,actual,expected))
                    eta=1/float(df)
                    self.assertLess(abs(log_normalizer_eta(eta,p)-expected),1e-11,(p,df))

    def test_eta_derivative_high_precision(self):
        with localcontext() as context:
            context.prec=100
            for p in [1,2,3,5,6,9,12]:
                self.assertEqual(derivative_log_normalizer_eta(0,p),p*(p-2)/4)
                for eta in ['1e-16','1e-8','.001','.015625','.015626','.1','1','100','1e12']:
                    e=D(eta);h=e*D('1e-25')
                    expected=float((decimal_normalizer(1/(e+h),p)-decimal_normalizer(1/(e-h),p))/(2*h))
                    actual=derivative_log_normalizer_eta(float(e),p)
                    self.assertLess(abs(actual-expected),1e-10*max(1,abs(expected)),(p,eta,actual,expected))

    def test_density_high_precision(self):
        with localcontext() as context:
            context.prec=100
            for p in [1,2,5]:
                for df in ['1e-250','.01','1','10','64','1e16','1e30']:
                    x=np.stack([np.zeros(p),np.ones(p),np.arange(p)+10.])
                    answer=student_t_logpdf(x,np.zeros(p),np.eye(p),float(df))
                    for row,value in zip(x,answer):
                        q=sum(D(float(t))**2 for t in row)
                        expected=decimal_normalizer(df,p)-(D(df)+p)/2*(1+q/D(df)).ln()
                        self.assertLess(abs(value-float(expected)),1e-10*max(1,abs(float(expected))),(p,df,row))

    def test_scipy_moderate_domain(self):
        rng=np.random.default_rng(8271)
        for p in [1,2,5,10]:
            A=rng.normal(size=(p,p));shape=A@A.T+np.eye(p)
            loc=rng.normal(size=p);x=rng.normal(size=(17,p))
            for df in [.01,.2,1,5,64,1000,np.inf]:
                expected=multivariate_t.logpdf(x,loc=loc,shape=shape,df=df)
                np.testing.assert_allclose(student_t_logpdf(x,loc,shape,df),expected,rtol=1e-12,atol=2e-12)

    def test_large_df_and_exact_p2(self):
        for df in [1e-300,.001,1,1e16,1e300,np.inf]:
            self.assertEqual(log_normalizer(df,2),-math.log(2*math.pi))
        x=np.array([[0.,0.],[1.,3.],[-5.,2.]])
        for df in [1e16,1e300,np.inf]:
            np.testing.assert_allclose(student_t_logpdf(x,np.zeros(2),np.eye(2),df),
                multivariate_normal.logpdf(x,mean=np.zeros(2),cov=np.eye(2)),rtol=1e-14,atol=1e-14)
        for p in [1,2,5]:
            self.assertEqual(log_normalizer_eta(np.nextafter(0.,1.),p),-.5*p*math.log(2*math.pi))
            self.assertTrue(np.isfinite(log_normalizer(np.nextafter(0.,1.),p)))

    def test_spd_policy(self):
        for shape in [np.zeros((2,2)),[[1,1],[1,1]],[[1,2],[2,1]],np.diag([1,1e-15])]:
            with self.assertRaises(CovarianceDomainError):validate_spd(shape)
        with self.assertRaises(CovarianceDomainError):validate_spd([[1,.1],[.2,1]])
        for scale in [1e-300,1.,1e300]:
            d=validate_spd(scale*np.eye(2))
            self.assertEqual(d.relative_minimum_eigenvalue,1.)
            self.assertTrue(np.isfinite(student_t_logpdf(np.zeros(2),np.zeros(2),scale*np.eye(2),1.)))

    def test_extreme_tail(self):
        answer=student_t_logpdf([1e200,0],[0,0],np.eye(2),3.)
        self.assertTrue(np.isfinite(answer))
        self.assertLess(answer,-2000)
        self.assertTrue(np.isneginf(student_t_logpdf([1e200,0],[0,0],np.eye(2),np.inf)))

if __name__=='__main__': unittest.main()
