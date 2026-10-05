"""Independent Decimal tail integration around the floating-point CDF boundary."""
import unittest
from decimal import Decimal as D,localcontext
import numpy as np
from scipy.special import stdtr
from fmvmm.numerics.robust_skew_t import student_logcdf
from test_robust_mvt import decimal_loggamma,PI


def decimal_student_logtail(a,df):
    """Independent additive-tail substitution, Decimal Simpson/Richardson.

    Production uses a logarithmic substitution and double-precision adaptive
    quadrature. Here t=a+s/rate, rate=(df+1)*a/(df+a^2), and calculations use
    60-digit arithmetic on a fixed interval. Three Richardson columns are
    evaluated on successively halved grids. The final comparison checks that
    the independent extrapolation has stabilized.
    """
    with localcontext() as context:
        context.prec=60
        a=D(str(a));df=D(str(df));rate=(df+1)*a/(df+a*a)
        log_pdf=(decimal_loggamma((df+1)/2)-decimal_loggamma(df/2)
                 -(df*PI).ln()/2-(df+1)/2*(1+a*a/df).ln())
        tables=[]
        for n in [512,1024,2048,4096]:
            h=D(60)/n
            def integrand(j):
                s=h*j;y=s/rate
                change=(2*a*y+y*y)/(df+a*a)
                return (-(df+1)/2*(1+change).ln()).exp()
            total=integrand(0)+integrand(n)
            for j in range(1,n):total+=(4 if j%2 else 2)*integrand(j)
            row=[h*total/3]
            if tables:
                for order in range(1,min(3,len(tables)+1)):
                    factor=D(2)**(2*order+2)
                    row.append((factor*row[-1]-tables[-1][order-1])/(factor-1))
            tables.append(row)
        if abs(tables[-1][-1]-tables[-2][-1])>D('1e-12'):
            raise AssertionError('Independent Decimal quadrature did not stabilize')
        return float(log_pdf-rate.ln()+tables[-1][-1].ln())


class SubnormalCDFTests(unittest.TestCase):
    def test_decimal_subnormal_transition(self):
        # The first case reproduced a 4.26e-4 error from logging a nonzero
        # quantized CDF. Other points straddle the normal/subnormal/zero boundary.
        for a,df in [(38.31816181,1e16),(37.5,1e16),(38.8,1e16)]:
            expected=decimal_student_logtail(a,df)
            actual=student_logcdf(-a,df)
            self.assertLess(abs(actual-expected),2e-10,(a,df,actual,expected))
        # Smooth finite-df derivative around a tail point with subnormal CDF.
        a=38.31816181;df=1e16;h=1e-4
        d1=(student_logcdf(-a+h,df)-student_logcdf(-a-h,df))/(2*h)
        d2=(student_logcdf(-a+h/2,df)-student_logcdf(-a-h/2,df))/h
        self.assertLess(abs(d1-d2),1e-7)

if __name__=='__main__':unittest.main()
