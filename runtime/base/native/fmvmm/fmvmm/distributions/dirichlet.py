import numpy as np
import fmvmm.utils.dirichlet as drm

def logpdf(x,alpha):
    
    return np.log(drm.pdf(np.array(alpha))(x))

def pdf(x,alpha):
    
    return drm.pdf(np.array(alpha))(x)

def loglike(x,alpha):
    
    return np.sum(logpdf(x, np.array(alpha)))

def total_params(alpha):
    p = len(alpha)
    
    return p

def rvs(alpha, size = 1):
    
    return np.random.dirichlet(alpha,size)

def fit(x, method = "meanprecision"):
    
    return drm.mle(x, method = method)
    

    