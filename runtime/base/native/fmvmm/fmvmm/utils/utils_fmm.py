import numpy as np
import pandas as pd
import itertools
import math
from sklearn.cluster import KMeans
from sklearn import mixture
from fmvmm.mixtures.mixmgh import approx_hessian_scipy
    
    


def fmm_loglikelihood(pi_temp,alpha_temp,data_lol,dist_comb):
    n = len(data_lol)
    k = len(alpha_temp)
    log_likelihood_values_temp = []
    for c in range(n):
        try:
            log_likelihood_old_temp = math.log(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        except:
            log_likelihood_old_temp = math.log1p(np.nansum(
                [pi_temp[f]*dist_comb[f].pdf(np.reshape(np.array(data_lol[c]),(1,len(data_lol[c]))), *alpha_temp[f]) for f in range(k)]))
        log_likelihood_values_temp.append(log_likelihood_old_temp)
    log_likelihood_old = np.sum(log_likelihood_values_temp)

    return log_likelihood_old


def fmm_responsibilities(pi_temp, alpha_temp, data_lol,dist_comb):
    n = len(data_lol)
    k = len(alpha_temp)
    gamma_temp = []
    for i in range(n):
        gamma_numer = []
        for j in range(k):
            temp_gamma_numer = (
                pi_temp[j]*dist_comb[j].pdf(np.reshape(np.array(data_lol[i]),(1,len(data_lol[i]))), *alpha_temp[j]))
            gamma_numer.append(temp_gamma_numer)
        gamma_row = gamma_numer / np.nansum(np.asanyarray(gamma_numer))
        gamma_temp.append(gamma_row)
    gamma_temp_ar = np.array(gamma_temp, dtype=np.float64)
    gamma_matrix = []
    for v in gamma_temp:
        gm_temp = v.tolist()
        gamma_matrix.append(gm_temp)
    return gamma_temp_ar, gamma_matrix


def fmm_pi_estimate(gamma_temp_ar,data_cwise):
    n = gamma_temp_ar.shape[0]
    k = gamma_temp_ar.shape[1]
    pi_new = []
    # nk = []
    # for g in range(k):
    #     nk_temp = np.nansum([gamma_temp_ar[w, g] for w in range(n)])
    #     pi_temp = nk_temp/n
    #     pi_new.append(pi_temp)
    #     nk.append(nk_temp)
    pi_new = [len(data_cwise[m])/n for m in range(k)]
    return pi_new


def _fit_component_with_kwargs(dist, x, fit_kwargs_by_module=None):
    fit_kwargs_by_module = {} if fit_kwargs_by_module is None else fit_kwargs_by_module
    kwargs = fit_kwargs_by_module.get(dist, {})
    return dist.fit(x, **kwargs)


def fmm_kmeans_init(data, k,dist_comb, fit_kwargs_by_module=None):
    n = len(data)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)
    data_lol = data.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    alpha_not=[
        _fit_component_with_kwargs(dist_comb[j], np.array(data_cwise[j]), fit_kwargs_by_module)
        for j in range(k)
    ]
    return pi_not, alpha_not

def fmm_gmm_init(data, k,dist_comb, fit_kwargs_by_module=None):
    n = len(data)
    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(data)
    data_lol = data.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[clf.predict(data)[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    alpha_not=[
        _fit_component_with_kwargs(dist_comb[j], np.array(data_cwise[j]), fit_kwargs_by_module)
        for j in range(k)
    ]
    return pi_not, alpha_not



def fmm_estimate_alphas(data_cwise, alpha_not, dist_comb_not, fit_kwargs_by_module=None):
    alpha_new = []
    for t in range(len(data_cwise)):
        if np.array(data_cwise[t]).ndim == 1:
            alpha_new_temp = alpha_not[t]
        else:
            alpha_new_temp = _fit_component_with_kwargs(
                dist_comb_not[t],
                np.array(np.array(data_cwise[t])),
                fit_kwargs_by_module,
            )

        alpha_new.append(alpha_new_temp)
    return alpha_new

def fmm_aic(alpha_new,log_likelihood_new, dist_comb):
    k = len(dist_comb)
    comp_params = np.sum([dist_comb[a].total_params(*alpha_new[a]) for a in range(len(dist_comb))])
    total_params = k-1 +  comp_params
    
    return -2 * log_likelihood_new + 2 * total_params
    



def fmm_bic(alpha_new,log_likelihood_new, dist_comb,n):
    k = len(dist_comb)
    comp_params = np.sum([dist_comb[a].total_params(*alpha_new[a]) for a in range(len(dist_comb))])
    total_params = k-1 +  comp_params

    return -2 * log_likelihood_new + total_params * np.log(n)

def fmm_icl(alpha_new,log_likelihood_new, dist_comb,n, gamma_temp_ar):
    
    bic_value = fmm_bic(alpha_new,log_likelihood_new, dist_comb,n)
    entropy_term = np.sum(gamma_temp_ar * np.log(np.clip(gamma_temp_ar, 1e-10, 1)))
    
    return bic_value + entropy_term
    
def full_neg_loglike(theta, data, param_shapes, lg_fun):
    n = data.shape[0]
    alpha_list = reshape_params(theta, param_shapes)
    ll = lg_fun(data, *alpha_list)
    total_ll = - np.sum(ll)
    
    return total_ll

def compute_info_scipy_fmvmm(logpdf_func, params_init, X, epsilon=1e-6):
    """
    Use SciPy to approximate the Hessian of the negative log-likelihood 
    for the GH mixture, then invert to get the empirical info matrix.

    Returns: I_e, Cov, SE
    """
    # Flatten final parameters
    n, p = X.shape
    param_shapes = [p.shape for p in params_init]
    theta_hat = flatten_params(params_init)

    d = len(theta_hat)

    # define partial function
    def nll_wrapper(th):
        return full_neg_loglike(th, X, param_shapes, logpdf_func)

    # compute Hessian
    H = approx_hessian_scipy(nll_wrapper, theta_hat, epsilon)

    return H

def flatten_params(params):
    """
    Flatten parameters into a single vector.
    
    Parameters:
        params (tuple): The parameters.
        
    Returns:
        flat_params (ndarray): The flattened parameters.
    """
    return np.array(np.concatenate([p.flatten() for p in params]),dtype=float)

def reshape_params(flat_params, param_shapes):
    """
    Reshape a flattened parameter vector into original shapes.
    
    Parameters:
        flat_params (ndarray): The flattened parameters.
        param_shapes (list): The shapes of the original parameters.
        
    Returns:
        params (tuple): The reshaped parameters.
    """
    params = []
    start = 0
    for shape in param_shapes:
        end = start + np.prod(shape)
        params.append(np.reshape(flat_params[start:end], shape))
        start = end
    return tuple(params)
