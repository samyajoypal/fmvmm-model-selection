import numpy as np
# import cupy as np
import pandas as pd
from scipy.special import logsumexp
from abc import ABCMeta, abstractmethod
from fmvmm.utils import utils_mixture


def _blend_parameter(old, new, fraction):
    """Convexly damp nested numeric parameter structures."""
    if isinstance(old, tuple):
        return tuple(
            _blend_parameter(old_item, new_item, fraction)
            for old_item, new_item in zip(old, new)
        )
    if isinstance(old, list):
        return [
            _blend_parameter(old_item, new_item, fraction)
            for old_item, new_item in zip(old, new)
        ]
    old_array = np.asarray(old, dtype=float)
    new_array = np.asarray(new, dtype=float)
    blended = old_array + fraction * (new_array - old_array)
    if np.asarray(old).shape == ():
        return float(blended)
    return blended
# import numpy

class BaseMixture(metaclass=ABCMeta):

    def __init__(self, n_clusters, EM_type, mixture_type, tol, print_log_likelihood, max_iter, verbose):
        self.n_clusters = n_clusters
        self.EM_type = EM_type
        self.mixture_type = mixture_type
        self.tol = tol
        self.print_log_likelihood = print_log_likelihood
        self.max_iter = max_iter
        self.verbose = verbose
        self.fitted=False

    def _process_data(self,X):
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray) and len(X.shape) == 2:
            return X
        elif isinstance(X, list):
            if all(isinstance(row, list) for row in X):
                return np.array(X)
            else:
                raise ValueError("Input should be a list of lists for conversion to 2D NumPy array.")
        else:
            raise ValueError("Input should be a Pandas DataFrame, a 2D NumPy array, or a list of lists.")

    # @abstractmethod
    # def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
    #     pass

    # @abstractmethod
    # def _estimate_weighted_log_prob_nonidentical(self, X, alpha, pi, dist_comb):
    #     pass

    def _estimate_log_prob_resp_identical(self, X, alpha, pi):
        weighted_log_prob = self._estimate_weighted_log_prob_identical(
            X, alpha, pi)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_log_prob_resp_nonidentical(self, X, alpha, pi, dist_comb):
        weighted_log_prob = self._estimate_weighted_log_prob_nonidentical(
            X, alpha, pi, dist_comb)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _e_step(self, X, alpha, pi, dist_comb=None):
        if self.mixture_type == "identical":
            log_prob_norm, log_resp = self._estimate_log_prob_resp_identical(
                X, alpha, pi)
        elif self.mixture_type == "nonidentical":
            log_prob_norm, log_resp = self._estimate_log_prob_resp_nonidentical(
                X, alpha, pi, dist_comb)
        return np.sum(log_prob_norm), log_resp
    
    
    def _fit(self,X,pi_temp,alpha_temp,estimate_alphas_function,dist_comb=None,**kwargs):
        X=self._process_data(X)
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.log_likelihoods=[]
        post_m_step = kwargs.pop("post_m_step", None)
        reject_nonmonotone_proposal = kwargs.pop("reject_nonmonotone_proposal", False)
        nonmonotone_tolerance = float(kwargs.pop("nonmonotone_tolerance", 1e-8))
        self._nonmonotone_proposal_rejections = 0
        post_iteration = kwargs.pop("post_iteration", None)
        stop=False
        while stop!=True:
            log_likelihood_old, log_gamma_temp = self._e_step(
                X, alpha_temp, pi_temp,dist_comb)
            if len(self.log_likelihoods)==0:
                self.log_likelihoods.append(log_likelihood_old)
            gamma_temp_ar=np.exp(log_gamma_temp)
            self.alpha_temp = alpha_temp
            self.pi_temp = pi_temp
            pi_new,alpha_new=self._m_step(gamma_temp_ar, X, estimate_alphas_function,dist_comb, **kwargs)
            if post_m_step is not None:
                pi_new, alpha_new = post_m_step(pi_new, alpha_new)
            log_likelihood_new, log_gamma_new = self._e_step(
                X, alpha_new, pi_new,dist_comb)
            if self.EM_type == "Soft" and log_likelihood_new < log_likelihood_old - nonmonotone_tolerance:
                if reject_nonmonotone_proposal:
                    # Native-parameter interpolation need not preserve a declared
                    # parameter domain (for example mean-one GH parameters), and
                    # an infinite df boundary cannot be linearly interpolated.
                    # Keep the valid incumbent and report the rejected proposal.
                    self._nonmonotone_proposal_rejections += 1
                    pi_new, alpha_new = pi_temp, alpha_temp
                    log_likelihood_new, log_gamma_new = log_likelihood_old, log_gamma_temp
                else:
                    accepted = False
                    for fraction in (0.5, 0.25, 0.125, 0.0625, 0.03125):
                        pi_try = _blend_parameter(pi_temp, pi_new, fraction)
                        pi_try = np.asarray(pi_try, dtype=float)
                        pi_try = pi_try / pi_try.sum()
                        alpha_try = _blend_parameter(alpha_temp, alpha_new, fraction)
                        ll_try, log_gamma_try = self._e_step(
                            X, alpha_try, pi_try, dist_comb
                        )
                        if ll_try >= log_likelihood_old - 1e-8:
                            pi_new, alpha_new = pi_try, alpha_try
                            log_likelihood_new, log_gamma_new = ll_try, log_gamma_try
                            accepted = True
                            break
                    if not accepted:
                        pi_new, alpha_new = pi_temp, alpha_temp
                        log_likelihood_new, log_gamma_new = log_likelihood_old, log_gamma_temp
            self.log_likelihoods.append(log_likelihood_new)
            if post_iteration is not None:
                post_iteration(pi_new, alpha_new, log_gamma_new)
            # print("pi_new", pi_new)
            # print("alpha_new", alpha_new)
            stop=self._stopping_criteria(self.log_likelihoods,self.tol,self.max_iter)
            log_like_diff = abs(log_likelihood_new-log_likelihood_old)/abs(log_likelihood_old)
            alpha_temp = alpha_new
            pi_temp = pi_new
            if self.print_log_likelihood==True:
                print("Loglikelihood:",log_likelihood_new, "Relative Difference:", log_like_diff )
            
        return pi_new,alpha_new, log_likelihood_new,log_gamma_new
    
    def _predict(self,X):
        X=self._process_data(X)
        if self.mixture_type == "identical":
            return self._estimate_weighted_log_prob_identical(X,self.alpha_new, self.pi_new).argmax(axis=1)
        elif self.mixture_type=="nonidentical":
            return self._estimate_weighted_log_prob_nonidentical(X,self.alpha_new, self.pi_new,self.dist_comb).argmax(axis=1)
            
            
    
    def _m_step(self,gamma_matrix,X,estimate_alphas_function,dist_comb,**kwargs):
        if self.EM_type=="Hard":
            cluster,data_cwise=utils_mixture.mixture_clusters(gamma_matrix, X)
            pi_new = np.bincount(cluster, minlength=gamma_matrix.shape[1]) / len(cluster)
            alpha_new=estimate_alphas_function(data_cwise,self.alpha_not,dist_comb,**kwargs)
        elif self.EM_type=="Soft":
            nj=gamma_matrix.sum(axis=0) + 10 * np.finfo(gamma_matrix.dtype).eps
            pi_new = nj / nj.sum()
            alpha_new=estimate_alphas_function(X,gamma_matrix,self.alpha_temp,**kwargs)

        return pi_new, alpha_new

    def _stopping_criteria(self, log_likelihoods, tol, n_iter_max):
        if len(log_likelihoods) == n_iter_max:
            return True
        elif abs(log_likelihoods[-1]-log_likelihoods[-2])/abs(log_likelihoods[-2]) <= tol:
            return True
        else:
            return False
    
    def aic(self):
        """
        Compute Akaike Information Criterion (AIC).
        
        Parameters:
            log_likelihood (float): Maximum log-likelihood value from model fitting.
            num_params (int): Total number of estimated parameters in the model.
    
        Returns:
            float: AIC value.
        """
        if self.mixture_type == "nonidentical":
            return self.list_aic
        else:
            return -2 * self.log_likelihood_new + 2 * self.total_parameters


    def bic(self):
        """
        Compute Bayesian Information Criterion (BIC).
        
        Parameters:
            log_likelihood (float): Maximum log-likelihood value from model fitting.
            num_params (int): Total number of estimated parameters in the model.
            num_samples (int): Number of observations (data points).
    
        Returns:
            float: BIC value.
        """
        if self.mixture_type == "nonidentical":
            return self.list_bic
        else:
            return -2 * self.log_likelihood_new + self.total_parameters * np.log(self.n)
    
    
    def icl(self):
        """
        Compute Integrated Completed Likelihood (ICL).
        
        Parameters:
            log_likelihood (float): Maximum log-likelihood value from model fitting.
            num_params (int): Total number of estimated parameters in the model.
            num_samples (int): Number of observations (data points).
            responsibilities (numpy.ndarray): Responsibility matrix of shape (n_samples, K),
                                              where K is the number of clusters.
    
        Returns:
            float: ICL value.
        """
        if self.mixture_type == "nonidentical":
            return self.list_icl
        else:
            bic_value = self.bic()

            # Compute entropy term: sum over all i and k of r_ik log(r_ik)
            entropy_term = np.sum(self.gamma_temp_ar * np.log(np.clip(self.gamma_temp_ar, 1e-10, 1)))

            return bic_value + entropy_term
