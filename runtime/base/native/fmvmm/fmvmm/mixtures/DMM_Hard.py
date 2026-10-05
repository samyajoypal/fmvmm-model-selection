import numpy as np
# import numpy
# import cupy as np
import pandas as pd
from scipy.stats import dirichlet
import fmvmm.utils.dirichlet as drm
from fmvmm.utils.utils_dmm import (kmeans_init, gmm_init, kmeans_init_adv, gmm_init_adv, random_init,
                             dmm_loglikelihood, dmm_responsibilities, dmm_pi_estimate)


from fmvmm.utils.utils_mixture import (mixture_clusters)
import time
from scipy.special import gammaln,logsumexp
from scipy.special import polygamma, psi, digamma
import sys
from fmvmm.mixtures._base import BaseMixture
from fmvmm.utils.utils_dmm import combined_info_and_se, mean_precision_inference


MAXINT = sys.maxsize
euler = -1 * psi(1)  # Euler-Mascheroni constant



def estimate_alphas(
    data_cwise, alpha_not, dist_comb, method, post_m_step=None
):
    alpha_new = []
    for t in range(len(data_cwise)):
        if np.array(data_cwise[t]).ndim == 1:
            alpha_new_temp = alpha_not[t]
        elif method == 'fixediteration':
            alpha_new_temp = drm.mle(
                np.array(data_cwise[t]), method="fixedpoint", maxiter=9223372036854775807)
        else:
            alpha_new_temp = drm.mle(np.array(data_cwise[t]))

        alpha_new.append(alpha_new_temp)
    return alpha_new





class DMM_Hard(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001,initialization="kmeans",method="meanprecision",print_log_likelihood=False,max_iter=25, verbose=True):
        super().__init__(n_clusters=n_clusters, EM_type="Hard", mixture_type="identical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.initialization=initialization
        self.method = method
        self.family = "dirichlet"

    def _log_pdf_dirichlet(self,X,alphas):
        threshold = 1e-10
        N,p=X.shape
        k=len(alphas)
        probs=np.empty((N, k))
        for j in range(k):
            alpha=alphas[j]
            for i in range(N):
                with np.errstate(under="ignore"):
                    x=X[i, :]
                    t1=[(alphm-1)*np.log(xm) for alphm,xm in zip(alpha,x)]
                    # t1 = [(alphm - 1) * (np.log1p(xm - 1) if xm < threshold else np.log(xm)) for alphm, xm in zip(alpha, x)]
                    t2=np.sum(gammaln(alpha))
                    t3=gammaln(np.sum(alpha))
                    probs[i,j]=np.sum(t1)-t2+t3
        return probs

    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        return self._log_pdf_dirichlet(X,alpha) + np.log(pi)

    def fit(self,sample, post_m_step=None):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (self.p)
        np.random.seed(0)
        # if self.initialization == "kmeans":
        #     self.pi_not, self.alpha_not = kmeans_init(self.data, self.k)
        # elif self.initialization == "gmm":
        #     self.pi_not, self.alpha_not = gmm_init(self.data, self.k)
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = kmeans_init_adv(self.data, self.k)
        elif self.initialization == "gmm":
            self.pi_not, self.alpha_not = gmm_init_adv(self.data, self.k)
        elif self.initialization == "random":
            self.pi_not, self.alpha_not = random_init(self.data, self.k)

        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not
        pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(self.data,self.pi_temp,self.alpha_temp,estimate_alphas,method=self.method, post_m_step=post_m_step)
        estimated_mean = []
        for a in alpha_new:
            mean_temp = [b/np.sum(a) for b in a]
            estimated_mean.append(mean_temp)
        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.estimated_mean = estimated_mean
        self.cluster = log_gamma_new.argmax(axis=1)
        # self.data_cwise = data_cwise
        self.gamma_temp_ar = np.exp(log_gamma_new)
        # self.gamma_matrix = gamma_matrix
        self.log_likelihood_new = log_likelihood_new
        if self.verbose:
            print("Hard DMM Fitting Done Successfully")
        end_time = time.time()
        self.execution_time=end_time-start_time


    def get_params(self):
        #print("The estimated pi values are ", self.pi_new)
        #print("The estimated alpha values are ", self.alpha_new)

        return self.pi_new, self.alpha_new

    def predict(self):
        return self.cluster

    def predict_new(self, x):
        x=self._process_data(x)
        data_lol = x.tolist()
        cluster, _ = mixture_clusters(self.gamma_temp_ar, data_lol)

        return cluster

    def get_mean(self):

        return self.estimated_mean

    def get_precision(self):
        preci=[np.sum(al) for al in self.alpha_new]

        return preci

    def responsibilities(self):

        return self.gamma_temp_ar

    def n_iter(self):
        return len(self.log_likelihoods)

    def get_info_mat(self, method = "score"):
        IM, SE = combined_info_and_se(self.pi_new, np.array(self.alpha_new), self.gamma_temp_ar, self.data,method, mode = "hard")

        return IM, SE
    
    def get_mean_precision_inference(self, method="score", alpha=0.05):
        I, _ = self.get_info_mat(method=method)
        cov = np.linalg.pinv(I)

        K, p = self.alpha_new.shape
        offset = (K-1)

        results = []

        for j in range(K):
            idx = slice(offset + j*p, offset + (j+1)*p)
            cov_alpha = cov[idx, idx]
            alpha_hat = self.alpha_new[j]

            res = mean_precision_inference(alpha_hat, cov_alpha, alpha)
            results.append(res)

        return results

    # def clustered_data(self):
    #     return pd.DataFrame(self.data_cwise).transpose()

    # def bic(self):

    #     return ((self.k-1)+(self.k*self.p))*np.log(self.n) - 2*(self.log_likelihood_new)

    # def aic(self):

    #     return 2*((self.k-1)+(self.k*self.p)) - 2*(self.log_likelihood_new)

    # def icl(self):
    #     entropy_s=[]
    #     for i in self.gamma_matrix:
    #         for j in i:
    #                 ent_temp=j*np.log1p(j)
    #                 entropy_s.append(ent_temp)



    #     entropy=np.sum(entropy_s)

    #     return ((self.k-1)+(self.k*self.p))*np.log(self.n) - 2*(self.log_likelihood_new) - entropy
