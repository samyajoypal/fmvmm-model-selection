import numpy as np
from numpy.linalg import inv, det
from scipy.stats import norm
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
from numpy.linalg import inv
from scipy.stats import norm
import time
from fmvmm.mixtures._base import BaseMixture
from fmvmm.utils.utils_mixture import (mixture_clusters)
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix, standard_errors_from_info_matrix


def dmvSN(X, mu, Sigma, lam):
    """
    Multivariate Skew-Normal PDF for each row of X, using the "Sahu/arela" 
    parameterization with shape = lam.
    
    X   : (n, p) array
    mu  : (p,) array
    Sigma : (p,p) covariance matrix
    lam : (p,) skewness vector
    
    Returns: (n,) array of PDF values
    """
    X = np.asarray(X)
    mu = np.asarray(mu)
    lam = np.asarray(lam)
    n, p = X.shape

    # Evaluate the multivariate normal part
    
    def dmvnorm(Xi, mu, Sigma):
        # Xi: (p,) single row
        diff = Xi - mu
        invS = inv(Sigma)
        quad = diff @ invS @ diff
        detS = det(Sigma)
        norm_const = (2.0*np.pi)**(p/2.0)*np.sqrt(detS)
        val = np.exp(-0.5*quad)/norm_const
        return val

    
    pdf_vals = np.zeros(n)
    inv_sqrt_Sigma = inv(sqrtm(Sigma + 1e-12*np.eye(p)))
    
    factor = lam @ inv_sqrt_Sigma

    for i in range(n):
        Xi = X[i]
        mvn_val = dmvnorm(Xi, mu, Sigma)  # scalar
        # A = factor dot (X[i] - mu) 
        A = factor @ (Xi - mu)
        # final pdf = 2*mvn_val*Phi(A)
        cdf_val = norm.cdf(A)
        pdf_vals[i] = 2.0*mvn_val*(cdf_val)

    return pdf_vals

def d_mixedmvSN(X, pi_list, mu_list, Sigma_list, lam_list):
    """
    Mixture PDF of g Skew-Normal components, each with parameters
    (mu_j, Sigma_j, lam_j).
    
    X        : (n, p)
    pi_list  : length g
    mu_list, Sigma_list, lam_list : each length g
    Returns: (n,) array of mixture pdf
    """
    g = len(pi_list)
    n = X.shape[0]
    total_pdf = np.zeros(n)
    for j in range(g):
        pdf_j = dmvSN(X, mu_list[j], Sigma_list[j], lam_list[j])
        total_pdf += pi_list[j]*pdf_j
    return total_pdf

def k_means_init_skewnormal(data, k):
    """
    K-Means initialization for Skew-Normal mixture.
    Returns
    -------
    pi_init : (k,) array
    alpha_init : list of (mu_j, Sigma_j, shape_j, no 'nu' needed)
    """
    data = np.asarray(data)
    n, p = data.shape

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)

    mu_init = kmeans.cluster_centers_  # shape (k, p)

    data_cwise = [[] for _ in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data[i])
    data_cwise = [np.asarray(arr).T for arr in data_cwise]

    pi_init = []
    alpha_init = []
    for j in range(k):
        cluster_data = data_cwise[j]  # shape (p, n_j)
        size_j = cluster_data.shape[1]
        pi_init.append(size_j/n)

        if size_j <= 1:
            Sigma_j = np.eye(p)
            shape_j = np.zeros(p)
        else:
            Sigma_j = np.cov(cluster_data)
            # shape from sign of third moment
            mean_diff = cluster_data.T - mu_init[j]
            shape_j = np.sign(np.sum(mean_diff**3, axis=0))

        # No 'nu'. So the alpha is (mu_j, Sigma_j, shape_j, None or something)
        # But we can store shape in the 3rd item, and there's no 4th item needed.
        alpha_init.append((mu_init[j], Sigma_j, shape_j))

    return np.array(pi_init), alpha_init

def estimate_alphas_skewnormal(
    X,
    gamma,                # (n, g) responsibilities
    alpha_prev,           # list of (mu_j, Sigma_j, shape_j) for each cluster j
    pi_prev,              # shape (g,)
    dmvSN_func,
    d_mixedmvSN_func,
    **kwargs
):
    """
    One M-step update for a mixture of multivariate Skew-Normal components,
    replicating the logic from your R code for 'Skew.normal'.
    
    Returns
    -------
    alpha_new : list of (mu_j, Sigma_j, shape_j)
    """

    X = np.asarray(X)
    n, p = X.shape
    g = len(alpha_prev)

    # We will build up alpha_new
    alpha_new = []

    # For each cluster j, we compute the E-step partials (Mtij2, A[i], E[i], etc.)
    # then update mu_j, Delta_j, Gamma_j => Sigma_j => shape_j
    # We'll store S1,S2,S3 as arrays
    S1 = np.zeros((n, g))
    S2 = np.zeros((n, g))
    S3 = np.zeros((n, g))

    # Precompute "delta_j, Delta_j, Gama_j" from shape_j, Sigma_j
    # delta_j = shape_j / sqrt(1 + shape_j^T shape_j)
    # Delta_j = sqrt(Sigma_j) @ delta_j
    # Gama_j  = Sigma_j - Delta_j Delta_j^T
    Delta_old_list = []
    Gama_old_list  = []
    for j in range(g):
        mu_j, Sigma_j, shape_j = alpha_prev[j]
        shape_j = np.asarray(shape_j)
        denom = np.sqrt(1.0 + shape_j @ shape_j)
        delta_j = shape_j / denom
        sqrt_Sigj = sqrtm(Sigma_j + 1e-12*np.eye(p))
        Delta_j = sqrt_Sigj @ delta_j.reshape(-1,1)
        Gama_j  = Sigma_j - Delta_j@Delta_j.T

        Delta_old_list.append(Delta_j)
        Gama_old_list.append(Gama_j)

    # Next we compute for each j:
    
    for j in range(g):
        mu_j, Sigma_j, shape_j = alpha_prev[j]
        Delta_j_old = Delta_old_list[j]
        Gama_j_old  = Gama_old_list[j]

        inv_Gama_j_old = inv(Gama_j_old + 1e-12*np.eye(p))
        # Mtij2_j
        val = (Delta_j_old.T @ inv_Gama_j_old @ Delta_j_old)[0,0]
        Mtij2_j = 1.0/(1.0 + val)
        Mtij_j = np.sqrt(Mtij2_j)

        
        diff = X - mu_j
        factor_j = (Mtij2_j*(Delta_j_old.T @ inv_Gama_j_old)).ravel()  # shape (p,)
        mutij_arr = np.einsum('j,ij->i', factor_j, diff)
        A_arr = mutij_arr/(Mtij_j + 1e-15)

        
        pdf_A = norm.pdf(A_arr)
        cdf_A = np.maximum(norm.cdf(A_arr), 1e-300)  # avoid zero
        E_arr = pdf_A/cdf_A
        
        u_arr = np.ones(n)

        
        gam_j = gamma[:, j]
        S1[:, j] = gam_j*u_arr
        
        S2[:, j] = gam_j*(mutij_arr*u_arr + Mtij_j*E_arr)
        
        S3[:, j] = gam_j*(mutij_arr**2*u_arr + Mtij2_j + Mtij_j*mutij_arr*E_arr)

    # Now we can update (mu_j, Delta_j, Gama_j => Sigma_j => shape_j)
    for j in range(g):
        mu_j_old, Sigma_j_old, shape_j_old = alpha_prev[j]
        Delta_j_old = Delta_old_list[j]

        sum_S1j = np.sum(S1[:, j])
        sum_S2j = np.sum(S2[:, j])
        sum_S3j = np.sum(S3[:, j])

        # mu_j_new
        if sum_S1j < 1e-15:
            mu_j_new = mu_j_old
        else:
            sum_S1_X = np.einsum('i,ij->j', S1[:, j], X)
            sum_S2_Delta = sum_S2j*Delta_j_old.ravel()
            mu_j_new = (sum_S1_X - sum_S2_Delta)/(sum_S1j + 1e-300)

        # Delta_j_new
        dif_new = X - mu_j_new
        sum_S2_dif = np.einsum('i,ij->j', S2[:, j], dif_new)
        Delta_j_new = sum_S2_dif/(sum_S3j + 1e-300)
        Delta_j_new = Delta_j_new.reshape(-1,1)

        # Gama_j_new
        n_j = np.sum(gamma[:, j])
        sum2 = np.zeros((p,p))
        for i in range(n):
            dif_i = dif_new[i].reshape(-1,1)
            s1_val = S1[i, j]
            s2_val = S2[i, j]
            s3_val = S3[i, j]
            sum2 += ( s1_val*(dif_i@dif_i.T)
                       - s2_val*(Delta_j_new@dif_i.T)
                       - s2_val*(dif_i@Delta_j_new.T)
                       + s3_val*(Delta_j_new@Delta_j_new.T) )
        Gama_j_new = sum2/(n_j + 1e-300)
        Sigma_j_new = Gama_j_new + Delta_j_new@Delta_j_new.T

        # shape_j_new
        inv_sqrt_Sigma_j_new = inv(sqrtm(Sigma_j_new + 1e-12*np.eye(p)))
        numerator = inv_sqrt_Sigma_j_new @ Delta_j_new
        denom_ = 1.0 - (Delta_j_new.T @ inv(Sigma_j_new + 1e-15*np.eye(p)) @ Delta_j_new)
        denom_ = np.sqrt(np.maximum(1e-300, denom_[0,0]))
        shape_j_new = (numerator/denom_).ravel()

        alpha_new.append((mu_j_new, Sigma_j_new, shape_j_new))

    return alpha_new

class SkewNormalMix(BaseMixture):
    """
    Mixture of Multivariate Skew-Normal distributions,
    each with parameter (mu_j, Sigma_j, shape_j).
    No 'nu' parameter is needed. 
    """
    def __init__(self,
                 n_clusters,
                 tol=1e-4,
                 initialization="kmeans",
                 print_log_likelihood=False,
                 max_iter=25,
                 verbose=True):
        super().__init__(
            n_clusters=n_clusters,
            EM_type="Soft",
            mixture_type="identical",
            tol=tol,
            print_log_likelihood=print_log_likelihood,
            max_iter=max_iter,
            verbose=verbose
        )
        self.k = n_clusters
        self.initialization = initialization
        self.family = "skew_normal" 

    def _log_pdf_skewnormal(self, X, alphas):
        """
        Evaluate log of Skew-Normal pdf for each row in X, for each cluster j.
        alphas[j] = (mu_j, Sigma_j, shape_j)
        Returns: (n, g)
        """
        N, p = X.shape
        g = len(alphas)
        logvals = np.zeros((N, g))
        for j in range(g):
            mu_j, Sigma_j, shape_j = alphas[j]
            dens_j = dmvSN(X, mu_j, Sigma_j, shape_j)
            dens_j = np.maximum(dens_j, 1e-300)
            logvals[:, j] = np.log(dens_j)
        return logvals

    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        """
        Weighted log-prob: log( f_j(X) ) + log(pi_j)
        """
        log_pdf = self._log_pdf_skewnormal(X, alpha)  # (n, g)
        log_pi  = np.log(pi)                          # (g,)
        return log_pdf + log_pi

    def fit(self, sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (2*self.p + (self.p*(self.p+1)/2) +1)

        # 1) Initialization
        if self.initialization == "kmeans":
            self.pi_not, alpha_init = k_means_init_skewnormal(self.data, self.k)
        else:
            raise NotImplementedError("Only 'kmeans' init is supported for SkewNormalMix.")

        self.alpha_not  = alpha_init
        self.alpha_temp = alpha_init
        self.pi_temp    = self.pi_not

        # Provide references for M-step
        def dmvSN_local(X_, mu_, Sigma_, shape_):
            return dmvSN(X_, mu_, Sigma_, shape_)
        def d_mixedmvSN_local(X_, pi_, mu_list, Sigma_list, shape_list):
            return d_mixedmvSN(X_, pi_, mu_list, Sigma_list, shape_list)

        self.dmvSN_func = dmvSN_local
        self.d_mixedmvSN_func = d_mixedmvSN_local

        # 2) Run the base-class _fit
        pi_new, alpha_new, log_likelihood_new, log_gamma_new = self._fit(
            self.data, 
            self.pi_temp,
            self.alpha_temp,
            estimate_alphas_function=self._estimate_alphas_wrapper
        )

        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.cluster = log_gamma_new.argmax(axis=1)
        self.gamma_temp_ar = np.exp(log_gamma_new)
        self.log_likelihood_new = log_likelihood_new

        if self.verbose:
            print("Mixtures of Skew-Normal Fitting Done Successfully")
        end_time = time.time()
        self.execution_time = end_time - start_time

    def _estimate_alphas_wrapper(self, X, gamma, alpha_prev, **kwargs):
        """
        A wrapper that calls estimate_alphas_skewnormal for a single M-step.
        """
        pi_prev = self.pi_temp
        alpha_new = estimate_alphas_skewnormal(
            X,
            gamma,
            alpha_prev,
            pi_prev,
            dmvSN_func=self.dmvSN_func,
            d_mixedmvSN_func=self.d_mixedmvSN_func,
            **kwargs
        )
        return alpha_new
    
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

    def responsibilities(self):

        return self.gamma_temp_ar

    def n_iter(self):
        return len(self.log_likelihoods)

    def get_info_mat(self):
        
        im = info_matrix(self)
        se = standard_errors_from_info_matrix(im)
        
        return im, se
