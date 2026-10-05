import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import inv, det
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from math import gamma as gamma_func, pi
from scipy.stats import t as student_t
import time
from fmvmm.mixtures._base import BaseMixture
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST
from fmvmm.utils.utils_mixture import (mixture_clusters)
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix, standard_errors_from_info_matrix

def k_means_init_t(data, k):
    """
    K-means initialization for a T mixture model.

    Returns
    -------
    pi_not : list of length k
        Mixing proportions
    alpha_not : list of length k
        Each element is (mu_j, Sigma_j, shape_j, nu_j).
        Where shape_j = 0 (vector of zeros) and nu_j = 4 for a start.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    n, p = data.shape

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)

    # cluster means
    mu_init = kmeans.cluster_centers_

    # grouping data by cluster
    data_cwise = [[] for _ in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data[i])
    data_cwise = [np.asarray(arr).T for arr in data_cwise]  # shape (p, n_j)

    Sigma_init = []
    shape_init = []
    nu_init = 4.0  # You can pick any reasonable start

    for j in range(k):
        cluster_data = data_cwise[j]
        if cluster_data.shape[1] > 1:
            Sigma_j = np.cov(cluster_data)  # shape (p, p)
        else:
            # if there's only one point, fallback
            Sigma_j = np.eye(p)
        Sigma_init.append(Sigma_j)
        shape_init.append(np.zeros(p))   # shape=0 => standard T

    pi_init = [len(data_cwise[j].T)/n for j in range(k)]

    # alpha_not = list of (mu_j, Sigma_j, shape_j, nu_j)
    alpha_not = []
    for j in range(k):
        alpha_not.append((
            mu_init[j],          # mu_j
            Sigma_init[j],       # Sigma_j
            shape_init[j],       # shape_j=0
            nu_init              # same nu across all clusters initially
        ))

    return pi_init, alpha_not


def estimate_alphas_t(
    X, 
    gamma,                 # shape (n, g)
    alpha_prev,            # list of (mu_j, Sigma_j, shape_j=0, nu) for each cluster j
    pi_prev,               # shape (g,)
    dmvt_ls_func,          # function dmvt_ls(X, mu, Sigma, shape, nu) 
    d_mixedmvST_func,      # function d_mixedmvST(X, pi, mu_list, Sigma_list, shape_list, nu)
    bounds_nu=(1e-4, 100.0),
    tol_nu=1e-6
):
    """
    One M-step update for mixture of T components, 
    following R code for family = 't'. 
    shape_j is forced to 0 for all j.

    Returns
    -------
    alpha_new: list of tuples (mu_j, Sigma_j, shape_j=0, nu_new)
    """
    n, p = X.shape
    g = len(alpha_prev)
    # The old common nu:
    nu_old = alpha_prev[0][3]  # same for all clusters

    mu_old_list = [alpha_prev[j][0] for j in range(g)]
    Sigma_old_list = [alpha_prev[j][1] for j in range(g)]
    # shape is zero => we skip storing it
    # we do keep them in the final alpha_new though

    # We'll create placeholders for new parameters
    mu_new_list = [None]*g
    Sigma_new_list = [None]*g

    # For numeric stability
    eps = 1e-10

    # Precompute cluster pdf for each j: 
    cluster_pdf = np.zeros((n, g))
    for j in range(g):
        mu_j, Sigma_j, shape_j, _ = alpha_prev[j]
        cluster_pdf[:, j] = dmvt_ls_func(X, mu_j, Sigma_j, shape_j, nu_old)

    # Mixture pdf:
    mix_pdf = np.sum(pi_prev * cluster_pdf, axis=1)
    mix_pdf = np.maximum(mix_pdf, 1e-300)

    

    def dmvt_no_skew(Xi, mu_j, Sigma_j, nu):
        """T-dist pdf ignoring skew, single row Xi => scalar pdf."""
        # Just call dmvt_ls_func with shape=0
        return dmvt_ls_func(Xi[np.newaxis,:], mu_j, Sigma_j, np.zeros(p), nu)[0]


    # We'll gather sums for S1, S2, S3 in each cluster.
    from math import gamma as GammaFunc

    for j in range(g):
        mu_j, Sigma_j, shape_j, _ = alpha_prev[j]
        # shape_j = 0 => typical T
        # needed constants
        sqrt_det_Sigma_j = np.sqrt(np.maximum(np.linalg.det(Sigma_j), 1e-300))

        # cluster pdf for j, all i
        pdf_j = cluster_pdf[:, j]
        pdf_j = np.maximum(pdf_j, 1e-300)

        # dj[i] = (X[i] - mu_j)^T Sigma_j^-1 (X[i] - mu_j)
        inv_Sigma_j = inv(Sigma_j)
        dif = X - mu_j
        dj = np.sum((dif @ inv_Sigma_j)*dif, axis=1)

        # We'll compute E[i], u[i] for i in [0..n-1]
        # Precompute constants:
        cE_num = 2.0*(nu_old**(nu_old/2.0))*GammaFunc((p+nu_old+1)/2.0)
        cE_den = GammaFunc(nu_old/2.0)*(pi**((p+1)/2.0))*sqrt_det_Sigma_j
        cE = cE_num / (cE_den + 1e-300)

        cU_num = 4.0*(nu_old**(nu_old/2.0))*GammaFunc((p+nu_old+2)/2.0)
        cU_den = GammaFunc(nu_old/2.0)*(pi**(p/2.0))*sqrt_det_Sigma_j
        cU = cU_num / (cU_den + 1e-300)

        

        E_arr = cE*(dj + nu_old)**(-(p+nu_old+1)/2.0) / pdf_j
        u_arr = cU*(dj + nu_old)**(-(p+nu_old+2)/2.0)*0.5 / pdf_j

        
        S1 = gamma[:, j]*u_arr
        S2 = gamma[:, j]*E_arr
        S3 = gamma[:, j]  # since shape=0 => everything else is zero/1

        # M-step updates:
        
        sum_S1 = np.sum(S1)
        if sum_S1 < 1e-15:
            # degenerate; fallback:
            mu_j_new = mu_j
        else:
            mu_j_new = np.einsum('i,ij->j', S1, X) / (sum_S1 + eps)

        # Then Sigma_j_new:
        
        dif_new = X - mu_j_new
        sum2 = np.zeros((p,p))
        for i in range(n):
            sum2 += S1[i]*(dif_new[i].reshape(-1,1) @ dif_new[i].reshape(1,-1))
        n_j = np.sum(gamma[:, j])
        Gama_j_new = sum2 / (n_j + eps)

        Sigma_j_new = Gama_j_new  # since Delta=0 => no additional term

        mu_new_list[j] = mu_j_new
        Sigma_new_list[j] = Sigma_j_new

    # Finally, update the common nu by maximizing the log-likelihood:

    def loglik_t(nu_val):
        # build alpha_temp with shape=0, mu_new_list, Sigma_new_list, nu_val
        # then compute mixture pdf
        alpha_temp = []
        for j in range(g):
            alpha_temp.append((mu_new_list[j], Sigma_new_list[j], np.zeros(p), nu_val))
        pdf_mix = d_mixedmvST_func(X, pi_prev, 
                                   mu=[a[0] for a in alpha_temp],
                                   Sigma=[a[1] for a in alpha_temp],
                                   lambda_=[a[2] for a in alpha_temp],
                                   nu=nu_val)
        pdf_mix = np.maximum(pdf_mix, 1e-300)
        return np.sum(np.log(pdf_mix))

    # Minimize negative log-likelihood
    res = minimize_scalar(lambda x: -loglik_t(x),
                          bounds=bounds_nu, method='bounded',
                          options={"xatol": tol_nu})
    nu_new = res.x

    # Build alpha_new
    alpha_new = []
    for j in range(g):
        alpha_new.append((mu_new_list[j], Sigma_new_list[j], np.zeros(p), nu_new))

    return alpha_new


class TMix(BaseMixture):
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
        self.family = "t"

    def _log_pdf_t(self, X, alphas):
        """
        Evaluate log of T pdf for each point in X, for each cluster j.

        alphas: list of (mu_j, Sigma_j, shape_j=0, nu_j),
        but we assume a single nu_j for all j, or at least the code does.
        """
        N, p = X.shape
        g = len(alphas)
        probs = np.empty((N, g))
        for j in range(g):
            mu_j, Sigma_j, shape_j, nu_j = alphas[j]
            # shape_j=0 => standard T 
            # We rely on your dmvt_ls for shape=0 => standard T
            pdf_j = self.dmvt_ls_func(X, mu_j, Sigma_j, shape_j, nu_j)
            with np.errstate(under="ignore"):
                probs[:, j] = np.log(pdf_j)
        return probs

    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        """
        Weighted log-prob for 'identical' mixture type:
        log( pdf_j(X) ) + log(pi_j )
        """
        log_pdf = self._log_pdf_t(X, alpha)   # shape (n, g)
        log_pi = np.log(pi)
        return log_pdf + log_pi

    def fit(self, sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (self.p + (self.p*(self.p+1)/2) +1)
        np.random.seed(0)

        # Initialization
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = k_means_init_t(self.data, self.k)
        else:
            raise NotImplementedError("Only kmeans initialization is implemented for TMix so far.")

        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not

        # Provide references to the needed functions for the M-step
        # We'll define or pass them in:
        def dmvt_ls_local(Y, mu, Sigma, shape, nu):
            return dmvt_ls(Y, mu, Sigma, shape, nu)  # your existing function

        def d_mixedmvST_local(Y, pi_array, mu, Sigma, lambda_, nu):
            return d_mixedmvST(Y, pi_array, mu, Sigma, lambda_, nu)  # your function

        self.dmvt_ls_func = dmvt_ls_local
        self.d_mixedmvST_func = d_mixedmvST_local

        # Now run the base-class _fit loop
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
            print("Mixtures of T-Fitting Done Successfully")
        end_time = time.time()
        self.execution_time = end_time - start_time

    def _estimate_alphas_wrapper(self, X, gamma, alpha_prev, **kwargs):
        """
        A wrapper that calls your 'estimate_alphas_t' once for M-step.
        We reconstruct pi_prev from the last E-step.
        """
        # We rely on self.pi_temp as the "pi_prev" inside the M-step,
        # or pass it explicitly from the base class's M-step function.
        pi_prev = self.pi_temp
        alpha_new = estimate_alphas_t(
            X,
            gamma,
            alpha_prev,
            pi_prev,
            dmvt_ls_func=self.dmvt_ls_func,
            d_mixedmvST_func=self.d_mixedmvST_func,
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