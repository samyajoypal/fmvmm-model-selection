import numpy as np
from numpy.linalg import inv, det
from scipy.linalg import sqrtm
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_func
import time
from fmvmm.mixtures._base import BaseMixture
from sklearn.cluster import KMeans
from fmvmm.utils.utils_mixture import (mixture_clusters)
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix, standard_errors_from_info_matrix


def dmvSS(X, mu, Sigma, lam, nu):
    """
    Multivariate Slash (or Skew-Slash) density for each row of X, 
    as in your prior code. If lam=0, it becomes the standard Slash distribution.

    X  : (n, p)
    mu : (p,)
    Sigma : (p, p)
    lam : (p,)  (for standard slash, lam=0)
    nu : scalar slash parameter
    Returns: (n,) array of PDF values
    """

    X = np.asarray(X)
    mu = np.asarray(mu)
    lam = np.asarray(lam)
    n, p = X.shape

    # Safeguard for near-singular Sigma
    det_Sig = np.linalg.det(Sigma)
    if det_Sig < 1e-300:
        Sigma = Sigma + 1e-6*np.eye(p)
        det_Sig = np.linalg.det(Sigma)

    inv_Sig = inv(Sigma)
    diff = X - mu
    dj = np.sum((diff @ inv_Sig)*diff, axis=1)  # Mahalanobis distances

    # factor for lam != 0 (Skew-Slash), lam=0 => factor is zero => pnorm(0)=0.5
    sqrt_Sig = sqrtm(Sigma + 1e-12*np.eye(p))
    inv_sqrt_Sig = inv(sqrt_Sig)
    factor = lam @ inv_sqrt_Sig  # shape (p,)
    A_vals = np.einsum('j,ij->i', factor, diff)  # shape (n,)

    pdf_vals = np.zeros(n)

    
    for i in range(n):
        dj_i = dj[i]
        A_i = A_vals[i]

        # Evaluate dmvSS by direct integral from 0..1
        def integrand(u):
            return (2.0*nu*(u**(nu-1.0))
                    * ((u/(2.0*np.pi))**(p/2.0))/np.sqrt(det_Sig)
                    * np.exp(-0.5*u*dj_i)
                    * norm.cdf(np.sqrt(u)*A_i))

        val, _ = quad(integrand, 0.0, 1.0, limit=50)
        pdf_vals[i] = val

    return pdf_vals

def d_mixedmvSS(X, pi_list, mu_list, Sigma_list, lam_list, nu):
    """
    Mixture of multivariate Slash (or Skew-Slash) distributions, 
    all sharing the same slash parameter nu, possibly lam=0 for each cluster 
    for pure slash.

    pi_list : length g
    mu_list, Sigma_list, lam_list : each length g
    nu : scalar
    Returns: (n,) array of mixture pdf values
    """
    g = len(pi_list)
    n = X.shape[0]
    total = np.zeros(n)
    for j in range(g):
        dens_j = dmvSS(X, mu_list[j], Sigma_list[j], lam_list[j], nu)
        total += pi_list[j]*dens_j
    return total

def estimate_alphas_slash(
    X,
    gamma,                   # responsibilities (n, g)
    alpha_prev,              # list of (mu_j, Sigma_j, shape_j=0, nu)
    pi_prev,                 # (g,)
    dmvSS_func,
    d_mixedmvSS_func,
    bounds_nu=(1e-4, 100.0),
    **kwargs
):
    """
    Single M-step update for a mixture of (non-skew) Slash components.
    shape=0 => symmetrical slash distribution.

    Returns
    -------
    alpha_new : list of length g => (mu_j, Sigma_j, shape_j=0, nu_new)
    """
    X = np.asarray(X)
    n, p = X.shape
    g = len(alpha_prev)

    # old nu (single shared)
    nu_old = alpha_prev[0][3]

    # Extract old parameters
    mu_old_list = [alpha_prev[j][0] for j in range(g)]
    Sigma_old_list = [alpha_prev[j][1] for j in range(g)]
    # shape_j=0 for slash => do not need a real shape

    # We'll compute E-step-like quantities: u[i] and E[i].
    # Because shape=0 => Delta=0 => A=0 => pnorm(0)=0.5 in the integral, etc.

    # We'll accumulate S1, S2, S3
    S1 = np.zeros((n, g))
    S2 = np.zeros((n, g))
    S3 = np.zeros((n, g))

    from numpy.linalg import inv
    from scipy.linalg import sqrtm
    from scipy.stats import norm
    from scipy.integrate import quad

    for j in range(g):
        mu_j = mu_old_list[j]
        Sigma_j = Sigma_old_list[j]

        # shape=0 => Delta=0 => Gama_j = Sigma_j
        # Mtij2 => 1/(1 + 0) => 1 => Mtij=1 => mutij[i]=0 => A=0
        # So a lot of the E-step formulas simplify.

        # pdf_j = dmvSS(X, mu_j, Sigma_j, shape=0, nu_old)
        pdf_j = dmvSS_func(X, mu_j, Sigma_j, np.zeros(p), nu_old)
        pdf_j = np.maximum(pdf_j, 1e-300)

        
        # 1) Compute dj
        inv_Sigma_j = inv(Sigma_j + 1e-12*np.eye(p))
        diff = X - mu_j
        dj_arr = np.sum((diff @ inv_Sigma_j)*diff, axis=1)

        # 2) Compute u[i]
        
        det_Sig_j = max(1e-300, np.linalg.det(Sigma_j))
        c_val_u = (nu_old*(2.0**(1 - p/2.0))*(np.pi**(-0.5*p))*(det_Sig_j**(-0.5)))

        def integrand_u(u, di):
            return u**(nu_old + p/2.0)*np.exp(-0.5*u*di)*0.5  # 0.5 from pnorm(0)

        u_arr = np.zeros(n)
        for i in range(n):
            di = dj_arr[i]
            val_int, _ = quad(lambda u: integrand_u(u, di), 0.0, 1.0, limit=50)
            u_arr[i] = (c_val_u*val_int) / pdf_j[i]

        # 3) Compute E[i]
        
        from scipy.stats import gamma as gamma_dist
        cE_num = (2.0**(nu_old+1))*nu_old*gamma_func((2*nu_old+p+1)/2.0)
        cE_den = ((np.pi)**(0.5*(p+1)))*np.sqrt(det_Sig_j)
        cE = cE_num/(cE_den + 1e-300)

        E_arr = np.zeros(n)
        for i in range(n):
            val_pdf = pdf_j[i]
            di = dj_arr[i]
            
            shape_ = (2*nu_old+p+1)/2.0
            rate_  = di/2.0
            cdf_val = gamma_dist.cdf(1.0, a=shape_, scale=1.0/(rate_+1e-15))

            power_factor = (di**(-(2*nu_old+p+1)/2.0)) if di>0 else 1e+15  # caution if di=0
            E_i = (cE*power_factor*cdf_val)/(val_pdf+1e-300)
            E_arr[i] = E_i

        # 4) S1, S2, S3 with shape=0 => Delta=0 => mutij=0 => Mtij=1 => 
        
        gam_j = gamma[:, j]
        S1[:, j] = gam_j*u_arr
        S2[:, j] = gam_j*E_arr
        S3[:, j] = gam_j  # since shape=0 => that formula is 1

    # Now update parameters:
    alpha_new_partial = []
    for j in range(g):
        mu_j_old = mu_old_list[j]
        sum_S1_j = np.sum(S1[:, j])
        sum_S2_j = np.sum(S2[:, j])
        sum_S3_j = np.sum(S3[:, j])

        
        if sum_S1_j < 1e-15:
            mu_j_new = mu_j_old
        else:
            sum_S1_X = np.einsum('i,ij->j', S1[:, j], X)
            mu_j_new = sum_S1_X/(sum_S1_j + 1e-300)

        
        dif_new = X - mu_j_new
        sum2 = np.zeros((p,p))
        for i in range(n):
            sum2 += S1[i, j]*np.outer(dif_new[i], dif_new[i])
        n_j = np.sum(gamma[:, j])
        Gama_j_new = sum2/(n_j + 1e-300)
        Sigma_j_new = Gama_j_new  # shape=0 => Delta=0 => Sigma=Gama

        # shape_j_new=0 => forcibly zero
        shape_j_new = np.zeros(p)

        alpha_new_partial.append((mu_j_new, Sigma_j_new, shape_j_new, nu_old))

    # Finally, update nu by maximizing the mixture log-likelihood
    def neg_loglik_slash(nu_val):
        mu_list = [a[0] for a in alpha_new_partial]
        Sig_list= [a[1] for a in alpha_new_partial]
        lam_list= [a[2] for a in alpha_new_partial]  # all zeros
        pdf_mix = d_mixedmvSS_func(X, pi_prev, mu_list, Sig_list, lam_list, nu_val)
        pdf_mix = np.maximum(pdf_mix, 1e-300)
        return -np.sum(np.log(pdf_mix))

    res = minimize_scalar(
        neg_loglik_slash,
        bounds=bounds_nu,
        method='bounded',
        options={"xatol":1e-6}
    )
    nu_new = res.x

    alpha_new = []
    for j in range(g):
        mu_j, Sig_j, shp_j, _ = alpha_new_partial[j]
        alpha_new.append((mu_j, Sig_j, shp_j, nu_new))

    return alpha_new

def k_means_init_slash(data, k):
    """
    K-means initialization for a mixture of (non-skew) Slash distributions.
    shape=0 by default.
    """
    data = np.asarray(data)
    n, p = data.shape

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)

    mu_init = kmeans.cluster_centers_  # (k,p)

    # group data
    data_cwise = [[] for _ in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data[i])
    data_cwise = [np.asarray(arr).T for arr in data_cwise]

    pi_init = []
    alpha_init = []
    # We pick a default slash parameter, e.g. nu=2
    for j in range(k):
        cluster_data = data_cwise[j]
        size_j = cluster_data.shape[1]
        pi_init.append(size_j/n)

        if size_j <= 1:
            Sigma_j = np.eye(p)
        else:
            Sigma_j = np.cov(cluster_data)

        # shape=0
        shape_j = np.zeros(p)
        nu_j = 2.0

        alpha_init.append((mu_init[j], Sigma_j, shape_j, nu_j))

    return np.array(pi_init), alpha_init


class SlashMix(BaseMixture):
    """
    Mixture of Multivariate Slash (non-skew) distributions,
    sharing a single slash parameter 'nu'.
    shape=0 => symmetrical slash.
    """

    def __init__(self,
                 n_clusters,
                 tol=1e-4,
                 initialization="kmeans",
                 print_log_likelihood=False,
                 max_iter=25,
                 verbose=True):
        super().__init__(n_clusters=n_clusters,
                         EM_type="Soft",
                         mixture_type="identical",
                         tol=tol,
                         print_log_likelihood=print_log_likelihood,
                         max_iter=max_iter,
                         verbose=verbose)
        self.k = n_clusters
        self.initialization = initialization
        self.family = "slash"

    def _log_pdf_slash(self, X, alphas):
        """
        Evaluate log-pdf of each cluster's Slash distribution at each row of X.
        alphas[j] = (mu_j, Sigma_j, shape_j=0, nu_j)
        Returns: (n, g) array of log densities
        """
        N, p = X.shape
        g = len(alphas)
        logvals = np.empty((N, g))
        for j in range(g):
            mu_j, Sigma_j, shape_j, nu_j = alphas[j]
            # shape_j=0 => standard slash
            pdf_j = dmvSS(X, mu_j, Sigma_j, shape_j, nu_j)
            pdf_j = np.maximum(pdf_j, 1e-300)
            logvals[:, j] = np.log(pdf_j)
        return logvals

    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        """
        Weighted log-prob for identical mixture type: log f_j(X) + log pi_j
        """
        log_pdf = self._log_pdf_slash(X, alpha)  # (n, g)
        log_pi = np.log(pi)                      # (g,)
        return log_pdf + log_pi

    def fit(self, sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (self.p + (self.p*(self.p+1)/2) +1)

        # Initialization
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = k_means_init_slash(self.data, self.k)
        else:
            raise NotImplementedError("Only 'kmeans' initialization is supported for SlashMix.")

        self.alpha_temp = self.alpha_not
        self.pi_temp    = self.pi_not

        # Provide references for M-step
        def dmvSS_local(data, mu, Sigma, lam, nu):
            return dmvSS(data, mu, Sigma, lam, nu)
        def d_mixedmvSS_local(data, pi_list, mu_list, Sigma_list, lam_list, nu):
            return d_mixedmvSS(data, pi_list, mu_list, Sigma_list, lam_list, nu)

        self.dmvSS_func = dmvSS_local
        self.d_mixedmvSS_func = d_mixedmvSS_local

        # Run the base-class EM
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
            print("Mixtures of Slash Fitting Done Successfully!")
        end_time = time.time()
        self.execution_time = end_time - start_time

    def _estimate_alphas_wrapper(self, X, gamma, alpha_prev, **kwargs):
        """
        A wrapper for the M-step: calls estimate_alphas_slash once.
        """
        pi_prev = self.pi_temp
        alpha_new = estimate_alphas_slash(
            X,
            gamma,
            alpha_prev,
            pi_prev,
            dmvSS_func=self.dmvSS_func,
            d_mixedmvSS_func=self.d_mixedmvSS_func,
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
