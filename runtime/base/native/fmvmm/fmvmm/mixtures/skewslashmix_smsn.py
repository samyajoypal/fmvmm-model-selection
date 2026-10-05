import numpy as np
from numpy.linalg import inv, det
from scipy.stats import norm
from scipy.linalg import sqrtm
from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.special import gamma as gamma_func
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import time
from fmvmm.mixtures._base import BaseMixture
from sklearn.cluster import KMeans
from fmvmm.utils.utils_mixture import (mixture_clusters)
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix, standard_errors_from_info_matrix

def dmvSS(X, mu, Sigma, lam, nu):
    """
    Multivariate Skew-Slash density for each row of X.
    X   : (n, p) array
    mu  : (p,) array
    Sigma : (p,p) covariance matrix
    lam : (p,) skewness vector
    nu  : scalar (the 'slash' parameter > 0)
    
    Returns: (n,) array of PDF values
    """

    X = np.asarray(X)
    mu = np.asarray(mu)
    lam = np.asarray(lam)
    n, p = X.shape

    # For stability, regularize Sigma if near-singular
    det_Sig = np.linalg.det(Sigma)
    if det_Sig < 1e-300:
        Sigma = Sigma + 1e-6*np.eye(p)
        det_Sig = np.linalg.det(Sigma)

    inv_Sig = np.linalg.inv(Sigma)
    sqrt_Sig = sqrtm(Sigma + 1e-12*np.eye(p))  # safer for numeric
    inv_sqrt_Sig = inv(sqrt_Sig)

    # Precompute the factor for the integrand:
    
    diff = X - mu
    factor = lam @ inv_sqrt_Sig  # shape (p,)
    factor_vals = np.einsum('j,ij->i', factor, diff)  # shape (n,)

    # Mahalanobis distance:
    # dj[i] = (X[i] - mu) ^T * inv_Sig * (X[i] - mu)
    dj = np.sum((diff @ inv_Sig) * diff, axis=1)

    
    def single_pdf(dj_i, fac_i):
        def integrand(u):
            # (u/(2*pi))^(p/2):
            #   = u^(p/2) / ( (2*pi)^(p/2) )
            return (2.0*nu*(u**(nu-1.0)) 
                    * ( (u/(2.0*np.pi))**(p/2.0) / np.sqrt(det_Sig) )
                    * np.exp(-0.5*u*dj_i)
                    * norm.cdf(np.sqrt(u)*fac_i) )
        val, _ = quad(integrand, 0.0, 1.0, limit=50)  # You might adjust 'limit'
        return val

    pdf_vals = np.zeros(n)
    for i in range(n):
        pdf_vals[i] = single_pdf(dj[i], factor_vals[i])

    return pdf_vals

def d_mixedmvSS(X, pi_list, mu_list, Sigma_list, lam_list, nu):
    """
    Mixture PDF of Skew-Slash components (common nu).
    X : (n, p)
    pi_list: length g
    mu_list, Sigma_list, lam_list: each length g
    nu : scalar slash parameter
    Returns: (n,) array of mixture densities
    """
    g = len(pi_list)
    n = X.shape[0]
    total_pdf = np.zeros(n)
    for j in range(g):
        dens_j = dmvSS(X, mu_list[j], Sigma_list[j], lam_list[j], nu)
        total_pdf += pi_list[j]*dens_j
    return total_pdf

def estimate_alphas_skewslash(
    X,
    gamma,                  # responsibilities, shape (n, g)
    alpha_prev,             # list of (mu_j, Sigma_j, shape_j, nu)
    pi_prev,                # shape (g,)
    dmvSS_func,             # function dmvSS(X, mu, Sigma, lam, nu)
    d_mixedmvSS_func,       # mixture pdf function
    bounds_nu=(1e-4, 100.0),
    **kwargs
):
    """
    Single M-step update for a mixture of Skew-Slash components, 
    replicating the R code logic (family == "Skew.slash").

    X        : (n, p)
    gamma    : (n, g) responsibilities
    alpha_prev : list of length g => (mu_j, Sigma_j, shape_j, nu_old)
                 We assume a single shared 'nu_old' across all clusters.
    pi_prev  : (g,) array of mixing proportions
    dmvSS_func      : function to evaluate a single Skew-Slash PDF
    d_mixedmvSS_func: function to evaluate the entire mixture
    bounds_nu       : bounds for searching the slash parameter nu

    Returns
    -------
    alpha_new : list of length g => (mu_j, Sigma_j, shape_j, nu_new)
    """

    X = np.asarray(X)
    n, p = X.shape
    g = len(alpha_prev)

    # old nu is the same for all clusters:
    nu_old = alpha_prev[0][3]

    mu_old_list    = [alpha_prev[j][0] for j in range(g)]
    Sigma_old_list = [alpha_prev[j][1] for j in range(g)]
    shape_old_list = [alpha_prev[j][2] for j in range(g)]

    # We'll store updated parameters
    alpha_temp = []

    
    Delta_old = []
    Gama_old = []
    for j in range(g):
        shape_j = shape_old_list[j]
        shape_j = np.asarray(shape_j)
        denom = np.sqrt(1.0 + shape_j @ shape_j)
        delta_j = shape_j / denom
        sqrt_Sigj = sqrtm(Sigma_old_list[j] + 1e-12*np.eye(p))
        Delta_j = sqrt_Sigj @ delta_j.reshape(-1,1)
        Gama_j = Sigma_old_list[j] - Delta_j@Delta_j.T
        Delta_old.append(Delta_j)
        Gama_old.append(Gama_j)

    

    S1 = np.zeros((n, g))
    S2 = np.zeros((n, g))
    S3 = np.zeros((n, g))

    from numpy.linalg import inv

    for j in range(g):
        mu_j = mu_old_list[j]
        Sigma_j = Sigma_old_list[j]
        shape_j = shape_old_list[j]
        Delta_j_old = Delta_old[j]
        Gama_j_old  = Gama_old[j]

        inv_Gama_j_old = inv(Gama_j_old + 1e-12*np.eye(p))

        # Mtij2
        val = (Delta_j_old.T @ inv_Gama_j_old @ Delta_j_old)[0,0]
        Mtij2_j = 1.0/(1.0 + val)
        Mtij_j  = np.sqrt(Mtij2_j)

        
        diff = X - mu_j
        factor_j = (Mtij2_j*(Delta_j_old.T @ inv_Gama_j_old)).ravel()  # shape (p,)
        mutij_arr = np.einsum('j,ij->i', factor_j, diff)
        A_arr = mutij_arr/(Mtij_j + 1e-15)

        # dj[i] = Mahalanobis
        inv_Sigma_j = inv(Sigma_j + 1e-12*np.eye(p))
        dj = np.sum((diff @ inv_Sigma_j)*diff, axis=1)

       

        # ============= Compute the "u[i]" via integrate =============
        def single_u_integral(di_val, A_val):
            """
            For a single data i: 
              faux(u) = u^(nu + p/2)*exp(-u*di_val/2)*pnorm(sqrt(u)*A_val)
            Then 
            u[i] = nu * 2^(1-p/2)*pi^(-p/2)*det(Sigma_j)^(-1/2)* 
                   (integrate of faux(u) from 0..1) / dmvSS(...)
            """
            from scipy.integrate import quad

            def faux(u):
                return (u**(nu_old + p/2.0)
                        * np.exp(-0.5*u*di_val)
                        * norm.cdf(np.sqrt(u)*A_val))

            # integrate
            aux22, _ = quad(faux, 0.0, 1.0, limit=50)
            
            det_Sigma_j = max(1e-300, np.linalg.det(Sigma_j))
            valC = (nu_old*(2.0**(1 - p/2.0))
                    * (np.pi**(-0.5*p))
                    * (det_Sigma_j**(-0.5)))
            return valC*aux22

            # We'll divide by dmvSS(...) later

        # We'll gather them for i in a loop:
        pdf_j = dmvSS_func(X, mu_j, Sigma_j, shape_j, nu_old)
        pdf_j = np.maximum(pdf_j, 1e-300)

        u_arr = np.zeros(n)
        for i in range(n):
            # integrand
            c_val = single_u_integral(dj[i], A_arr[i])
            u_arr[i] = c_val / pdf_j[i]

        # ============= Compute E array =============
        
        det_Sigma_j = max(1e-300, np.linalg.det(Sigma_j))
        cE1_num = (2.0**(nu_old+1)*nu_old*gamma_func((2*nu_old+p+1)/2.0))
        cE1_den = ((np.pi)**(0.5*(p+1)))*np.sqrt(det_Sigma_j)
        # => cE = cE1_num/cE1_den, then we divide by pdf_j
        cE = cE1_num/(cE1_den + 1e-300)

        
        from scipy.special import gammainc

        E_arr = np.zeros(n)
        for i in range(n):
            denom_pdf = pdf_j[i]
            val = (dj[i] + A_arr[i]**2)
            power_factor = val**(-(2*nu_old+p+1)/2.0)

            
            from scipy.stats import gamma as gamma_dist

            shape_ = (2*nu_old + p + 1)/2.0
            rate_ = (dj[i] + A_arr[i]**2)/2.0
            # scale = 1/rate_
            cdf_val = gamma_dist.cdf(1.0, a=shape_, scale=1.0/(rate_+1e-15))

            E_val = cE * power_factor * cdf_val / (denom_pdf+1e-300)
            E_arr[i] = E_val

        # Now define S1[i,j], S2[i,j], S3[i,j]:
        gam_j = gamma[:, j]
        S1[:, j] = gam_j*u_arr
        S2[:, j] = gam_j*(mutij_arr*u_arr + Mtij_j*E_arr)
        S3[:, j] = gam_j*(mutij_arr**2*u_arr + Mtij2_j + Mtij_j*mutij_arr*E_arr)

    # Now M-step updates for each cluster
    alpha_new_part = []
    for j in range(g):
        mu_j_old    = mu_old_list[j]
        shape_j_old = shape_old_list[j]
        Delta_j_old = Delta_old[j]
        Gama_j_old  = Gama_old[j]

        sum_S1 = np.sum(S1[:, j])
        sum_S2 = np.sum(S2[:, j])
        sum_S3 = np.sum(S3[:, j])

        # mu_j_new
        if sum_S1 < 1e-15:
            mu_j_new = mu_j_old
        else:
            sum_S1_X = np.einsum('i,ij->j', S1[:, j], X)
            sum_S2_delta = sum_S2*(Delta_j_old.ravel())
            mu_j_new = (sum_S1_X - sum_S2_delta)/(sum_S1 + 1e-300)

        # Delta_j_new
        dif_new = X - mu_j_new
        sum_S2_dif = np.einsum('i,ij->j', S2[:, j], dif_new)
        Delta_j_new = sum_S2_dif/(sum_S3 + 1e-300)
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
        from numpy.linalg import inv
        inv_sqrt_Sigma_j_new = inv(sqrtm(Sigma_j_new + 1e-12*np.eye(p)))
        numerator = inv_sqrt_Sigma_j_new @ Delta_j_new
        denom_ = 1.0 - (Delta_j_new.T @ inv(Sigma_j_new + 1e-15*np.eye(p)) @ Delta_j_new)
        denom_ = np.sqrt(np.maximum(1e-300, denom_[0,0]))
        shape_j_new = (numerator/denom_).ravel()

        alpha_new_part.append((mu_j_new, Sigma_j_new, shape_j_new, nu_old))

    # Finally, update nu by maximizing the log-likelihood:
    # logvero.SS(nu) = sum(log(d.mixedmvSS(X, pi_prev, mu, Sigma, shape, nu)))
    def neg_loglik_slash(nu_val):
        mu_list    = [x[0] for x in alpha_new_part]
        Sigma_list = [x[1] for x in alpha_new_part]
        lam_list   = [x[2] for x in alpha_new_part]
        mix_pdf = d_mixedmvSS_func(X, pi_prev, mu_list, Sigma_list, lam_list, nu_val)
        mix_pdf = np.maximum(mix_pdf, 1e-300)
        return -np.sum(np.log(mix_pdf))

    res_nu = minimize_scalar(
        neg_loglik_slash, 
        bounds=bounds_nu, 
        method='bounded',
        options={"xatol":1e-6}
    )
    nu_new = res_nu.x

    # Build final alpha_new with the updated nu_new
    alpha_new = []
    for j in range(g):
        mu_j, Sig_j, shp_j, _ = alpha_new_part[j]
        alpha_new.append((mu_j, Sig_j, shp_j, nu_new))

    return alpha_new

def k_means_init_skewslash(data, k):
    """
    K-means initialization for a Skew-Slash mixture model.
    Returns
    -------
    pi_not : (k,) mixing proportions
    alpha_not : list of (mu_j, Sigma_j, shape_j, nu_j=2) or some default 
    """
    data = np.asarray(data)
    n, p = data.shape

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)

    mu_init = kmeans.cluster_centers_  # (k, p)

    # group data
    data_cwise = [[] for _ in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data[i])
    data_cwise = [np.asarray(arr).T for arr in data_cwise]

    pi_init = []
    alpha_init = []
    for j in range(k):
        cluster_data = data_cwise[j]
        size_j = cluster_data.shape[1]
        pi_init.append(size_j/n)

        if size_j <= 1:
            Sigma_j = np.eye(p)
            shape_j = np.zeros(p)
        else:
            Sigma_j = np.cov(cluster_data)
            # shape_j from sign( sum( (x-mu)^3 ) ) or similar
            mean_diff = cluster_data.T - mu_init[j]
            skew_vec  = np.sign(np.sum(mean_diff**3, axis=0))
            shape_j   = skew_vec

        nu_init = 2.0  # some default slash parameter
        alpha_init.append((mu_init[j], Sigma_j, shape_j, nu_init))

    return np.array(pi_init), alpha_init

class SkewSlashMix(BaseMixture):
    """
    Mixture of Multivariate Skew-Slash Distributions.
    A single slash parameter nu is shared across all clusters.
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
        self.family = "skew_slash"

    def _log_pdf_slash(self, X, alphas):
        """
        Evaluate log of Skew-Slash pdf for each row of X, for each cluster j.
        alphas[j] = (mu_j, Sigma_j, shape_j, nu)
        """
        N, p = X.shape
        g = len(alphas)
        logvals = np.empty((N, g))
        for j in range(g):
            mu_j, Sigma_j, shape_j, nu_j = alphas[j]
            dens_j = dmvSS(X, mu_j, Sigma_j, shape_j, nu_j)
            dens_j = np.maximum(dens_j, 1e-300)
            logvals[:, j] = np.log(dens_j)
        return logvals

    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        """
        Weighted log-prob: log(f_j(X)) + log(pi_j).
        """
        log_pdf = self._log_pdf_slash(X, alpha)  # shape (n, g)
        log_pi = np.log(pi)                      # shape (g,)
        return log_pdf + log_pi

    def fit(self, sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (2*self.p + (self.p*(self.p+1)/2) +1)

        # Initialization
        if self.initialization=="kmeans":
            self.pi_not, self.alpha_not = k_means_init_skewslash(self.data, self.k)
        else:
            raise NotImplementedError("Only kmeans initialization is implemented for SkewSlashMix.")

        self.alpha_temp = self.alpha_not
        self.pi_temp    = self.pi_not

        # We define references to pass into M-step:
        def dmvSS_local(data, mu, Sigma, lam, nu):
            return dmvSS(data, mu, Sigma, lam, nu)
        def d_mixedmvSS_local(data, pi_list, mu_list, Sigma_list, lam_list, nu):
            return d_mixedmvSS(data, pi_list, mu_list, Sigma_list, lam_list, nu)

        self.dmvSS_func = dmvSS_local
        self.d_mixedmvSS_func = d_mixedmvSS_local

        # Fit loop from base class
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
            print("Mixtures of Skew-Slash Fitting Done Successfully.")
        end_time = time.time()
        self.execution_time = end_time - start_time

    def _estimate_alphas_wrapper(self, X, gamma, alpha_prev, **kwargs):
        """
        A simple wrapper that calls estimate_alphas_skewslash for one M-step.
        """
        pi_prev = self.pi_temp
        alpha_new = estimate_alphas_skewslash(
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
