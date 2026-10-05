import numpy as np
from numpy.linalg import inv, det
from scipy.stats import norm
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import time
from fmvmm.mixtures._base import BaseMixture
from fmvmm.utils.utils_mixture import (mixture_clusters)
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix, standard_errors_from_info_matrix

def dmvnorm(X, mu, Sigma):
    """
    Multivariate normal density for each row of X.
    X : (n, p)
    mu : (p,) array
    Sigma : (p, p) covariance matrix
    Returns: (n,) array of PDF values
    """
    X = np.asarray(X)
    mu = np.asarray(mu)
    n, p = X.shape

    # Ensure invertibility, add small regularization if needed
    det_Sigma = np.linalg.det(Sigma)
    if det_Sigma < 1e-300:
        # fallback or regularize
        Sigma = Sigma + 1e-6*np.eye(p)
        det_Sigma = np.linalg.det(Sigma)

    inv_Sigma = np.linalg.inv(Sigma)

    diff = X - mu
    quad_form = np.sum(diff @ inv_Sigma * diff, axis=1)  # (n,)

    norm_const = (2.0*np.pi)**(p/2.0) * np.sqrt(det_Sigma)
    pdf_vals = np.exp(-0.5 * quad_form) / norm_const
    return pdf_vals

def dmvSNC(X, mu, Sigma, lam, nu):
    """
    Multivariate Skew Contaminated Normal density for each row in X.
    
    X: (n, p)
    mu: (p,)
    Sigma: (p, p)
    lam: (p,)  -- skewness vector
    nu: (2,)   -- [nu1, nu2], each in (0, 1) typically

    Returns: (n,) array of PDF values
    """
    X = np.asarray(X)
    mu = np.asarray(mu)
    lam = np.asarray(lam)
    n, p = X.shape
    nu1, nu2 = nu[0], nu[1]

    # Precompute Sigma^{-1/2}:
    # Protect against numerical issues if Sigma is near-singular
    det_Sigma = np.linalg.det(Sigma)
    if det_Sigma < 1e-300:
        Sigma = Sigma + 1e-6*np.eye(p)
    sqrt_Sigma = sqrtm(Sigma)
    inv_sqrt_Sigma = np.linalg.inv(sqrt_Sigma)

    # factor = lam^T * inv_sqrt_Sigma  => shape (1, p)
    # We'll apply to (X - mu) row-wise:
    factor = lam @ inv_sqrt_Sigma  # shape (p,) dot (p,p) => shape (p,)
    
    # A1[i] = sqrt(nu2) * factor . (X[i]-mu)
    # A2[i] = factor . (X[i]-mu)
    diff = (X - mu)
    A2 = np.einsum('j,ij->i', factor, diff)     # shape (n,)
    A1 = np.sqrt(nu2)*A2                        # shape (n,)

    # Evaluate the two normal densities:
    # 1) dmvnorm with Sigma/nu2
    dmvnorm_part1 = dmvnorm(X, mu, Sigma/nu2)
    # 2) dmvnorm with Sigma
    dmvnorm_part2 = dmvnorm(X, mu, Sigma)

    # Combine with cdf:
    term1 = nu1 * dmvnorm_part1 * norm.cdf(A1)
    term2 = (1.0 - nu1) * dmvnorm_part2 * norm.cdf(A2)

    dens = np.real_if_close(2.0*(term1 + term2), tol=1000)
    return np.asarray(np.real(dens), dtype=float)

def d_mixedmvSNC(X, pi_list, mu_list, Sigma_list, lam_list, nu):
    """
    Mixture of g Skew-CN components, all sharing the same nu=[nu1, nu2].

    pi_list: (g,) mixing proportions
    mu_list, Sigma_list, lam_list: each is a list of length g
    nu: (2,) array
    Returns: (n,) vector of mixture densities
    """
    g = len(pi_list)
    total = np.zeros(X.shape[0])
    for j in range(g):
        val_j = np.real_if_close(dmvSNC(X, mu_list[j], Sigma_list[j], lam_list[j], nu), tol=1000)
        val_j = np.asarray(np.real(val_j), dtype=float)
        total += pi_list[j]*val_j
    return total

def k_means_init_skewcn(data, k):
    """
    Perform K-means initialization for a Skew Contaminated Normal mixture.
    - data : (n, p) array
    - k    : number of clusters

    Returns
    -------
    pi_init  : 1D array of length k
    alpha_init : list of length k, where
                 alpha_init[j] = (mu_j, Sigma_j, shape_j, nu= [0.5, 0.5])
    """
    data = np.asarray(data)
    n, p = data.shape

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)

    mu_init = kmeans.cluster_centers_  # shape (k, p)

    # Group points by cluster
    data_cwise = [[] for _ in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data[i])
    data_cwise_arr = [np.asarray(arr).T for arr in data_cwise]  # each shape (p, n_j)

    # For shape: sign of third moment (like SkewTMix)
    alpha_init = []
    pi_init = []
    for j in range(k):
        cluster_data = data_cwise_arr[j]  # shape (p, n_j)
        size_j = cluster_data.shape[1]
        pi_init.append(size_j / n)

        if size_j <= 1:
            # degenerate cluster
            Sigma_j = np.eye(p)
            shape_j = np.zeros(p)
        else:
            Sigma_j = np.cov(cluster_data)
            # shape_j from sign( sum( (x - mean)^3 ) ), one approach:
            mean_diff = cluster_data.T - mu_init[j]
            skewness = np.sign(np.sum(mean_diff**3, axis=0))
            shape_j = skewness

        # We'll store (mu_j, Sigma_j, shape_j, nu= [0.5, 0.5]) by default
        alpha_init.append((mu_init[j], Sigma_j, shape_j, np.array([0.5, 0.5])))

    return np.array(pi_init), alpha_init


def estimate_alphas_skewcn(
    X,
    gamma,                   # responsibilities, shape (n, g)
    alpha_prev,              # list of (mu_j, Sigma_j, shape_j, nu_j) 
                             # Here, we assume all clusters share the same nu => the 4th item is the same array for each j
    pi_prev,                 # shape (g,)
    dmvSNC_func, 
    d_mixedmvSNC_func,
    bounds_nu = [(0.01, 0.99), (0.01, 0.99)],
    # You can tune the above bounds if your problem requires different intervals
    **kwargs
):
    """
    Perform one M-step update for a mixture of Skew Contaminated Normal components.
    Follows the logic from the R code (family = "Skew.cn").

    Returns
    -------
    alpha_new : list of length g of (mu_j, Sigma_j, shape_j, nu_new)
    """

    X = np.asarray(X)
    n, p = X.shape
    g = len(alpha_prev)

    # The old shared nu (2D vector)
    nu_old = alpha_prev[0][3]  # shape (2,) e.g. [nu1, nu2]

    # Extract old parameters
    mu_old_list = [alpha_prev[j][0] for j in range(g)]
    Sigma_old_list = [alpha_prev[j][1] for j in range(g)]
    shape_old_list = [alpha_prev[j][2] for j in range(g)]
    # We'll compute Delta_j, Gama_j from shape_j, Sigma_j

    # For numeric stability
    eps = 1e-10

    # 1) Compute d1 = dmvSNC(X, mu_j, Sigma_j, shape_j, nu_old) for each j
    #    mixture pdf d2 = sum_j pi_j * d1_j
    #    tal[i,j] = d1[i,j]* pi_j / d2[i]
    # But we already have gamma=tal from the E-step in the base class. 
    # So we skip recomputing it. 
    # We do, however, need the partial E-step quantities: u[i,j], E[i,j], etc.

    # We define arrays for S1, S2, S3 in each cluster
    S1_mat = np.zeros((n, g))
    S2_mat = np.zeros((n, g))
    S3_mat = np.zeros((n, g))

    # We'll produce "Delta_j_old" and "Gama_j_old" for each cluster 
    # as in the R code:
    Delta_old = []
    Gama_old  = []
    for j in range(g):
        shape_j = shape_old_list[j]
        Sigma_j = Sigma_old_list[j]

        # delta_j = shape_j / sqrt(1 + shape_j' shape_j)
        shape_j = np.asarray(shape_j)
        denom = np.sqrt(1.0 + shape_j @ shape_j)
        delta_j = shape_j / denom

        # Delta_j = sqrtm(Sigma_j) @ delta_j
        sqrt_Sigma_j = sqrtm(Sigma_j + 1e-12*np.eye(p))  # safer
        Delta_j = sqrt_Sigma_j @ delta_j.reshape(-1,1)  # (p,1)
        # Gama_j = Sigma_j - Delta_j Delta_j^T
        Gama_j = Sigma_j - Delta_j @ Delta_j.T

        Delta_old.append(Delta_j)
        Gama_old.append(Gama_j)

    

    for j in range(g):
        mu_j = mu_old_list[j]
        shape_j = shape_old_list[j]
        # We'll compute delta_j, Gama_j, etc from old 
        Delta_j_old = Delta_old[j]
        Gama_j_old  = Gama_old[j]

        # Precompute inv(Gama_j_old)
        inv_Gama_j_old = np.linalg.inv(Gama_j_old + 1e-12*np.eye(p))

        # Compute Mtij2
        tmp_val = (Delta_j_old.T @ inv_Gama_j_old @ Delta_j_old)[0,0]
        Mtij2_j = 1.0/(1.0 + tmp_val)
        Mtij_j  = np.sqrt(Mtij2_j)

        # Compute pdf_j = dmvSNC(X, mu_j, Sigma_j, shape_j, nu_old)
        pdf_j = dmvSNC_func(X, mu_j, Sigma_old_list[j], shape_j, nu_old)
        pdf_j = np.maximum(pdf_j, 1e-300)

        # We'll build A[i], mutij[i]
        Dif = X - mu_j
        mutij_arr = np.zeros(n)
        A_arr = np.zeros(n)

        # factor = Mtij2_j * Delta_j_old^T inv_Gama_j_old
        # shape (1, p) times Dif[i,:] => scalar
        factor_j = (Mtij2_j*(Delta_j_old.T @ inv_Gama_j_old)).ravel()  # shape (p,)

        mutij_arr = np.einsum('j,ij->i', factor_j, Dif)  # shape (n,)
        A_arr = mutij_arr / (Mtij_j + eps)


        nu1, nu2 = nu_old
        dmv_norm1 = dmvnorm(X, mu_j, Sigma_old_list[j]/nu2)
        dmv_norm2 = dmvnorm(X, mu_j, Sigma_old_list[j])

        # For A in the pnorm, see the code: "pnorm( sqrt(nu[2]) * A )" and "pnorm(A)"
        # For E's 'dnorm(...)' the same arguments.
        from scipy.stats import norm

        pnorm_A1 = norm.cdf(np.sqrt(nu2)*A_arr)
        pnorm_A2 = norm.cdf(A_arr)
        dnorm_A1 = norm.pdf(np.sqrt(nu2)*A_arr)
        dnorm_A2 = norm.pdf(A_arr)

        
        u_arr = (2.0/pdf_j)* ( nu1*nu2*dmv_norm1*pnorm_A1 + (1-nu1)*dmv_norm2*pnorm_A2 )
        E_arr = (2.0/pdf_j)* ( nu1*np.sqrt(nu2)*dmv_norm1*dnorm_A1 + (1-nu1)*dmv_norm2*dnorm_A2 )

        # Now S1, S2, S3:
        gamma_j = gamma[:, j]
        S1_j = gamma_j*u_arr
        S2_j = gamma_j*(mutij_arr*u_arr + Mtij_j*E_arr)
        S3_j = gamma_j*(mutij_arr**2*u_arr + Mtij2_j + Mtij_j*mutij_arr*E_arr)

        S1_mat[:, j] = S1_j
        S2_mat[:, j] = S2_j
        S3_mat[:, j] = S3_j

    

    alpha_new_list = []
    for j in range(g):
        # old references
        mu_j_old = mu_old_list[j]
        Delta_j_old = Delta_old[j]
        shape_j_old = shape_old_list[j]
        # sums:
        sum_S1 = np.sum(S1_mat[:, j])
        sum_S2 = np.sum(S2_mat[:, j])
        sum_S3 = np.sum(S3_mat[:, j])

        if sum_S1 < eps:
            # fallback if the cluster is essentially empty
            mu_j_new = mu_j_old
        else:
            # mu_j_new
            sum_S1_X = np.einsum('i,ij->j', S1_mat[:, j], X)
            sum_S2_Delta_old = sum_S2*Delta_j_old.ravel()
            mu_j_new = (sum_S1_X - sum_S2_Delta_old)/(sum_S1 + eps)

        # Then Delta_j_new
        Dif_new = X - mu_j_new
        # sum_i [ S2[i,j]* Dif_new[i] ]
        sum_S2_Dif = np.einsum('i,ij->j', S2_mat[:, j], Dif_new)
        Delta_j_new = sum_S2_Dif/(sum_S3 + eps)
        Delta_j_new = Delta_j_new.reshape(-1,1)

        # Gama_j_new
        sum2 = np.zeros((p,p))
        for i in range(n):
            dif_i = Dif_new[i].reshape(-1,1)
            s1_val = S1_mat[i, j]
            s2_val = S2_mat[i, j]
            s3_val = S3_mat[i, j]
            sum2 += ( s1_val*(dif_i @ dif_i.T)
                       - s2_val*(Delta_j_new @ dif_i.T)
                       - s2_val*(dif_i @ Delta_j_new.T)
                       + s3_val*(Delta_j_new @ Delta_j_new.T) )
        n_j = np.sum(gamma[:, j])
        Gama_j_new = sum2/(n_j + eps)
        Sigma_j_new = Gama_j_new + Delta_j_new @ Delta_j_new.T

        # shape_j_new

        inv_sqrt_Sigma_j_new = np.linalg.inv(sqrtm(Sigma_j_new + 1e-12*np.eye(p)))
        numerator = inv_sqrt_Sigma_j_new @ Delta_j_new
        denom_ = 1.0 - (Delta_j_new.T @ np.linalg.inv(Sigma_j_new + eps*np.eye(p)) @ Delta_j_new)
        denom_ = np.sqrt(np.maximum(1e-300, denom_[0,0]))
        shape_j_new = (numerator/denom_).ravel()

        alpha_new_list.append((mu_j_new, Sigma_j_new, shape_j_new, nu_old))  # temp, we haven't updated nu yet

    # Next: update the shared nu by maximizing sum(log(d_mixedmvSNC(...))).
    # We'll define a function to pass to 'minimize' (L-BFGS-B).
    def neg_loglik_snc(nu_val):
        # nu_val is shape (2,).
        # Build a temp alpha with the same mu, Sigma, shape but new nu
        mu_list  = [a[0] for a in alpha_new_list]
        Sig_list = [a[1] for a in alpha_new_list]
        lam_list = [a[2] for a in alpha_new_list]
        # mixture pdf
        mix_pdf = d_mixedmvSNC_func(X, pi_prev, mu_list, Sig_list, lam_list, nu_val)
        mix_pdf = np.maximum(mix_pdf, 1e-300)
        return -np.sum(np.log(mix_pdf))  # negative for minimization

    # Use scipy.optimize.minimize with L-BFGS-B to update nu
    res = minimize(
        fun=neg_loglik_snc,
        x0=nu_old,  # initial guess
        method='L-BFGS-B',
        bounds=bounds_nu
    )
    nu_new = res.x  # shape(2,)

    # Now finalize alpha_new with the updated nu_new in each cluster
    alpha_final = []
    for j in range(g):
        mu_j, Sig_j, shp_j, _ = alpha_new_list[j]
        alpha_final.append((mu_j, Sig_j, shp_j, nu_new))

    return alpha_final


class SkewContMix(BaseMixture):
    """
    Mixture of Multivariate Skew Contaminated Normal (Skew-CN),
    sharing a single 2D vector nu=[nu1, nu2].
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
        self.family = "skew_cn"

    def _log_pdf_skewcn(self, X, alphas):
        """
        Evaluate log of Skew-CN pdf for each row in X, for each cluster j.

        alphas[j] = (mu_j, Sigma_j, shape_j, nu_j) 
                    but we assume a single nu_j for all j in practice.
        Returns a (n, g) array of log-densities.
        """
        N, p = X.shape
        g = len(alphas)
        log_pdf = np.empty((N, g))
        for j in range(g):
            mu_j, Sigma_j, shape_j, nu_j = alphas[j]
            dens_j = dmvSNC(X, mu_j, Sigma_j, shape_j, nu_j)
            dens_j = np.maximum(dens_j, 1e-300)
            log_pdf[:, j] = np.log(dens_j)
        return log_pdf

    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        """
        Weighted log-prob: log( f_j(X) ) + log(pi_j )
        """
        log_pdf = self._log_pdf_skewcn(X, alpha)   # (n, g)
        log_pi = np.log(pi)                       # (g,)
        return log_pdf + log_pi

    def fit(self, sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (2*self.p + (self.p*(self.p+1)/2) +2)

        # 1) Initialization
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = k_means_init_skewcn(self.data, self.k)
        else:
            raise NotImplementedError("Only 'kmeans' initialization is supported for SkewContMix.")

        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not

        # 2) Provide references for M-step
        #    We'll define small wrappers that link to the above functions.
        def dmvSNC_local(data, mu, Sigma, lam, nu):
            return dmvSNC(data, mu, Sigma, lam, nu)

        def d_mixedmvSNC_local(data, pi_list, mu_list, Sig_list, lam_list, nu):
            return d_mixedmvSNC(data, pi_list, mu_list, Sig_list, lam_list, nu)

        self.dmvSNC_func = dmvSNC_local
        self.d_mixedmvSNC_func = d_mixedmvSNC_local

        # 3) Run the base-class _fit method
        pi_new, alpha_new, log_like_new, log_gamma_new = self._fit(
            self.data,
            self.pi_temp,
            self.alpha_temp,
            estimate_alphas_function=self._estimate_alphas_wrapper
        )

        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.cluster = log_gamma_new.argmax(axis=1)   # cluster assignment
        self.gamma_temp_ar = np.exp(log_gamma_new)
        self.log_likelihood_new = log_like_new

        if self.verbose:
            print("Mixtures of Skew-CN Fitting Done Successfully")
        end_time = time.time()
        self.execution_time = end_time - start_time

    def _estimate_alphas_wrapper(self, X, gamma, alpha_prev, **kwargs):
        """
        Wrapper that calls estimate_alphas_skewcn for a single M-step.
        """
        pi_prev = self.pi_temp  # from the last E-step
        alpha_new = estimate_alphas_skewcn(
            X,
            gamma,
            alpha_prev,
            pi_prev,
            dmvSNC_func=self.dmvSNC_func,
            d_mixedmvSNC_func=self.d_mixedmvSNC_func
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
