import numpy as np
from fmvmm.mixtures._base import BaseMixture
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST_custom
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm, inv
import time
from scipy.optimize import minimize_scalar
from fmvmm.utils.utils_mixture import (mixture_clusters)
from scipy.special import gamma
from scipy.stats import t
from numpy.linalg import det
from fmvmm.mixsmsn.information_matrix_smsn import info_matrix, standard_errors_from_info_matrix

def k_means_init_skewt(data, k):
    """
    Perform K-means initialization for a skew-t mixture model.

    Parameters:
    - data: pandas.DataFrame or np.ndarray of shape (n_samples, n_features).
    - k: Number of clusters.

    Returns:
    - pi_not: Mixing proportions for each cluster.
    - mu_not: Cluster means.
    - cov_not: Covariance matrices for each cluster.
    - shape_not: Shape (skewness) parameters for each cluster.
    - delta: Normalized skewness vectors for each cluster.
    - Delta: Skewness-adjusted covariance components for each cluster.
    - Gama: Residual covariance matrices for each cluster.
    """
    # Ensure data is a NumPy array
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()

    n, p = data.shape

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)

    # Initialize cluster means
    mu_not = kmeans.cluster_centers_

    # Group data points by cluster
    data_cwise = [[] for _ in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data[i])
    
    # Convert grouped data to NumPy arrays
    data_cwise_arr = [np.asarray(cluster_data).T for cluster_data in data_cwise]

    # Initialize covariance matrices and skewness parameters
    alpha_not = []

    for j, cluster_data in enumerate(data_cwise_arr):
        # Covariance matrix
        cov_not_temp = np.cov(cluster_data)
        
        # Shape (skewness) parameter
        if cluster_data.shape[1] > 0:  # Ensure the cluster is not empty
            mean_diff = cluster_data.T - mu_not[j]
            skewness = np.sign(np.sum(mean_diff**3, axis=0))
    
        else:
            skewness = np.zeros(p)
        alpha_not.append((mu_not[j], cov_not_temp, skewness, 4))

    # Mixing proportions
    pi_not = [len(data_cwise[m]) / n for m in range(k)]

    return pi_not, alpha_not

def gmm_init_skewt(data, k):
    """
    Perform Gaussian Mixture Model (GMM) initialization for a skew-t mixture model.

    Parameters:
    - data: pandas.DataFrame or np.ndarray of shape (n_samples, n_features).
    - k: Number of clusters.

    Returns:
    - pi_not: Mixing proportions for each cluster.
    - mu_not: Cluster means.
    - cov_not: Covariance matrices for each cluster.
    - shape_not: Shape (skewness) parameters for each cluster.
    - delta: Normalized skewness vectors for each cluster.
    - Delta: Skewness-adjusted covariance components for each cluster.
    - Gama: Residual covariance matrices for each cluster.
    """
    # Ensure data is a NumPy array
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()

    n, p = data.shape

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=k, random_state=1)
    gmm.fit(data)

    # Extract GMM parameters
    mu_not = gmm.means_  # Cluster means
    covariances = gmm.covariances_  # Covariance matrices
    if gmm.covariance_type == 'full':
        cov_not = covariances
    elif gmm.covariance_type == 'diag':
        cov_not = [np.diag(cov) for cov in covariances]
    elif gmm.covariance_type == 'tied':
        cov_not = [covariances for _ in range(k)]
    elif gmm.covariance_type == 'spherical':
        cov_not = [np.eye(p) * cov for cov in covariances]
    else:
        raise ValueError("Unsupported covariance type.")

    # Mixing proportions
    pi_not = gmm.weights_

    # Group data points by cluster
    labels = gmm.predict(data)
    data_cwise = [[] for _ in range(k)]
    for i in range(n):
        data_cwise[labels[i]].append(data[i])

    # Convert grouped data to NumPy arrays
    data_cwise_arr = [np.asarray(cluster_data).T for cluster_data in data_cwise]

    # Initialize skewness parameters and related components
    alpha_not = []
    shape_not = []
    delta = []
    Delta = []
    Gama = []

    for j, cluster_data in enumerate(data_cwise_arr):
        if cluster_data.shape[1] > 0:  # Ensure the cluster is not empty
            mean_diff = cluster_data.T - mu_not[j]
            skewness = np.sign(np.sum(mean_diff**3, axis=0))
            # shape_not.append(skewness)

            # Calculate delta, Delta, and Gama
            delta_j = skewness / np.sqrt(1 + skewness @ skewness)
            Delta_j = sqrtm(cov_not[j]) @ delta_j
            Gama_j = cov_not[j] - np.outer(Delta_j, Delta_j)

            delta.append(delta_j)
            Delta.append(Delta_j)
            Gama.append(Gama_j)
        else:
            # shape_not.append(np.zeros(p))
            skewness = np.zeros(p)
            delta.append(np.zeros(p))
            Delta.append(np.zeros(p))
            Gama.append(np.zeros((p, p)))
        alpha_not.append((mu_not[j], cov_not[j],skewness, 4))

    return pi_not, alpha_not


def estimate_alphas_skewt(
    X, gamma, alpha_prev, pi_prev,  # from the Soft EM (M-step)
    dmvt_ls_func = dmvt_ls, d_mixedmvST_func = d_mixedmvST_custom,
    bounds_nu=(1e-4, 100.0),
    tol_nu=1e-6
):
    """
    Single M-step update for a mixture of multivariate Skew-t components,
    following the logic from the R function smsn.mmix (Skew.t part).

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Data matrix.
    gamma : ndarray of shape (n, g)
        Posterior responsibilities (soft cluster assignments).
    alpha_prev : list of length g
        Each element alpha_prev[j] = (mu_j, Sigma_j, shape_j, nu_j).
        We assume all clusters share the same nu, so alpha_prev[j][3] should be identical.
    pi_prev : 1D array of length g
        Mixing proportions from the previous iteration.
    dmvt_ls_func : callable
        Function dmvt_ls_func(X, mu, Sigma, shape, nu) -> PDF values (length n).
    d_mixedmvST_func : callable
        Function that computes the mixture PDF: sum_j pi_j * dmvt_ls(X,...).
    bounds_nu : tuple
        Bounds for optimizing nu in [low, high].
    tol_nu : float
        Tolerance for the nu optimizer.

    Returns
    -------
    alpha_new : list of length g
        Updated parameters for each cluster j:
        alpha_new[j] = (mu_j_new, Sigma_j_new, shape_j_new, nu_new).
        The same nu_new is placed in each cluster's 4th element.
    """

    n, p = X.shape
    g = len(alpha_prev)

    
    nu_prev = alpha_prev[0][3]

    # We will store intermediate items and final results here
    alpha_new = [None]*g

    # Precompute each cluster's old Delta, Gama, etc.
    # We do that so we can use "Delta_old" in the new mu update.
    Delta_old = []
    Gama_old = []
    for j in range(g):
        mu_j, Sigma_j, shape_j, _ = alpha_prev[j]
        # delta_j = shape_j / sqrt(1 + shape_j' shape_j)
        shape_j = np.asarray(shape_j).reshape(-1)  # ensure 1D
        denom = np.sqrt(1.0 + shape_j @ shape_j)
        delta_j = shape_j / denom  # p-vector

        # Delta_j = sqrtm(Sigma_j) @ delta_j
        Delta_j = sqrtm(Sigma_j) @ delta_j.reshape(-1,1)  # shape (p,1)

        # Gama_j = Sigma_j - Delta_j %*% t(Delta_j)
        Gama_j = Sigma_j - Delta_j@Delta_j.T

        Delta_old.append(Delta_j)
        Gama_old.append(Gama_j)

    

    def compute_E_u(X_i, mu_j, Sigma_j, Delta_j, Gama_j, shape_j, nu_val):
        """
        Compute the E[i], u[i] terms for a single point X_i in cluster j
        following the R code formula.
        """
        p = X_i.shape[0]
        dif_i = X_i - mu_j
        # For convenience, invert Gama_j once outside the loop in the M-step
        # but here we do a single case (we can pass a precomputed invGama_j if we want).
        return 0.0, 0.0  # Will fill in in the loop below

    alpha_new_list = []
    
    # Precompute cluster pdfs for each j to avoid repeated calls
    # We'll do it in a vectorized manner.
    cluster_pdf = np.zeros((n, g))
    for j in range(g):
        mu_j, Sigma_j, shape_j, nu_j = alpha_prev[j]
        cluster_pdf[:, j] = dmvt_ls_func(X, mu_j, Sigma_j, shape_j, nu_prev)

    # Mixture PDF for each i
    d_mixed = np.sum(pi_prev * cluster_pdf, axis=1)
    # Protect against zeros
    d_mixed[d_mixed < 1e-300] = 1e-300

    # Precompute inverse Sigma_j, inverse Gama_j
    inv_Sigma = []
    inv_Gama_old = []
    for j in range(g):
        _, Sigma_j, _, _ = alpha_prev[j]
        inv_Sigma.append(inv(Sigma_j))
        inv_Gama_old.append(inv(Gama_old[j]))

    # We will store updated parameters:
    mu_new = []
    Sigma_new = []
    shape_new = []
    

    for j in range(g):
        # alpha old
        mu_j, Sigma_j, shape_j, _ = alpha_prev[j]
        Delta_j_old = Delta_old[j]
        Gama_j_old = Gama_old[j]

        invGama_j_old = inv_Gama_old[j]
        invSigma_j = inv_Sigma[j]

        # 1) Compute constants: Mtij^2, Mtij
        delta_j_old = shape_j / np.sqrt(1.0 + shape_j @ shape_j)
        Mtij2 = 1.0 / (1.0 + (delta_j_old.reshape(1,-1) @ invGama_j_old @ delta_j_old.reshape(-1,1))[0,0])
        Mtij = np.sqrt(Mtij2)

        # We'll build arrays for E[i], u[i], mutij[i], A[i], etc.
        E_arr = np.zeros(n)
        u_arr = np.zeros(n)
        mutij_arr = np.zeros(n)
        dj_arr = np.zeros(n)

        
        sqrt_det_Sigma_j = np.sqrt(np.maximum(1e-300, np.linalg.det(Sigma_j)))
        
        from math import gamma as gamma_func, pi, sqrt
        from scipy.stats import t as student_t

        

        cE_num = 2.0*(nu_prev**(nu_prev/2.0))*gamma_func((p+nu_prev+1)/2.0)
        cE_den = gamma_func(nu_prev/2.0)*(pi**((p+1)/2.0))*sqrt_det_Sigma_j
        cE = cE_num / (cE_den + 1e-300)

        cU_num = 4.0*(nu_prev**(nu_prev/2.0))*gamma_func((p+nu_prev+2)/2.0)
        cU_den = gamma_func(nu_prev/2.0)* (pi**(p/2.0))*sqrt_det_Sigma_j
        cU = cU_num / (cU_den + 1e-300)

        for i in range(n):
            dif_i = X[i] - mu_j
            dj_val = dif_i @ invSigma_j @ dif_i  # scalar
            dj_arr[i] = dj_val

            mutij_val = Mtij2 * (delta_j_old @ invGama_j_old @ dif_i)
            mutij_arr[i] = mutij_val
            A_val = mutij_val / Mtij

            # pdf_j(i) = cluster_pdf[i, j]
            pdf_j_i = cluster_pdf[i, j]
            if pdf_j_i < 1e-300:
                # avoid zero
                pdf_j_i = 1e-300

            

            E_num = cE*(dj_val + nu_prev + A_val**2)**(-(p+nu_prev+1)/2.0)
            E_arr[i] = E_num / pdf_j_i  # denominator

            
            u_num = cU*(dj_val + nu_prev)**(-(p+nu_prev+2)/2.0)
            
            z_val = np.sqrt((p + nu_prev + 2)/(dj_val + nu_prev)) * A_val
            cdf_val = student_t.cdf(z_val, df=(p + nu_prev + 2))
            u_arr[i] = u_num*cdf_val / (pdf_j_i + 1e-300)

        # Now S1, S2, S3, etc.
        
        S1 = gamma[:, j]*u_arr
        
        S2 = gamma[:, j]*( mutij_arr*u_arr + Mtij*E_arr )
        
        S3 = gamma[:, j]*( mutij_arr**2*u_arr + Mtij2 + Mtij*mutij_arr*E_arr )

        # 2) Update pi_j inside _m_step . 

        # 3) M-step updates:
        
        sum_S1 = np.sum(S1)
        sum_S2 = np.sum(S2)
        sum_S1_X = np.einsum('i,ij->j', S1, X)  # weighted sum of X
        sum_S2_delta = sum_S2*Delta_j_old.ravel()  # same for Delta_j_old

        mu_j_new = (sum_S1_X - sum_S2_delta) / (sum_S1 + 1e-300)

        # Then re-define Dif with new mu_j:
        Dif = X - mu_j_new
        
        sum_S2_Dif = np.einsum('i,ij->j', S2, Dif)
        sum_S3 = np.sum(S3) 
        Delta_j_new = sum_S2_Dif / (sum_S3 + 1e-300)
        Delta_j_new = Delta_j_new.reshape(-1,1)  # make (p,1)

        
        n_j = np.sum(gamma[:, j])

        sum2 = np.zeros((p, p))
        for i in range(n):
            # Dif_i = X[i] - mu_j_new
            dif_i = Dif[i].reshape(-1,1)
            s1_val = gamma[i,j]*u_arr[i]
            s2_val = gamma[i,j]*(mutij_arr[i]*u_arr[i] + Mtij*E_arr[i])  # same as S2[i]
            s3_val = gamma[i,j]*(mutij_arr[i]**2*u_arr[i] + Mtij2 + Mtij*mutij_arr[i]*E_arr[i])  # S3[i]
            # Accumulate
            sum2 += ( s1_val*dif_i@dif_i.T
                       - s2_val*Delta_j_new@dif_i.T
                       - s2_val*dif_i@Delta_j_new.T
                       + s3_val*(Delta_j_new@Delta_j_new.T) )

        Gama_j_new = sum2 / (n_j + 1e-300)
        Sigma_j_new = Gama_j_new + Delta_j_new@Delta_j_new.T

        
        inv_sqrt_Sigma_j_new = inv(sqrtm(Sigma_j_new + 1e-10*np.eye(p)))
        numerator = inv_sqrt_Sigma_j_new @ Delta_j_new
        denom_ = 1.0 - (Delta_j_new.T @ inv(Sigma_j_new) @ Delta_j_new)
        denom_ = np.sqrt(np.maximum(1e-300, denom_[0,0]))
        shape_j_new = (numerator / denom_).ravel()  # make 1D

        mu_new.append(mu_j_new)
        Sigma_new.append(Sigma_j_new)
        shape_new.append(shape_j_new)



    def loglik_skewt_nu(nu_val):
        
        alpha_temp = []
        for j in range(g):
            alpha_temp.append((mu_new[j], Sigma_new[j], shape_new[j], nu_val))

        mix_pdf = d_mixedmvST_func(X, pi_prev, alpha_temp)
        mix_pdf = np.maximum(mix_pdf, 1e-300)
        return np.sum(np.log(mix_pdf))

    # Optimize nu in the given bounds
    
    res = minimize_scalar(
        lambda x: -loglik_skewt_nu(x),
        bounds=bounds_nu,
        method='bounded',
        options={"xatol": tol_nu}
    )
    nu_new = res.x

    # Place final alpha_new in a list of (mu, Sigma, shape, nu)
    alpha_new = []
    for j in range(g):
        alpha_new.append((mu_new[j], Sigma_new[j], shape_new[j], nu_new))

    return alpha_new




class SkewTMix(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True):
        super().__init__(n_clusters=n_clusters, EM_type="Soft", mixture_type="identical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.initialization=initialization
        self.family = "skew_t"
        
    def _log_pdf_skewt(self,X,alphas):
        N,p=X.shape
        k=len(alphas)
        probs=np.empty((N, k))
        for j in range(k):
            alpha=alphas[j]
            with np.errstate(under="ignore"):
                probs[:, j] = np.log(dmvt_ls(X,*alpha))
        
        return probs
    
    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        return self._log_pdf_skewt(X,alpha) + np.log(pi)
    
    def fit(self,sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (2*self.p + (self.p*(self.p+1)/2) +1)
        np.random.seed(0)
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = k_means_init_skewt(self.data, self.k)
        elif self.initialization == "gmm":
            self.pi_not, self.alpha_not = gmm_init_skewt(self.data, self.k)
        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not
        pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(self.data,self.pi_temp,self.alpha_temp,estimate_alphas_skewt,pi_prev = self.pi_temp)
        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.cluster = log_gamma_new.argmax(axis=1)
        self.gamma_temp_ar = np.exp(log_gamma_new)
        self.log_likelihood_new = log_likelihood_new
        if self.verbose:
            print("Mixtures of Skew-t Fitting Done Successfully")
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


    def responsibilities(self):

        return self.gamma_temp_ar

    def n_iter(self):
        return len(self.log_likelihoods)

    def get_info_mat(self):
        
        im = info_matrix(self)
        se = standard_errors_from_info_matrix(im)
        
        return im, se
        