from fmvmm.mixtures._base import BaseMixture
from fmvmm.mixtures.MixGHD import mainMGHD, MAPGH, llikGH
import time
import numpy as np
from fmvmm.distributions.multivariate_genhyperbolic import _alphabar2chipsi
from fmvmm.utils.utils_mixture import (mixture_clusters)
from fmvmm.distributions.multivariate_genhyperbolic import logpdf as logpdfmgh
from scipy.optimize import approx_fprime


def _as_scalar(value):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError("Expected a scalar parameter.")
    return float(arr.reshape(-1)[0])


def full_neg_loglike(theta, data, K, p):
    """
    Negative log-likelihood of the entire GH mixture model, 
    for flattened parameter vector theta.

    data: shape (n, p)
    K: number of mixture components
    p: dimension
    """
    pi_list, alpha_list = unflatten_mgh_params(theta, p, K)
    n = data.shape[0]

    total_ll = 0.0
    for i in range(n):
        x_i = data[i]  # shape (p,)
        mixture_val = 0.0
        for k in range(K):
            pi_k = pi_list[k]
            lmbd, chi, psi, mu, Sigma, gvec = alpha_list[k]
            # GH pdf => exp(logpdf)
            pdf_k = np.exp(logpdfmgh(x_i[None], lmbd, chi, psi, mu, Sigma, gvec))[0]
            mixture_val += pi_k * pdf_k
        # add -log(...) to total
        total_ll += -np.log(max(mixture_val, 1e-300))
    return total_ll

def approx_hessian_scipy(fun, x0, epsilon=1e-6):
    """
    Approximate Hessian of a scalar function fun(x) using nested calls to approx_fprime.
    fun: R^d -> scalar
    x0: shape(d,) initial point
    epsilon: step size for finite difference

    Returns: (d, d) array
    """
    d = len(x0)
    H = np.zeros((d, d), dtype=float)

    # baseline gradient at x0
    grad0 = approx_fprime(x0, fun, epsilon)

    for i in range(d):
        # perturb x0 in dimension i
        x_perturb = x0.copy()
        x_perturb[i] += epsilon

        grad_i = approx_fprime(x_perturb, fun, epsilon)
        # row i of Hessian = (grad_i - grad0)/ epsilon
        H[i,:] = (grad_i - grad0) / epsilon

    # symmetrize
    H = 0.5 * (H + H.T)
    return H

def compute_info_scipy(model, epsilon=1e-6):
    """
    Use SciPy to approximate the Hessian of the negative log-likelihood 
    for the GH mixture, then invert to get the empirical info matrix.

    Returns: I_e, Cov, SE
    """
    # Flatten final parameters
    X = model.data
    n, p = X.shape
    K = len(model.pi_new)

    theta_hat = flatten_mgh_params(model.pi_new, model.alpha_new)
    d = len(theta_hat)

    # define partial function
    def nll_wrapper(th):
        return full_neg_loglike(th, X, K, p)

    # compute Hessian
    H = approx_hessian_scipy(nll_wrapper, theta_hat, epsilon)
    # H is the Hessian of the negative log-likelihood => observed info

    # invert
    try:
        Cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Cov = np.linalg.pinv(H)

    diag_cov = np.diag(Cov)
    diag_cov_clipped = np.where(diag_cov>0, diag_cov, 0)
    SE = np.sqrt(diag_cov_clipped)
    SE[SE == 0.0] = 0.0001

    return H, Cov, SE

def flatten_mgh_params(pi_list, alpha_list):
    """
    Flatten mixture parameters into a 1D NumPy array.

    Parameters
    ----------
    pi_list : array-like, shape (K,)
        Mixture weights for each of the K components.
    alpha_list : list of length K
        Each element is [lambda, chi, psi, mu, Sigma, gamma], where:
          - lambda, chi, psi : float (scalars)
          - mu : shape (p,)
          - Sigma : shape (p,p)
          - gamma : shape (p,)

    Returns
    -------
    theta : 1D np.ndarray
        The flattened parameter vector.
    """
    K = len(pi_list)
    final_list = []

    # 1) Append the mixing weights
    #    shape (K,) => K floats
    final_list.extend(pi_list)

    # 2) For each cluster k, append the GH parameters in a fixed order
    for k in range(K):
        lam_k, chi_k, psi_k, mu_k, Sigma_k, gamma_k = alpha_list[k]

        # scalars
        final_list.append(lam_k)
        final_list.append(chi_k)
        final_list.append(psi_k)

        # mu_k shape (p,)
        final_list.extend(mu_k)

        # Sigma_k shape (p,p) => flatten row-wise
        final_list.extend(Sigma_k.flatten())

        # gamma_k shape (p,)
        final_list.extend(gamma_k)

    # Convert everything to a float array
    theta = np.array(final_list, dtype=float)
    return theta


def unflatten_mgh_params(theta, p, K):
    """
    Inverse of flatten_mgh_params.

    Parameters
    ----------
    theta : 1D np.ndarray
        Flattened parameter vector (see flatten_mgh_params).
    p : int
        Dimension of data (mu, gamma in R^p).
    K : int
        Number of mixture components.

    Returns
    -------
    pi_list : np.ndarray, shape (K,)
        Mixture weights.
    alpha_list : list of length K
        alpha_list[k] = [lambda, chi, psi, mu, Sigma, gamma].
    """
    idx = 0

    # 1) Extract mixing proportions
    pi_list = theta[idx : idx + K]
    idx += K

    alpha_list = []

    # 2) Extract each cluster's GH parameters
    for _ in range(K):
        lam_k = theta[idx]
        chi_k = theta[idx + 1]
        psi_k = theta[idx + 2]
        idx += 3

        mu_k = theta[idx : idx + p]
        idx += p

        Sigma_flat = theta[idx : idx + p * p]
        idx += p * p
        Sigma_k = Sigma_flat.reshape((p, p))

        gamma_k = theta[idx : idx + p]
        idx += p

        alpha_list.append([lam_k, chi_k, psi_k, mu_k, Sigma_k, gamma_k])

    return pi_list, alpha_list




def single_loglike_j(theta, x_j, p, K):
    """
    Return log( sum_{k=1 to K} pi_k * GHpdf_k(x_j) ) 
    given the flattened parameter vector theta.

    x_j: shape (p,)
    p: dimension of data
    K: number of mixture components
    """
    # 1) unflatten
    pi_list, alpha_list = unflatten_mgh_params(theta, p, K)

    # 2) compute mixture pdf = sum_k pi_k * GHpdf_k( x_j; alpha_list[k] )
    #    alpha_list[k] => [lambda, chi, psi, mu, Sigma, gamma]
    mixture_val = 0.0
    for k in range(K):
        pi_k = pi_list[k]
        lmbd, chi, psi, mu, Sigma, gvec = alpha_list[k]
        pdf_k = np.exp( logpdfmgh(x_j[None], lmbd, chi, psi, mu, Sigma, gvec) )[0]
        # logpdf returns shape (1,) => pdf is exp of that
        mixture_val += pi_k * pdf_k

    # 3) log of that sum
    # if mixture_val <= 0 => numeric issues => clip
    return np.log(max(mixture_val, 1e-300))




def numeric_gradient_single_obs(x_j, theta, p, K, eps=1e-6):
    """
    Central-difference gradient approximation for single_loglike_j(...) 
    at point 'theta'.
    """
    d = len(theta)
    grad = np.zeros(d)

    for i in range(d):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i]  += eps
        theta_minus[i] -= eps

        f_plus  = single_loglike_j(theta_plus,  x_j, p, K)
        f_minus = single_loglike_j(theta_minus, x_j, p, K)
        grad[i] = (f_plus - f_minus) / (2 * eps)

    return grad




def compute_empirical_info_numeric(model, eps=1e-6):
    X = model.data
    n, p = X.shape
    K = len(model.pi_new)

    pi_list = model.pi_new
    alpha_list = model.alpha_new
    theta_hat = flatten_mgh_params(pi_list, alpha_list)
    d = len(theta_hat)

    I_e = np.zeros((d, d))
    for j in range(n):
        grad_j = numeric_gradient_single_obs(X[j], theta_hat, p, K, eps=eps)
        I_e += np.outer(grad_j, grad_j)

    # Optionally add ridge
    # I_e += 1e-8 * np.eye(d)

    # invert
    try:
        Cov = np.linalg.inv(I_e)
    except np.linalg.LinAlgError:
        Cov = np.linalg.pinv(I_e)
    
    diag_cov = np.diag(Cov)
    diag_cov_clipped = np.where(diag_cov > 0, diag_cov, 0)
    SE = np.sqrt(diag_cov_clipped)

    return I_e, Cov, SE










def extract_cluster_params(mixture_dict):
    """
    Extracts and structures the cluster parameters from the given mixture dictionary.
    
    Parameters:
        mixture_dict (dict): A dictionary containing cluster information and mixing proportions.
    
    Returns:
        list: A list of lists where each sublist corresponds to a cluster and contains:
              [cpl[1], cpl[0], mu, sigma, alpha].
    """
    # Get the number of clusters from the length of 'pi'
    num_clusters = len(mixture_dict.get("pi", []))
    
    # Initialize the result list
    result = []
    
    for i in range(num_clusters):
        if i in mixture_dict:
            cluster = mixture_dict[i]
            # Extract parameters in the required order
            cpl1 = _as_scalar(cluster["cpl"][1])
            cpl0 = _as_scalar(cluster["cpl"][0])
            mu = cluster["mu"].tolist()
            sigma = cluster["sigma"]
            alpha = cluster["alpha"].tolist()
            chi, psi = _alphabar2chipsi(cpl0, cpl1)
            result.append([cpl1, chi, psi, mu, sigma, alpha])
    
    return result






class MixMGH(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True):
        super().__init__(n_clusters=n_clusters, EM_type="Soft", mixture_type="identical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.initialization=initialization
        self.family = "mgh"
    def fit(self, sample):
        """
        Fit the mixture of generalized hyperbolic distributions model.
        
        Parameters:
            X (array-like): data matrix with shape (n_samples, n_features)
            gpar0 (dict, optional): initial parameter dictionary for the GH mixture.
            n_iter (int, optional): maximum number of iterations for the EM algorithm.
            label (array-like, optional): known cluster labels for semi-supervised learning.
        
        Returns:
            self: the fitted model.
        """
        start_time = time.time()
        self.data=self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (2*self.p + (self.p*(self.p+1)/2) +3)
        # Call the mainMGHD function from MixGHD.py to perform the EM algorithm.
        self.results = mainMGHD(data=self.data, G=self.n_clusters, n= self.max_iter,
                                 eps=self.tol)
        self.log_likelihoods = self.results.get("loglik", [])
        self.fitted = True
        self.pi_new = self.results["gpar"]["pi"]
        self.alpha_new = extract_cluster_params(self.results["gpar"])
        self.cluster = self.results["map"]
        self.gamma_temp_ar = self.results["z"]
        self.log_likelihood_new = self.log_likelihoods[-1]
        if self.verbose:
            print("Mixtures of Generalized Hyperbolic Fitting Done Successfully")
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
    
    def get_info_mat(self, eps=1e-5):
        """
        Compute numeric empirical info matrix & SEs for the final fitted mixture.
        """
        print("Warning: This is experimental; Using Numerical Differentiation")
        # I_e, Cov, SE = compute_empirical_info_numeric(self, eps=eps)
        I_e, Cov, SE = compute_info_scipy(self, epsilon=eps)
        
        return I_e, SE
