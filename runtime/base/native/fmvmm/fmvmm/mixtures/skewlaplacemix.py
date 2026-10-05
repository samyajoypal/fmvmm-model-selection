"""
some matlab codes were received personally from Prof.Fatma Zehra Doğru,
author of Finite Mixtures of Multivariate Skew Laplace Distributions.
Many thanks to her. Rest was incorporated from the paper itself.

"""

import numpy as np
from fmvmm.mixtures._base import BaseMixture
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time
from fmvmm.utils.utils_mixture import (mixture_clusters)
from numpy.linalg import cholesky
import math
from scipy.special import gamma as gamma_func

###############################################################################
# (A) Generate random samples from a SINGLE Multivariate Skew-Laplace
###############################################################################

def sample_skewlaplace(mu, Sigma, gamma,n, random_state=None):
    """
    Generate 'n' samples in R^p from a single p-variate Skew-Laplace distribution.

    The parameterization matches ``compute_log_pdf_skewlaplace``:
      1) W_i ~ Gamma((p+1)/2, scale=2)
      2) Z_i ~ Normal(0, I_p)
      3) X_i = mu + W_i * gamma + sqrt(W_i) * L Z_i,
         where L L^T = Sigma.

    Parameters
    ----------
    n : int
        Number of samples
    mu : (p,) array
        Location vector
    Sigma : (p,p) array
        Positive-definite scatter matrix
    gamma : (p,) array
        Skewness vector
    random_state : int or None
        If set, seed for reproducibility.

    Returns
    -------
    X : array, shape (n, p)
    """
    # Convert mu, gamma to NumPy arrays (if not already)
    mu = np.asarray(mu, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    rng = np.random.default_rng(random_state)
    p = len(mu)

    w_samples = rng.gamma(shape=(p+1)/2.0, scale=2.0, size=n)

    # 2) z_i ~ Normal(0, I_p)
    z_samples = rng.normal(size=(n, p))

    # 3) L = chol(Sigma)
    L = cholesky(Sigma)

    return (
        mu
        + w_samples[:, None] * gamma
        + np.sqrt(w_samples)[:, None] * (z_samples @ L.T)
    )


def _regularize_cov(cov, eps=1e-8):
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = float(np.min(eigvals))
        if min_eig < eps:
            cov = cov + (eps - min_eig) * np.eye(cov.shape[0])
    except np.linalg.LinAlgError:
        cov = cov + eps * np.eye(cov.shape[0])
    return cov


def compute_empirical_info_and_se(model, pi = None, alpha = None, data = None, gamma_res = None, use_model = True):
    """
    Given a fitted SkewLaplaceMix model, compute:
      1) Empirical information matrix I_e = sum_j s_j s_j^T,
      2) Covariance = (I_e)^{-1},
      3) Standard errors = sqrt(diag(Cov)).

    Returns
    -------
    SE : (d,) array
        Standard errors for each parameter in the vector of parameters
        (in the same order we construct the score).
    Cov : (d, d) array
        Approx. covariance matrix = I_e^{-1}.
    """
    # 1) Extract final parameters from the model
    #    pi_k: shape (K,)
    #    alpha_est: list of dicts or list-lists: alpha_est[k] = [mu_k, Sigma_k, gamma_k]
    if use_model:
        pi_est = model.pi_new
    else:
        pi_est = pi
    if use_model:
        alpha_est = model.alpha_new  # length K
    else:
        alpha_est = alpha



    K = len(pi_est)
    # alpha_est[k][0] => mu_k
    # alpha_est[k][1] => Sigma_k
    # alpha_est[k][2] => gamma_k
    p = alpha_est[0][0].shape[0]

    # 2) Data & final responsibilities
    if use_model:
        X = model.data
    else:
        X = data
    n = X.shape[0]
    # responsibilities => shape (n, K)
    if use_model:
        Z = model.gamma_temp_ar  # e.g. exp(log_gamma_new)
    else:
        Z = gamma_res

    # 3) Compute v1_{i,k}, v2_{i,k}
    V1 = np.zeros((n, K))
    V2 = np.zeros((n, K))

    inv_Sigmas = []
    a_values = []
    for k_i in range(K):
        Sigma_k = alpha_est[k_i][1]
        inv_Sig = np.linalg.inv(Sigma_k)
        inv_Sigmas.append(inv_Sig)

        g_k = alpha_est[k_i][2]
        a_k = np.sqrt(1.0 + g_k @ inv_Sig @ g_k)
        a_values.append(a_k)

    for i in range(n):
        x_i = X[i]
        for k_i in range(K):
            mu_k = alpha_est[k_i][0]
            g_k  = alpha_est[k_i][2]
            invS_k = inv_Sigmas[k_i]
            a_k = a_values[k_i]

            diff = x_i - mu_k
            dist_sq = diff @ invS_k @ diff
            dist = np.sqrt(max(dist_sq, 1e-14))

            # v1_{i,k} = a_k / dist
            # v2_{i,k} = [1 + sqrt(a_k^2 * dist^2)] / a_k^2
            V1[i, k_i] = a_k / dist
            V2[i, k_i] = (1.0 + np.sqrt(a_k*a_k * dist_sq)) / (a_k*a_k)

    # 4) Score function for a single observation
    def score_single_observation(x, z_row, v1_row, v2_row):
        """
        Returns the score vector s_j for one observation j,
        given z_{j,k}, v1_{j,k}, v2_{j,k}, pi_k, alpha[k].
        Order:
          - (K-1) partials wrt pi_0.. pi_{K-2},
          - For each k: partial wrt mu_k, Sigma_k, gamma_k
        """
        score_list = []

        # (A) partial wrt pi_k => eqn(4.2)
        # Each is a scalar => wrap in np.array([ ... ])
        for r in range(K-1):
            val = z_row[r]/pi_est[r] - z_row[K-1]/pi_est[K-1]
            score_list.append(np.array([val]))  # ensure shape (1,)

        # (B) partial wrt mu_k, Sigma_k, gamma_k
        for k_i in range(K):
            mu_k = alpha_est[k_i][0]
            Sigma_k = alpha_est[k_i][1]
            g_k = alpha_est[k_i][2]
            invS_k = inv_Sigmas[k_i]

            diff = x - mu_k

            # # eqn(4.3) => partial wrt mu_k => shape (p,)
            # s_mu_k = (
            #     z_row[k_i]*v1_row[k_i]*(invS_k @ diff)
            #     - z_row[k_i]*v2_row[k_i]*(invS_k @ g_k)
            # )

            # mu
            s_mu_k = z_row[k_i] * (
                v1_row[k_i]*(invS_k @ diff)
                - (invS_k @ g_k)
            )

            # eqn(4.4) => partial wrt Sigma_k => shape (p(p+1)/2,)
            GkGkT = np.outer(g_k, g_k)
            outer_diff = np.outer(diff, diff)

            A = (
                invS_k
                - v1_row[k_i] * (invS_k @ outer_diff @ invS_k)
                - v2_row[k_i] * (invS_k @ GkGkT   @ invS_k)
            )



            A_diag = np.diag(np.diag(A))          # shape (p,p) with only diagonal of A
            A_corrected = -A + 0.5 * A_diag       # the bracket in eqn. (4.4)
            sSigma_full = z_row[k_i] * A_corrected

            # Flatten with vech-lower
            sSigma_k_list = []
            for ii in range(p):
                for jj in range(ii, p):
                    sSigma_k_list.append(sSigma_full[ii, jj])
            sSigma_k = np.array(sSigma_k_list)

            # # eqn(4.5) => partial wrt gamma_k => shape (p,)
            # s_gamma_k = (
            #     z_row[k_i]*v1_row[k_i]*(invS_k @ diff)
            #     - z_row[k_i]*v2_row[k_i]*(invS_k @ g_k)
            # )
            s_gamma_k = z_row[k_i] * (
            (invS_k @ diff)
            - v2_row[k_i]*(invS_k @ g_k)
        )

            score_list.append(s_mu_k)
            score_list.append(sSigma_k)
            score_list.append(s_gamma_k)

        # Now everything in score_list is a 1D array => we can safely concatenate
        return np.concatenate(score_list)

    # 5) Sum up s_j s_j^T over j=1..n
    test_s = score_single_observation(X[0], Z[0], V1[0], V2[0])
    d = len(test_s)

    I_e = np.zeros((d, d))
    for j in range(n):
        s_j = score_single_observation(X[j], Z[j], V1[j], V2[j])
        I_e += np.outer(s_j, s_j)

    # 6) Cov = I_e^{-1}, SE = sqrt(diag(Cov))
    # eigenvalues = np.linalg.eigvalsh(I_e)
    # if np.any(eigenvalues <= 0):
    #     _reg = 1e-6  # Small regularization term
    #     I_e = I_e + _reg * np.eye(I_e.shape[0])
    Cov = np.linalg.pinv(I_e)
    SE = np.sqrt(np.diag(Cov))

    return I_e, SE

def compute_empirical_info_se_and_scores(
    model,
    pi=None,
    alpha=None,
    data=None,
    gamma_res=None,
    use_model=True,
    return_cov=True,
):
    """
    Same logic as compute_empirical_info_and_se(...), but ALSO returns
    the per-observation score matrix S (n, d), so you can build OPG,
    sandwich estimators, etc.

    Returns
    -------
    I_e : (d,d)
    SE  : (d,)
    S   : (n,d)
    Cov : (d,d)   (only if return_cov=True)
    """

    # ---------- 1) extract params ----------
    if use_model:
        pi_est = model.pi_new
        alpha_est = model.alpha_new
        X = model.data
        Z = model.gamma_temp_ar
    else:
        pi_est = pi
        alpha_est = alpha
        X = data
        Z = gamma_res

    K = len(pi_est)
    n = X.shape[0]
    p = alpha_est[0][0].shape[0]

    # ---------- 2) precompute inv(Sigma_k) and a_k ----------
    inv_Sigmas = []
    a_values = []
    for k in range(K):
        Sigma_k = alpha_est[k][1]
        invS = np.linalg.inv(Sigma_k)
        inv_Sigmas.append(invS)

        g_k = alpha_est[k][2]
        a_k = np.sqrt(1.0 + g_k @ invS @ g_k)
        a_values.append(a_k)

    # ---------- 3) compute V1,V2 ----------
    # V1[i,k] = a_k / dist
    # V2[i,k] = (1 + sqrt(a_k^2 * dist^2))/a_k^2
    V1 = np.zeros((n, K), float)
    V2 = np.zeros((n, K), float)

    for i in range(n):
        x_i = X[i]
        for k in range(K):
            mu_k = alpha_est[k][0]
            invS = inv_Sigmas[k]
            a_k = a_values[k]

            diff = x_i - mu_k
            dist_sq = float(diff @ invS @ diff)
            dist_sq = max(dist_sq, 1e-14)
            dist = np.sqrt(dist_sq)

            V1[i, k] = a_k / dist
            V2[i, k] = (1.0 + np.sqrt(a_k * a_k * dist_sq)) / (a_k * a_k)

    # ---------- 4) score for a single observation ----------
    def score_single_observation(x, z_row, v1_row, v2_row):
        score_list = []

        # (A) pi block (K-1) as in your existing code
        for r in range(K - 1):
            val = z_row[r] / pi_est[r] - z_row[K - 1] / pi_est[K - 1]
            score_list.append(np.array([val], float))

        # (B) component blocks
        for k in range(K):
            mu_k = alpha_est[k][0]
            g_k = alpha_est[k][2]
            invS = inv_Sigmas[k]

            diff = x - mu_k

            # mu-score (p,)
            s_mu = z_row[k] * (v1_row[k] * (invS @ diff) - (invS @ g_k))

            # Sigma-score (vech upper) (p(p+1)/2,)
            GGT = np.outer(g_k, g_k)
            outer_diff = np.outer(diff, diff)

            A = (
                invS
                - v1_row[k] * (invS @ outer_diff @ invS)
                - v2_row[k] * (invS @ GGT @ invS)
            )
            A_diag = np.diag(np.diag(A))
            A_corrected = -A + 0.5 * A_diag
            sSigma_full = z_row[k] * A_corrected

            sSigma = []
            for ii in range(p):
                for jj in range(ii, p):
                    sSigma.append(sSigma_full[ii, jj])
            sSigma = np.array(sSigma, float)

            # gamma-score (p,)
            s_gamma = z_row[k] * ((invS @ diff) - v2_row[k] * (invS @ g_k))

            score_list.append(s_mu)
            score_list.append(sSigma)
            score_list.append(s_gamma)

        return np.concatenate(score_list)

    # ---------- 5) build S and OPG ----------
    s0 = score_single_observation(X[0], Z[0], V1[0], V2[0])
    d = s0.size

    S = np.zeros((n, d), float)
    for i in range(n):
        S[i, :] = score_single_observation(X[i], Z[i], V1[i], V2[i])

    I_e = S.T @ S

    # numerical safety (same spirit as your pinv)
    Cov = np.linalg.pinv(I_e)
    SE = np.sqrt(np.maximum(np.diag(Cov), 0.0))

    if return_cov:
        return I_e, SE, S, Cov
    return I_e, SE, S


def k_means_init_skewlaplace(X, k):
    """
    1) Run KMeans to get cluster assignments.
    2) Compute each cluster's mean, covariance.
    3) gamma_k = 0 (shape vector).
    4) pi_k = #points in cluster / n.
    """
    n, p = X.shape

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_  # shape (k, p)

    # pi array
    pi_init = np.zeros(k)
    alpha_init = []
    for cluster_id in range(k):
        # collect points in that cluster
        idx = np.where(labels == cluster_id)[0]
        pi_init[cluster_id] = len(idx) / n

        # mean = cluster center
        mu_k = centers[cluster_id]

        # covariance => empirical from data in cluster
        if len(idx) > 1:
            cov_k = np.cov(X[idx].T, bias=False)  # shape (p,p)
        else:
            # if only one point, fallback to identity or very small
            cov_k = np.eye(p)

        # gamma_k => np.zeros(p)
        gamma_k = np.zeros(p)

        alpha_init.append((mu_k, cov_k, gamma_k))

    return pi_init, alpha_init

def gmm_init_skewlaplace(X, k):
    """
    1) Run GaussianMixture to get cluster assignments.
    2) Extract means, covariances, weights.
    3) gamma_k = 0.
    4) pi_k from GMM.
    """
    n, p = X.shape

    gm = GaussianMixture(n_components=k, random_state=0).fit(X)
    pi_init = gm.weights_  # shape (k,)
    centers = gm.means_    # shape (k, p)
    covariances = gm.covariances_  # shape (k, p, p)

    alpha_init = []
    for cluster_id in range(k):
        mu_k = centers[cluster_id]
        cov_k = covariances[cluster_id]
        gamma_k = np.zeros(p)
        alpha_init.append(( mu_k,cov_k, gamma_k))

    return pi_init, alpha_init

def estimate_alphas_skewlaplace(X, gamma_matrix, alpha_old):
    """
    M-step update for a K-component Multivariate Skew-Laplace mixture,
    using the same formulas as your 2-component MATLAB code but generalized.

    Parameters
    ----------
    X : array, shape (n, p)
        Data matrix.
    gamma_matrix : array, shape (n, K)
        Responsibilities (E-step), gamma_matrix[i, k] = posterior probability that
        sample i belongs to cluster k.
    alpha_old : list of dict, length K
        alpha_old[k] = {"mu": (p,), "Sigma": (p,p), "gamma": (p,)} - old parameters.

    Returns
    -------
    alpha_new : list of dict, length K
        Updated parameter set with structure matching alpha_old.
    """

    n, p = X.shape
    K = len(alpha_old)

    alpha_new = [None]*K
    eps = 1e-12

    # For convenience, precompute inverses and a_k = sqrt(1 + gamma_k^T S_k^-1 gamma_k)
    inv_Sigma_old = []
    a_array = np.zeros(K)
    for k in range(K):
        Sk_inv = np.linalg.inv(alpha_old[k][1])
        inv_Sigma_old.append(Sk_inv)
        gk_old = alpha_old[k][2]
        a_array[k] = np.sqrt(1.0 + gk_old @ Sk_inv @ gk_old)

    V1 = np.zeros((n, K), dtype=float)
    V2 = np.zeros((n, K), dtype=float)
    for k in range(K):
        diff = X - alpha_old[k][0]
        dist_sq = np.einsum("ij,jk,ik->i", diff, inv_Sigma_old[k], diff)
        dist = np.sqrt(np.clip(dist_sq, 1e-14, None))
        a_sq = a_array[k] * a_array[k]
        V1[:, k] = a_array[k] / dist
        V2[:, k] = (1.0 + a_array[k] * dist) / a_sq

    # Now finish updates for each cluster k, using the partial sums:
    for k in range(K):
        weights = np.asarray(gamma_matrix[:, k], dtype=float)
        n_k = float(np.sum(weights))
        if n_k <= eps:
            alpha_new[k] = alpha_old[k]
            continue

        sum_z_v1 = float(np.dot(weights, V1[:, k]))
        sum_z_v2 = float(np.dot(weights, V2[:, k]))
        if sum_z_v1 <= eps:
            alpha_new[k] = alpha_old[k]
            continue

        sum_z_x = np.sum(weights[:, None] * X, axis=0)
        sum_z_v1_x = np.sum((weights * V1[:, k])[:, None] * X, axis=0)
        denom_gamma = sum_z_v2 - (n_k * n_k / sum_z_v1)
        if denom_gamma <= eps:
            alpha_new[k] = alpha_old[k]
            continue

        gk_new = (sum_z_x - n_k * (sum_z_v1_x / sum_z_v1)) / denom_gamma
        mu_k_new = (sum_z_v1_x - n_k * gk_new) / sum_z_v1

        residual = X - mu_k_new
        scatter = np.einsum(
            "i,ij,ik->jk", weights * V1[:, k], residual, residual
        )
        cross = np.einsum("i,ij,k->jk", weights, residual, gk_new)
        scatter = (
            scatter
            - cross
            - cross.T
            + sum_z_v2 * np.outer(gk_new, gk_new)
        )
        Sigma_k_new = _regularize_cov(scatter / n_k)

        alpha_new[k] = (mu_k_new, Sigma_k_new, gk_new)

    return alpha_new

def compute_log_pdf_skewlaplace(X, mu_k, Sigma_k, gamma_k):
    """
    Computes the log PDF of a multivariate skew-Laplace distribution for given parameters.

    Parameters:
    ----------
    X : array-like, shape (N, p)
        Data points where the density needs to be evaluated.
    mu_k : array-like, shape (p,)
        Mean vector of the distribution.
    Sigma_k : array-like, shape (p, p)
        Scatter matrix (must be positive-definite).
    gamma_k : array-like, shape (p,)
        Skewness vector.

    Returns:
    -------
    log_probs : array, shape (N,)
        Log-density values at each data point.
    """
    N, p = X.shape

    # Invert & logdet Sigma_k
    sign, logdetS = np.linalg.slogdet(Sigma_k)  # sign should be +1 for PD matrix
    invS_k = np.linalg.solve(Sigma_k, np.eye(p))  # More stable than np.linalg.inv()

    # Compute a_k = sqrt(1 + gamma_k.T @ invS_k @ gamma_k)
    gamma_invS_gamma = gamma_k @ invS_k @ gamma_k  # Scalar
    a_k = np.sqrt(1.0 + gamma_invS_gamma)

    # Normalizing constant in log domain
    logC_k = (-0.5 * logdetS
              - (p * np.log(2.0)
                 + (p - 1) * 0.5 * math.log(math.pi)
                 + np.log(a_k)
                 + np.log(gamma_func((p + 1) / 2.0))))

    # Compute Mahalanobis distance
    Xm = X - mu_k  # shape (N, p)
    dist_sq = np.einsum('ij,jk,ik->i', Xm, invS_k, Xm)  # shape (N,)
    dist_val = np.sqrt(np.clip(dist_sq, 1e-14, None))  # Ensure no sqrt(0) errors

    # Compute linear term
    lin_term = np.einsum('ij,j->i', Xm @ invS_k, gamma_k)  # shape (N,)

    # Compute exponent
    exponent = -a_k * dist_val + lin_term

    log_probs = logC_k + exponent
    return log_probs


class SkewLaplaceMix(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001,initialization="kmeans",print_log_likelihood=False,max_iter=25, verbose=True):
        super().__init__(n_clusters=n_clusters, EM_type="Soft", mixture_type="identical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.initialization=initialization
        self.family = "skew_laplace"

    def _log_pdf_skewlaplace(self,X,alphas):
        """
        Returns an array of shape (N, K) where entry (i, k) is
        log [ f_k( X[i] ) ], the log of the k-th Skew-Laplace pdf at X[i].

        X: shape (N, p)
        alphas: list of length K, each is {"mu": (p,), "Sigma": (p,p), "gamma": (p,)}.
        """
        N, p = X.shape
        K = len(alphas)
        log_probs = np.zeros((N, K))

        for k in range(K):
            mu_k = alphas[k][0]        # shape (p,)
            Sigma_k = alphas[k][1]  # shape (p,p)
            gamma_k = alphas[k][2]  # shape (p,)

            log_probs[:, k] = compute_log_pdf_skewlaplace(X, mu_k, Sigma_k, gamma_k)

        return log_probs


    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        return self._log_pdf_skewlaplace(X,alpha) + np.log(pi)

    def fit(self,sample):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (2*self.p + (self.p*(self.p+1)/2))
        np.random.seed(0)
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = k_means_init_skewlaplace(self.data, self.k)
        elif self.initialization == "gmm":
            self.pi_not, self.alpha_not = gmm_init_skewlaplace(self.data, self.k)
        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not
        pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(self.data,self.pi_temp,self.alpha_temp,estimate_alphas_skewlaplace)
        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.cluster = log_gamma_new.argmax(axis=1)
        self.gamma_temp_ar = np.exp(log_gamma_new)
        self.log_likelihood_new = log_likelihood_new
        if self.verbose:
            print("Mixtures of Skew-Laplace Fitting Done Successfully")
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

        IM, SE = compute_empirical_info_and_se(self)

        return IM, SE
