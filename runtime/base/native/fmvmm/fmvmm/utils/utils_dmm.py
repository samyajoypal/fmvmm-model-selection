import numpy as np
from sklearn.cluster import KMeans
from sklearn import mixture
from scipy.stats import dirichlet
import math
import pandas as pd
import conorm
from scipy.special import gammaln, psi, digamma
import scipy.special as sp
from scipy.stats import norm

# =============================================================================
# Helpers: ALR (reference = last component) and Jacobians
# =============================================================================

_EPS = 1e-15

def alr_transform(pi):
    """
    ALR transform with reference component pi_K (last component).
    pi: shape (K,), positive, sums to 1
    returns eta: shape (K-1,)
    """
    pi = np.asarray(pi, dtype=float)
    pi = np.clip(pi, _EPS, 1.0)
    pi = pi / pi.sum()
    return np.log(pi[:-1] / pi[-1])

def alr_inverse(eta):
    """
    Inverse ALR transform with reference component pi_K (last component).
    eta: shape (K-1,)
    returns pi: shape (K,)
    """
    eta = np.asarray(eta, dtype=float)
    exp_eta = np.exp(eta)
    denom = 1.0 + exp_eta.sum()
    pi = np.empty(len(eta) + 1, dtype=float)
    pi[:-1] = exp_eta / denom
    pi[-1] = 1.0 / denom
    return pi

def jacobian_pi_wrt_eta(pi):
    """
    Jacobian J = d pi / d eta for ALR with reference last component.
    pi: shape (K,)
    returns J: shape (K, K-1)
    """
    pi = np.asarray(pi, dtype=float)
    K = len(pi)
    p = pi[:-1]  # length K-1
    J = np.zeros((K, K-1), dtype=float)

    # For j=1..K-1, r=1..K-1:
    # d pi_j / d eta_r = pi_j (delta_{jr} - pi_r)
    for j in range(K - 1):
        for r in range(K - 1):
            J[j, r] = p[j] * ((1.0 if j == r else 0.0) - p[r])

    # For j=K:
    # d pi_K / d eta_r = - pi_K * pi_r
    for r in range(K - 1):
        J[K - 1, r] = -pi[K - 1] * p[r]

    return J

# =============================================================================
# Wald CIs (works with info in unconstrained coordinates: eta + alpha_flat)
# =============================================================================

def wald_confidence_intervals_dmm(mle, info_matrix, alpha=0.05):
    """
    Compute Wald confidence intervals for a Dirichlet Mixture Model (DMM).

    IMPORTANT (correct implementation):
    - The information matrix is assumed to be for an unconstrained parameterization:
        theta_tilde = (eta_1,...,eta_{K-1}, alpha_11,...,alpha_KD),
      where eta is the ALR transform of pi with reference pi_K.
    - This avoids singularity from the simplex constraint.

    Parameters
    ----------
    mle : tuple
        (pi, alphas):
        - pi: list/array of K mixture weights (summing to 1)
        - alphas: list of K lists (or array KxD) of Dirichlet parameters

    info_matrix : ndarray
        Observed information matrix on unconstrained scale,
        shape: ((K-1) + K*D, (K-1) + K*D)

    alpha : float
        Significance level for (1 - alpha)*100% confidence intervals

    Returns
    -------
    list of tuples
        Each tuple contains (lower_bound, upper_bound) for each parameter,
        in the order: [pi_1, ..., pi_K, alpha_11, ..., alpha_KD]
    """
    pi, alphas = mle
    pi = np.asarray(pi, dtype=float)
    pi = np.clip(pi, _EPS, 1.0)
    pi = pi / pi.sum()

    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim == 1:
        raise ValueError("alphas must be 2D: shape (K, D)")
    K = len(pi)
    D = alphas.shape[1]

    expected_dim = (K - 1) + K * D
    if info_matrix.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"info_matrix must have shape {(expected_dim, expected_dim)} "
            f"for (eta, alpha) parameterization, got {info_matrix.shape}."
        )

    # Covariance on unconstrained scale
    try:
        cov_matrix = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(info_matrix)

    z = norm.ppf(1 - alpha / 2)
    ci_list = []

    # --- Step 1: CIs for pi via ALR coordinates eta ---
    eta_hat = alr_transform(pi)  # length K-1
    cov_eta = cov_matrix[:(K-1), :(K-1)]

    # CI for each eta_j
    eta_ci = []
    for j in range(K - 1):
        se_eta_j = np.sqrt(max(cov_eta[j, j], 0.0))
        eta_low = eta_hat[j] - z * se_eta_j
        eta_high = eta_hat[j] + z * se_eta_j
        eta_ci.append((eta_low, eta_high))

    # Back-transform each eta_j interval holding other eta's fixed at eta_hat
    # This matches your manuscript approach and keeps pi in the simplex.
    for j in range(K - 1):
        eta_low_vec = eta_hat.copy()
        eta_high_vec = eta_hat.copy()
        eta_low_vec[j] = eta_ci[j][0]
        eta_high_vec[j] = eta_ci[j][1]
        pi_low = alr_inverse(eta_low_vec)
        pi_high = alr_inverse(eta_high_vec)

        # componentwise bounds for pi_j under this 1D perturbation
        ci_list.append((min(pi_low[j], pi_high[j]), max(pi_low[j], pi_high[j])))

    # CI for pi_K: use delta method on log(pi_K) from cov_eta (consistent and correct)
    # log pi_K = - log(1 + sum exp(eta_r))
    # gradient: d log pi_K / d eta_r = - pi_r  (for r=1..K-1)
    p = pi[:-1]  # pi_r, r=1..K-1
    grad_log_piK = -p  # shape (K-1,)
    var_log_piK = float(grad_log_piK @ cov_eta @ grad_log_piK.T)
    var_log_piK = max(var_log_piK, 0.0)
    se_log_piK = np.sqrt(var_log_piK)

    log_piK = np.log(pi[-1])
    log_piK_low = log_piK - z * se_log_piK
    log_piK_high = log_piK + z * se_log_piK
    ci_list.append((np.exp(log_piK_low), np.exp(log_piK_high)))

    # --- Step 2: log-transform alpha and delta method SEs (diagonal only) ---
    # cov entries correspond to alpha_flat appended after eta
    flat_alphas = alphas.reshape(-1)  # length K*D
    for i in range(K * D):
        alpha_hat = float(flat_alphas[i])
        alpha_hat = max(alpha_hat, _EPS)
        idx = (K - 1) + i
        var_alpha = float(cov_matrix[idx, idx])
        var_alpha = max(var_alpha, 0.0)

        se_log_alpha = np.sqrt(var_alpha) / alpha_hat  # delta method
        log_alpha = np.log(alpha_hat)
        ci_low = np.exp(log_alpha - z * se_log_alpha)
        ci_high = np.exp(log_alpha + z * se_log_alpha)
        ci_list.append((ci_low, ci_high))

    return ci_list

# =============================================================================
# Score vectors (unconstrained: eta + alpha)
# =============================================================================

def score_vector_observation(pi, alpha, x_i, gamma_i):
    """
    Computes the score vector for a single observation x_i on the UNCONSTRAINED scale:
        theta_tilde = (eta_1,...,eta_{k-1}, alpha_11,...,alpha_kd)

    - eta is ALR(pi) with reference pi_k.
    - alpha is unchanged.
    Returns:
        score_i: shape ((k-1) + k*d,)
    """
    pi = np.asarray(pi, dtype=float)
    pi = np.clip(pi, _EPS, 1.0)
    pi = pi / pi.sum()

    alpha = np.asarray(alpha, dtype=float)
    k, d = alpha.shape

    x_i = np.asarray(x_i, dtype=float)
    x_i = np.clip(x_i, _EPS, 1.0)

    gamma_i = np.asarray(gamma_i, dtype=float)

    # Score w.r.t eta_r (r=1..k-1): s_i(eta_r) = gamma_ir - pi_r
    score_eta = gamma_i[:k-1] - pi[:k-1]  # shape (k-1,)

    # Score w.r.t alpha
    score_alpha_blocks = []
    logx = np.log(x_i)
    for j in range(k):
        alpha_j = alpha[j]
        alpha_sum = np.sum(alpha_j)
        grad = sp.psi(alpha_sum) - sp.psi(alpha_j) + logx  # shape (d,)
        score_alpha_blocks.append(gamma_i[j] * grad)

    score_alpha = np.concatenate(score_alpha_blocks)  # shape (k*d,)
    return np.concatenate([score_eta, score_alpha])

def empirical_info_matrix(pi, alpha, gamma, x, mode='soft'):
    """
    Empirical observed information matrix using outer products of individual scores.
    Computed on UNCONSTRAINED scale (eta + alpha), so it is invertible in theory.

    Returns:
        I_emp: shape ((k-1) + k*d, (k-1) + k*d)
    """
    if mode == 'hard':
        gamma = hard_assignments(gamma)

    x = np.asarray(x, dtype=float)
    x = np.clip(x, _EPS, 1.0)

    N, d = x.shape
    k = len(pi)
    big_dim = (k - 1) + k * d
    I_emp = np.zeros((big_dim, big_dim), dtype=float)

    for i in range(N):
        s_i = score_vector_observation(pi, alpha, x[i], gamma[i])
        I_emp += np.outer(s_i, s_i)

    return I_emp

# =============================================================================
# Louis method pieces (full blocks, correct)
# =============================================================================

def hard_assignments(gamma):
    n_data, k = gamma.shape
    hard = np.zeros_like(gamma)
    max_indices = np.argmax(gamma, axis=1)
    hard[np.arange(n_data), max_indices] = 1.0
    return hard

def mixture_counts(gamma, mode='soft'):
    if mode == 'hard':
        gamma_hard = hard_assignments(gamma)
        N_j = np.sum(gamma_hard, axis=0)
    else:
        N_j = np.sum(gamma, axis=0)
    N = np.sum(N_j)
    return N_j, N

def single_dirichlet_info(alpha_j, n_j, ridge_factor=1e-10, use_pinv=False):
    """
    Correct Dirichlet information for one component on the raw alpha scale.
    Returns I_alpha and a robust inverse.
    """
    alpha_j = np.asarray(alpha_j, dtype=float)
    alpha_j = np.clip(alpha_j, _EPS, np.inf)

    d = len(alpha_j)
    alpha_sum = float(np.sum(alpha_j))

    D_vals = n_j * sp.polygamma(1, alpha_j)      # n_j * trigamma(alpha_jm)
    psi_sum = sp.polygamma(1, alpha_sum)         # trigamma(sum alpha_j)
    G = - n_j * psi_sum                          # off-diagonal contribution

    I_alpha = np.diag(D_vals) + G * np.ones((d, d), dtype=float)

    M = I_alpha.shape[0]
    try:
        I_alpha_inv = np.linalg.inv(I_alpha)
    except np.linalg.LinAlgError:
        if use_pinv:
            I_alpha_inv = np.linalg.pinv(I_alpha)
        else:
            ridge_mat = I_alpha + ridge_factor * np.eye(M)
            try:
                I_alpha_inv = np.linalg.inv(ridge_mat)
            except np.linalg.LinAlgError:
                I_alpha_inv = np.linalg.pinv(ridge_mat)

    return I_alpha, I_alpha_inv

def missing_info_alpha(alpha_j, gamma_j, x):
    """
    Missing information for one alpha_j block:
        sum_i gamma_ij (1-gamma_ij) A_i A_i^T
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, _EPS, 1.0)

    gamma_j = np.asarray(gamma_j, dtype=float)
    alpha_j = np.asarray(alpha_j, dtype=float)
    alpha_j = np.clip(alpha_j, _EPS, np.inf)

    N, d = x.shape
    alpha_sum = float(np.sum(alpha_j))

    A = (sp.psi(alpha_sum) - sp.psi(alpha_j) + np.log(x))  # (N,d)
    w = gamma_j * (1.0 - gamma_j)                          # (N,)
    # weighted sum of outer products
    I_miss = np.zeros((d, d), dtype=float)
    for i in range(N):
        I_miss += w[i] * np.outer(A[i], A[i])
    return I_miss

def louis_info_full_unconstrained(pi, alpha, gamma, x, mode='soft'):
    """
    Full Louis observed information on UNCONSTRAINED parameterization:
        theta_tilde = (eta_1,...,eta_{k-1}, alpha_11,...,alpha_kd)

    Includes cross-block terms:
        I_obs(eta, alpha_j) and I_obs(alpha_j, alpha_r) for j != r.

    Returns:
        I_obs: shape ((k-1) + k*d, (k-1) + k*d)
    """
    if mode == 'hard':
        gamma = hard_assignments(gamma)

    pi = np.asarray(pi, dtype=float)
    pi = np.clip(pi, _EPS, 1.0)
    pi = pi / pi.sum()

    alpha = np.asarray(alpha, dtype=float)
    k, d = alpha.shape

    x = np.asarray(x, dtype=float)
    x = np.clip(x, _EPS, 1.0)

    gamma = np.asarray(gamma, dtype=float)
    N = gamma.shape[0]

    big_dim = (k - 1) + k * d
    I_comp = np.zeros((big_dim, big_dim), dtype=float)
    I_miss = np.zeros((big_dim, big_dim), dtype=float)

    # ---- (A) eta block: complete and missing ----
    # Complete info for eta (k-1 x k-1): N * (diag(p) - p p^T), p = pi[0:k-1]
    p = pi[:k-1]
    I_comp_eta = N * (np.diag(p) - np.outer(p, p))

    # Missing info for eta: sum_i Cov(I_r, I_s | x_i) for r,s=1..k-1
    # Cov(I_r, I_s|x_i) = gamma_ir (delta_rs - gamma_is), for r,s<k
    I_miss_eta = np.zeros((k - 1, k - 1), dtype=float)
    g_sub = gamma[:, :k-1]  # (N, k-1)
    for i in range(N):
        gi = g_sub[i]
        I_miss_eta += np.diag(gi) - np.outer(gi, gi)

    I_comp[:k-1, :k-1] = I_comp_eta
    I_miss[:k-1, :k-1] = I_miss_eta

    # Precompute A_i^{(j)} for all i,j: A[j][i, m]
    logx = np.log(x)
    A_all = []
    for j in range(k):
        alpha_j = np.clip(alpha[j], _EPS, np.inf)
        a_sum = float(np.sum(alpha_j))
        A_all.append(sp.psi(a_sum) - sp.psi(alpha_j) + logx)  # (N,d)

    # ---- (B) alpha blocks: complete and missing (within-block) ----
    N_j = gamma.sum(axis=0)  # (k,)
    for j in range(k):
        I_j, _ = single_dirichlet_info(alpha[j], N_j[j])
        rs = (k - 1) + j * d
        I_comp[rs:rs + d, rs:rs + d] = I_j
        I_miss[rs:rs + d, rs:rs + d] = missing_info_alpha(alpha[j], gamma[:, j], x)

    # ---- (C) cross blocks from missing information only ----
    # (C1) eta - alpha cross:
    # For r=1..k-1, component j=1..k:
    # I_obs(eta_r, alpha_{jm}) = -Cov(s_eta^c, s_alpha^c)
    # s_eta^c = I_r - pi_r, s_alpha^c = I_j A^{(j)}.
    # Cov(I_r, I_j|x_i) = gamma_ir(1-gamma_ir) if j=r; else -gamma_ir gamma_ij
    # => I_obs = - sum_i Cov * A
    for r in range(k - 1):
        for j in range(k):
            if j == r:
                w = gamma[:, r] * (1.0 - gamma[:, r])      # (N,)
                block = - (w[:, None] * A_all[j]).sum(axis=0)  # (d,)
            else:
                w = gamma[:, r] * gamma[:, j]              # (N,)
                block = (w[:, None] * A_all[j]).sum(axis=0)   # (d,)

            # place into I_miss then later subtract, or directly into I_obs via I_comp=0
            # Here: cross blocks are missing-only, so I_comp cross=0 and I_obs cross = -I_miss cross.
            # Our computed "block" already equals I_obs(eta_r, alpha_j*) as per formulas above.
            row = r
            col_start = (k - 1) + j * d
            I_comp[row, col_start:col_start + d] = 0.0
            I_comp[col_start:col_start + d, row] = 0.0

            # Store in I_obs directly later. For now, fill I_miss so that I_obs=I_comp-I_miss works:
            # Need I_miss cross = -I_obs cross
            I_miss[row, col_start:col_start + d] = -block
            I_miss[col_start:col_start + d, row] = -block

    # (C2) alpha_j - alpha_r cross for j != r:
    # I_obs(alpha_jm, alpha_rt) = sum_i gamma_ij gamma_ir A_im^{(j)} A_it^{(r)}
    for j in range(k):
        for r in range(k):
            if r == j:
                continue
            w = gamma[:, j] * gamma[:, r]  # (N,)
            # block (d x d): sum_i w_i * (A_j[i][:,None] * A_r[i][None,:])
            block = np.zeros((d, d), dtype=float)
            Aj = A_all[j]
            Ar = A_all[r]
            for i in range(N):
                block += w[i] * np.outer(Aj[i], Ar[i])

            rs_j = (k - 1) + j * d
            rs_r = (k - 1) + r * d
            # Again, cross blocks are missing-only, so I_obs cross = - I_miss cross.
            # We want I_obs cross = block -> I_miss cross = -block
            I_miss[rs_j:rs_j + d, rs_r:rs_r + d] = -block
            I_miss[rs_r:rs_r + d, rs_j:rs_j + d] = -block.T

    # ---- Observed information ----
    I_obs = I_comp - I_miss
    return I_obs

# =============================================================================
# Main entry: observed info + SE (Louis or score), consistent and correct
# =============================================================================

def combined_info_and_se(pi, alpha, gamma, x, method='score', mode='soft'):
    """
    Computes observed Fisher Information and SEs.

    CORRECT VERSION:
    - Works on UNCONSTRAINED parameterization:
        theta_tilde = (eta_1,...,eta_{k-1}, alpha_11,...,alpha_kd),
      where eta is ALR(pi) with reference last weight.
    - This avoids singularity from the simplex constraint.
    - Louis method includes cross-block terms (eta-alpha and alpha-alpha between components).

    Returns:
        I_obs: observed info matrix on unconstrained scale
              shape ((k-1) + k*d, (k-1) + k*d)
        se:    standard errors on the same unconstrained scale (sqrt(diag(inv(I_obs))))
    """
    if method == "louis":
        I_obs = louis_info_full_unconstrained(pi, alpha, gamma, x, mode=mode)
        try:
            I_inv = np.linalg.inv(I_obs)
        except np.linalg.LinAlgError:
            I_inv = np.linalg.pinv(I_obs)
        se = np.sqrt(np.maximum(np.diag(I_inv), 0.0))
        return I_obs, se

    elif method == "score":
        IM = empirical_info_matrix(pi, alpha, gamma, x, mode=mode)
        try:
            IM_inv = np.linalg.inv(IM)
        except np.linalg.LinAlgError:
            IM_inv = np.linalg.pinv(IM)
        se = np.sqrt(np.maximum(np.diag(IM_inv), 0.0))
        return IM, se

    else:
        raise NotImplementedError("Please use louis or score method. Other methods are not implemented!")


# -------------------------------------------------
# Mean + precision inference via delta method
# -------------------------------------------------

def mean_precision_inference(alpha_hat, cov_alpha, alpha_level=0.05):
    """
    alpha_hat : (p,)
    cov_alpha : (p,p)
    """

    p = len(alpha_hat)
    z = norm.ppf(1 - alpha_level/2)

    tau = alpha_hat.sum()
    mu = alpha_hat / tau

    # ---------- Precision ----------
    ones = np.ones(p)
    var_tau = ones @ cov_alpha @ ones
    se_tau = np.sqrt(var_tau)
    ci_tau = (tau - z*se_tau, tau + z*se_tau)

    # ---------- Mean ----------
    G = (np.eye(p) - np.outer(mu, np.ones(p))) / tau
    cov_mu = G @ cov_alpha @ G.T
    se_mu = np.sqrt(np.diag(cov_mu))

    ci_mu = []
    for m in range(p):
        eta = np.log(mu[m]/(1-mu[m]))
        se_eta = se_mu[m]/(mu[m]*(1-mu[m]))
        lo = eta - z*se_eta
        hi = eta + z*se_eta
        ci_mu.append((
            np.exp(lo)/(1+np.exp(lo)),
            np.exp(hi)/(1+np.exp(hi))
        ))

    return {
        "tau": tau,
        "se_tau": se_tau,
        "ci_tau": ci_tau,
        "mu": mu,
        "se_mu": se_mu,
        "ci_mu": ci_mu
    }


def mixture_proportions_info(pi, N_j):
    """
    For mixture weights pi (length k), compute the unconstrained (k x k)
    information matrix I_pi and its inverse, using:
       I_pi       = diag(N_j / pi_j^2),
       I_pi_inv   = diag(pi_j^2 / N_j).
    """
    # pi, N_j should each be shape (k,)
    I_diag = N_j / (pi**2)
    I_pi = np.diag(I_diag)

    I_inv_diag = (pi**2) / N_j
    I_pi_inv = np.diag(I_inv_diag)

    return I_pi, I_pi_inv


def dirichlet_kl_divergence(alpha1, alpha2):
    """
    Compute KL divergence between two Dirichlet distributions.

    Parameters:
        alpha1 (numpy array): Parameters of the first Dirichlet distribution.
        alpha2 (numpy array): Parameters of the second Dirichlet distribution.

    Returns:
        float: KL divergence between the two Dirichlet distributions.
    """
    sum_alpha1 = np.sum(alpha1)
    sum_alpha2 = np.sum(alpha2)

    term1 = gammaln(sum_alpha1) - gammaln(sum_alpha2)
    term2 = np.sum(gammaln(alpha2) - gammaln(alpha1))
    term3 = np.sum((alpha1 - alpha2) * (digamma(alpha1) - digamma(sum_alpha1)))

    kl_divergence = term1 + term2 + term3
    return kl_divergence

def dmm_kl_divergence(pi, omega, f_models, g_models):
    """
    Compute KL divergence between two Dirichlet mixture models.

    Parameters:
        pi (numpy array): Mixing coefficients for the first mixture model.
        omega (numpy array): Mixing coefficients for the second mixture model.
        f_models (list of numpy arrays): Parameters of Dirichlet distributions in the first mixture model.
        g_models (list of numpy arrays): Parameters of Dirichlet distributions in the second mixture model.

    Returns:
        float: KL divergence between the two Dirichlet mixture models.
    """
    kl_divergence = 0.0

    for a in range(len(pi)):
        term_num = 0.0
        term_denom = 0.0

        for aprime in range(len(pi)):
            term_num += pi[aprime] * np.exp(-dirichlet_kl_divergence(f_models[a], f_models[aprime]))

        for b in range(len(omega)):
            term_denom += omega[b] * np.exp(-dirichlet_kl_divergence(f_models[a], g_models[b]))

        if term_denom==0:
            term_denom=term_denom+1e-243
        if term_num==0:
            term_num=term_num+1e-243
        kl_divergence += pi[a] * np.log(term_num / term_denom)

    return kl_divergence


def dmm_mc_kl_divergence(pi, omega, f_models, g_models, n_samples=1000):
    """
    Monte Carlo estimation of KL Divergence between two Dirichlet mixture models.

    Parameters:
        pi (numpy array): Mixing coefficients for the first mixture model.
        omega (numpy array): Mixing coefficients for the second mixture model.
        f_models (2D numpy array): Parameters of Dirichlet distributions in the first mixture model.
        g_models (2D numpy array): Parameters of Dirichlet distributions in the second mixture model.
        n_samples (int): Number of samples to draw.

    Returns:
        float: Estimated KL Divergence.
    """
    np.random.seed(0)
    k=len(pi)
    assert n_samples>k
    nis=[int(n_samples*i) for i in pi]
    xis=[]
    for j in range(k):
        xis.extend(np.random.dirichlet(f_models[j],nis[j]))

    logs = []

    for xi in xis:
        # # Draw a sample xi from the mixture model using pi and f_models
        # xi_=[pi[j] * np.random.dirichlet(f_models[j]) for j in range(len(pi))]
        # xi = np.sum(xi_,axis=0)
        # # print(xi_)
        # Calculate f_pdf and g_pdf
        with np.errstate(under='ignore'):
            f_pdf_=[pi[j] * dirichlet.pdf(xi, f_models[j]) for j in range(len(pi))]
            f_pdf = np.nansum(np.array(f_pdf_,dtype=np.float128))
            g_pdf_=[omega[j] * dirichlet.pdf(xi, g_models[j]) for j in range(len(omega))]
            g_pdf = np.nansum(np.array(g_pdf_,dtype=np.float128))

            # Calculate log(f(x_i)/g(x_i)) and append to logs
            logs.append(np.log(f_pdf / g_pdf))

    # Average over all samples
    return np.nansum(logs)/len(logs)



def sample_dirichlet_gibbs(alpha, n_iterations=1000):
  """
  This function draws samples from a Dirichlet distribution using the Gibbs sampler and averages them.

  Args:
      alpha: A numpy array of length n_components representing the Dirichlet parameters.
      n_samples: The number of samples to draw.

  Returns:
      A numpy array of shape (n_samples, n_components) containing the average of the samples.
  """

  # Initialize an empty list to store samples
  samples = []

  # Iterate for the desired number of samples
  for _ in range(n_iterations):
    # Generate a single sample
    sample = _sample_dirichlet_gibbs(alpha)  # Helper function for one sample
    samples.append(sample)

  # Convert the list of samples to a numpy array and return the average
  return np.mean(np.array(samples), axis=0)


def _sample_dirichlet_gibbs(alpha, n_samples=1):
  """
  This function draws samples from a Dirichlet distribution using the Gibbs sampler.

  Args:
      alpha: A numpy array of length n_components representing the Dirichlet parameters.
      n_samples: The number of samples to draw.

  Returns:
      A numpy array of shape (n_samples, n_components) containing the samples.
  """
  n_components=len(alpha)
  # Initialize the samples matrix with random values from a uniform distribution
  samples = np.random.rand(n_samples, n_components)

  # Normalize each row to sum to 1 (represents probability distribution)
  samples /= samples.sum(axis=1, keepdims=True)

  for _ in range(n_samples - 1):
    # Iterate over each component
    for k in range(n_components):
      # Calculate the sum of alphas excluding the current component
      other_alpha_sum = alpha.sum() - alpha[k]
      # Sample from a beta distribution using the updated alpha values
      samples[:, k] = np.random.beta(alpha[k] + samples[:, :k].sum(axis=1), other_alpha_sum + samples[:, k+1:].sum(axis=1))

  return samples.flatten()


def dmm_mcmc_kl_divergence(pi, omega, f_models, g_models, n_samples=1000):
    """
    Monte Carlo estimation of KL Divergence between two Dirichlet mixture models.

    Parameters:
        pi (numpy array): Mixing coefficients for the first mixture model.
        omega (numpy array): Mixing coefficients for the second mixture model.
        f_models (2D numpy array): Parameters of Dirichlet distributions in the first mixture model.
        g_models (2D numpy array): Parameters of Dirichlet distributions in the second mixture model.
        n_samples (int): Number of samples to draw.

    Returns:
        float: Estimated KL Divergence.
    """
    logs = []
    np.random.seed(0)

    for _ in range(n_samples):
        # Draw a sample xi from the mixture model using pi and f_models
        xi = np.sum([pi[j] * sample_dirichlet_gibbs(f_models[j]) for j in range(len(pi))],axis=0)
        # print(xi)
        # Calculate f_pdf and g_pdf
        f_pdf = np.sum([pi[j] * dirichlet.pdf(xi, f_models[j]) for j in range(len(pi))])
        g_pdf = np.sum([omega[j] * dirichlet.pdf(xi, g_models[j]) for j in range(len(omega))])

        # Calculate log(f(x_i)/g(x_i)) and append to logs
        logs.append(np.log(f_pdf / g_pdf))

    # Average over all samples
    return np.mean(logs)



def closure(d_mat):
    d_mat = np.atleast_2d(d_mat)
    if np.any(d_mat < 0):
        raise ValueError("Cannot have negative proportions")
    if d_mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(d_mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    d_mat = d_mat / d_mat.sum(axis=1, keepdims=True)
    return d_mat.squeeze()

def multiplicative_replacement(d_mat, delta=None):
    d_mat = closure(d_mat)
    z_mat = (d_mat == 0)

    num_feats = d_mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1. / num_feats)**2

    zcnts = 1 - tot * delta
    if np.any(zcnts) < 0:
        raise ValueError('The multiplicative replacement created negative '
                         'proportions. Consider using a smaller `delta`.')
    d_mat = np.where(z_mat, delta, zcnts * d_mat)
    return d_mat.squeeze()


def clr(mat):
    r"""
    Performs centre log ratio transformation.

    This function transforms compositions from Aitchison geometry to
    the real space. The :math:`clr` transform is both an isometry and an
    isomorphism defined on the following spaces

    :math:`clr: S^D \rightarrow U`

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

    .. math::
        clr(x) = \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right]

    where :math:`g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.

    Parameters
    ----------
    mat : array_like, float
       a matrix of proportions where
       rows = compositions and
       columns = components

    Returns
    -------
    numpy.ndarray
         clr transformed matrix

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import clr
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])

    """
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()










def kmeans_init(data, k):
    n = len(data)
    p = len(data.columns)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)
    mu_not = kmeans.cluster_centers_
    #alsum=60
    alsum = p*5
    alpha_not = mu_not*alsum
    data_lol = data.values.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    return pi_not, alpha_not


def gmm_init(data, k):
    n = len(data)
    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(data)
    mu_not = clf.means_
    alsum = 60
    alpha_not = mu_not*alsum
    data_lol = data.values.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[clf.predict(data)[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    return pi_not, alpha_not


def random_init(data, k, random_seed=0):
    np.random.seed(random_seed)
    p = data.shape[1]
    alpha_not = []
    for h in range(k):
        alpha_not_temp = np.random.uniform(0, 50, p)
        alpha_not.append(alpha_not_temp)
    pi_not = sum(np.random.dirichlet([0.5 for i in range(k)], 1).tolist(), [])

    return pi_not, alpha_not


def kmeans_init_adv(data, k):
    n = len(data)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data)
    #alpha_not=mu_not*alsum
    data_lol = data.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[kmeans.labels_[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    alpha_new = []
    for i in range(k):
        data_cwise_ar = np.array(data_cwise[i])
        E = data_cwise_ar.mean(axis=0)
        E2 = (data_cwise_ar ** 2).mean(axis=0)
        E3 = ((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E
        alpha_new.append(E3.tolist())
    return pi_not, alpha_new


def gmm_init_adv(data, k):
    n = len(data)
    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(data)
    #alpha_not=mu_not*alsum
    data_lol = data.tolist()
    data_cwise = [[] for i in range(k)]
    for i in range(n):
        data_cwise[clf.predict(data)[i]].append(data_lol[i])
    pi_not = [len(data_cwise[m])/n for m in range(k)]
    alpha_new = []
    for i in range(k):
        data_cwise_ar = np.array(data_cwise[i])
        E = data_cwise_ar.mean(axis=0)
        E2 = (data_cwise_ar ** 2).mean(axis=0)
        E3 = ((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E
        alpha_new.append(E3.tolist())
    return pi_not, alpha_new


def dmm_loglikelihood(pi_temp, alpha_temp, data_lol):
    n = len(data_lol)
    k = len(alpha_temp)
    log_likelihood_values_temp = []
    for c in range(n):
        try:
            log_likelihood_old_temp = math.log(np.nansum(
                [pi_temp[f]*dirichlet.pdf(data_lol[c], alpha_temp[f]) for f in range(k)]))
        except:
            log_likelihood_old_temp = math.log1p(np.nansum(
                [pi_temp[f]*dirichlet.pdf(data_lol[c], alpha_temp[f]) for f in range(k)]))
        log_likelihood_values_temp.append(log_likelihood_old_temp)
    log_likelihood_old = np.sum(log_likelihood_values_temp)

    return log_likelihood_old


def dmm_responsibilities(pi_temp, alpha_temp, data_lol):
    n = len(data_lol)
    k = len(alpha_temp)
    gamma_temp = []
    for i in range(n):
        gamma_numer = []
        for j in range(k):
            temp_gamma_numer = (
                pi_temp[j]*dirichlet.pdf(data_lol[i], alpha_temp[j]))
            gamma_numer.append(temp_gamma_numer)
        gamma_row = gamma_numer / np.nansum(gamma_numer)
        gamma_temp.append(gamma_row)
    gamma_temp_ar = np.array(gamma_temp, dtype=np.float64)
    gamma_matrix = []
    for v in gamma_temp:
        gm_temp = v.tolist()
        gamma_matrix.append(gm_temp)
    return gamma_temp_ar, gamma_matrix


def dmm_pi_estimate(gamma_temp_ar):
    n = gamma_temp_ar.shape[0]
    k = gamma_temp_ar.shape[1]
    pi_new = []
    nk = []
    for g in range(k):
        nk_temp = np.nansum([gamma_temp_ar[w, g] for w in range(n)])
        pi_temp = nk_temp/n
        pi_new.append(pi_temp)
        nk.append(nk_temp)
    return pi_new


def count_to_comp(df):
    df_array=np.array(df)


    nf  = conorm.tmm_norm_factors(df)["norm.factors"]

    lj=[]
    for j in range(df_array.shape[1]):
        lj_temp=nf[j]*np.sum(df_array[:, j])
        lj.append(lj_temp)


    sj=[]
    for j in range(df_array.shape[1]):
        sj_temp=lj[j]/(np.sum(lj)/df_array.shape[1])
        sj.append(sj_temp)

    x_lol=[]
    for i in range(df_array.shape[0]):
        xi=[]
        for j in range(df_array.shape[1]):
            xi_temp=df_array[i,j]/sj[j]
            xi.append(xi_temp)
        xi_sum=np.sum(xi)
        xi_trans=[xi[k]/xi_sum for k in range(df_array.shape[1])]


        x_lol.append(xi_trans)
        #x_lol.append(xi)

    data=pd.DataFrame(x_lol)
    # trans_data=pd.DataFrame(multiplicative_replacement(data))
    trans_data=pd.DataFrame(data)

    return trans_data




def dirichlet_covariance(alpha):
    p = len(alpha)
    alpha0 = np.sum(alpha)
    cov = np.zeros((p, p))

    for i in range(p):
        for j in range(p):
            if i == j:
                cov[i, j] = (alpha[i] * (alpha0 - alpha[i])) / (alpha0**2 * (alpha0 + 1))
            else:
                cov[i, j] = -(alpha[i] * alpha[j]) / (alpha0**2 * (alpha0 + 1))

    return cov
