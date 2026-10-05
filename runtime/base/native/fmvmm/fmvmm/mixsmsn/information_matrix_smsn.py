"""
Source codes are from R package mixsmsn.
Converted to python as closely as possible. Many thanks to the authors.
"""

import numpy as np
import math
from scipy.linalg import sqrtm
from numpy.linalg import det, inv
from scipy.stats import gamma as gamma_dist, t as student_t, norm
from fmvmm.mixsmsn.dens import dmvt_ls, d_mixedmvST
from math import pi, sqrt, log
from scipy.special import gamma as gamma_func, digamma
from math import exp

def iPhi_SN(Ai, di):
    """
    R's I.Phi(...) for Skew.normal:
      I.Phi(Ai=Ai, di=di) = exp(-di/2) * pnorm(Ai)
    """
    return math.exp(-0.5*di)*norm.cdf(Ai)

def iPhi_SN_phi(Ai, di):
    """
    R's I.phi(...) for Skew.normal:
      I.phi(Ai=Ai, di=di) = exp(-di/2)*dnorm(Ai)
    """
    return math.exp(-0.5*di)*norm.pdf(Ai)

# def info_matrix_skewnormal(
#     X,
#     pi_list,         # shape (g,)
#     mus,             # list of length g: mu_j (p,)
#     Sigmas,          # list of length g: Sigma_j (p,p)
#     lambdas,         # list of length g: shape_j (p,)
#     d_mixedmvSN_func,# mixture PDF for Skew-Normal
#     dmvSN_func,      # single-component Skew-Normal PDF
#     g,
#     p
# ):
#     """
#     Compute the observed information matrix for a mixture of multivariate Skew-Normal distributions,
#     following the R snippet if (class(model) == "Skew.normal"){ ... }.
#
#     Parameter order for cluster j:
#       mu_j => p
#       shape_j => p
#       Sigma_j => p*(p+1)//2
#     Then if g>1 => (g-1) partials for pi.
#
#     Returns
#     -------
#     IM : (M, M) numpy array
#     """
#     X = np.asarray(X)
#     n = X.shape[0]
#
#     # cluster_params = 2p + p(p+1)//2
#     cluster_params = 2*p + (p*(p+1)//2)
#     # total dimension M
#     M = g*cluster_params + (g-1 if g>1 else 0)
#     IM = np.zeros((M, M), dtype=float)
#
#     def fill_sigma_derivs(S_i, offset, Ssigma):
#         """
#         For cluster j, we place the partial derivatives of Sigma
#         after the first 2p slots (mu_j, shape_j).
#         """
#         start_ = offset + 2*p
#         for k_ in range(len(Ssigma)):
#             S_i[start_ + k_] = Ssigma[k_]
#
#     for i in range(n):
#         # Build gradient vector S_i
#         S_i = np.zeros(M, dtype=float)
#         # Evaluate mixture pdf at X[i]
#         mix_pdf_val = d_mixedmvSN_func(X[i:i+1,:], pi_list, mus, Sigmas, lambdas)[0]
#
#         # partial wrt pi => accumulate after cluster blocks
#         dSipi = np.zeros(g-1, dtype=float)
#
#         for j_ in range(g):
#             mu_j = mus[j_]
#             Sigma_j = Sigmas[j_]
#             lam_j = lambdas[j_]
#
#             diff = X[i] - mu_j
#             Dr = sqrtm(Sigma_j + 1e-15*np.eye(p))
#             Dr_inv = inv(Dr)
#             d_sig = det(Sigma_j)
#             # Ai = lam_j^T Dr_inv (y_i - mu_j)
#             Ai = lam_j @ Dr_inv @ diff
#             # di = diff^T Sigma_j^-1 diff
#             di = diff @ inv(Sigma_j) @ diff
#
#             # partial wrt mu_j
#             # dir.dmu = -2*(Dr_inv@Dr_inv)(y[i]-mu_j)
#             dir_dmu = -2.0*(Dr_inv@Dr_inv)@diff.reshape(-1,1)
#             dir_dmu = dir_dmu.ravel()
#             # dAir.dmu = -Dr_inv@lam_j
#             dAir_dmu = -(Dr_inv@lam_j.reshape(-1,1)).ravel()
#
#             # dPsi.dmu => c_ * [ dAir_dmu*iPhi_SN_phi(...) - 0.5*dir_dmu*iPhi_SN(...) ]
#             c_ = 2.0*(d_sig**-0.5)/((2.0*math.pi)**(p/2.0))
#             val_iPhi = iPhi_SN(Ai, di)
#             val_iPhi_phi = iPhi_SN_phi(Ai, di)
#             dPsi_dmu_j = c_*( dAir_dmu*val_iPhi_phi - 0.5*dir_dmu*val_iPhi )
#
#             # partial wrt shape_j => lam_j
#             # dAir.dlambda = Dr_inv (y_i - mu_j)
#             dAir_dlambda = Dr_inv@diff.reshape(-1,1)
#             dAir_dlambda = dAir_dlambda.ravel()
#             dPsi_dlambda_j = c_*( dAir_dlambda*val_iPhi_phi )
#
#             # partial wrt Sigma => do double loop
#             Ssigma_j = np.zeros(p*(p+1)//2, dtype=float)
#             l_ = 0
#             idx_ = 0
#             # We'll define iPhi(Ai,di) again, iPhi_SN(Ai,di)
#             for l_ in range(p):
#                 for m_ in range(l_, p):
#                     D = np.zeros((p,p), dtype=float)
#                     D[l_,m_] = 1.0
#                     D[m_,l_] = 1.0
#
#                     Dr_det = det(Dr)
#                     val_ddet = -(1.0/(Dr_det**2))*deriv_der(Dr, Dr_inv, D)
#                     left_mat = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
#                     val_dir = -(diff.reshape(1,-1)@left_mat@diff.reshape(-1,1))[0,0]
#                     val_dAir = -(lam_j.reshape(1,-1)@Dr_inv@D@Dr_inv@diff.reshape(-1,1))[0,0]
#
#                     c2_ = 2.0/((2.0*math.pi)**(p/2.0))
#                     # from snippet:
#                     # dPsi.dsigma = c2_ * [ val_ddet*iPhi_SN(...) - 0.5*val_dir*d_sig^-0.5*iPhi_SN(...) + d_sig^-0.5*val_dAir*iPhi_SN_phi(...) ]
#                     part_ = ( val_ddet*val_iPhi
#                               - 0.5*val_dir*(d_sig**-0.5)*val_iPhi
#                               + (d_sig**-0.5)*val_dAir*val_iPhi_phi )
#                     dPsi_dsigma_k = c2_*part_
#                     Ssigma_j[idx_] = dPsi_dsigma_k
#                     idx_ += 1
#
#             factor_j = pi_list[j_]/(mix_pdf_val + 1e-300)
#             dPsi_dmu_j      = factor_j*dPsi_dmu_j
#             dPsi_dlambda_j  = factor_j*dPsi_dlambda_j
#             Ssigma_j        = factor_j*Ssigma_j
#
#             # place them in S_i
#             offset_j = j_*cluster_params
#             S_i[offset_j : offset_j+p]       = dPsi_dmu_j
#             S_i[offset_j + p : offset_j + 2*p] = dPsi_dlambda_j
#             fill_sigma_derivs(S_i, offset_j, Ssigma_j)
#
#         # partial wrt pi => if g>1
#         if g>1:
#             # compute single-comp pdf for each cluster
#             f_clust = np.zeros(g)
#             for j_ in range(g):
#                 pdf_j_ = dmvSN_func(X[i:i+1,:], mus[j_], Sigmas[j_], lambdas[j_])
#                 f_clust[j_] = pdf_j_[0]
#             for j_ in range(g-1):
#                 dSipi[j_] = (1.0/(mix_pdf_val+1e-300))*( f_clust[j_] - f_clust[g-1] )
#             pi_offset = g*cluster_params
#             S_i[ pi_offset : pi_offset+(g-1) ] = dSipi
#
#         IM += np.outer(S_i, S_i)
#
#     return IM

def info_matrix_skewnormal(
    X,
    pi_list,
    mus,
    Sigmas,
    lambdas,
    d_mixedmvSN_func,
    dmvSN_func,
    g,
    p,
    *,
    return_scores: bool = False,
    ridge: float = 0.0,
):
    X = np.asarray(X)
    n = X.shape[0]

    cluster_params = 2*p + (p*(p+1)//2)
    M = g*cluster_params + (g-1 if g > 1 else 0)

    # store per-observation scores if requested
    S = np.zeros((n, M), dtype=float) if return_scores else None
    IM = np.zeros((M, M), dtype=float)

    def fill_sigma_derivs(S_i, offset, Ssigma):
        start_ = offset + 2*p
        S_i[start_ : start_ + len(Ssigma)] = Ssigma

    for i in range(n):
        S_i = np.zeros(M, dtype=float)
        mix_pdf_val = d_mixedmvSN_func(X[i:i+1, :], pi_list, mus, Sigmas, lambdas)[0]

        dSipi = np.zeros(g-1, dtype=float)

        for j_ in range(g):
            mu_j = mus[j_]
            Sigma_j = Sigmas[j_]
            lam_j = lambdas[j_]

            diff = X[i] - mu_j
            Dr = sqrtm(Sigma_j + 1e-15*np.eye(p))
            Dr_inv = inv(Dr)
            d_sig = det(Sigma_j)

            Ai = lam_j @ Dr_inv @ diff
            di = diff @ inv(Sigma_j) @ diff

            dir_dmu = (-2.0*(Dr_inv@Dr_inv)@diff.reshape(-1,1)).ravel()
            dAir_dmu = (-(Dr_inv@lam_j.reshape(-1,1))).ravel()

            c_ = 2.0*(d_sig**-0.5)/((2.0*math.pi)**(p/2.0))
            val_iPhi = iPhi_SN(Ai, di)
            val_iPhi_phi = iPhi_SN_phi(Ai, di)
            dPsi_dmu_j = c_*(dAir_dmu*val_iPhi_phi - 0.5*dir_dmu*val_iPhi)

            dAir_dlambda = (Dr_inv@diff.reshape(-1,1)).ravel()
            dPsi_dlambda_j = c_*(dAir_dlambda*val_iPhi_phi)

            Ssigma_j = np.zeros(p*(p+1)//2, dtype=float)
            idx_ = 0
            for l_ in range(p):
                for m_ in range(l_, p):
                    D = np.zeros((p,p), dtype=float)
                    D[l_, m_] = 1.0
                    D[m_, l_] = 1.0

                    Dr_det = det(Dr)
                    val_ddet = -(1.0/(Dr_det**2))*deriv_der(Dr, Dr_inv, D)
                    left_mat = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
                    val_dir = -(diff.reshape(1,-1)@left_mat@diff.reshape(-1,1))[0,0]
                    val_dAir = -(lam_j.reshape(1,-1)@Dr_inv@D@Dr_inv@diff.reshape(-1,1))[0,0]

                    c2_ = 2.0/((2.0*math.pi)**(p/2.0))
                    part_ = (
                        val_ddet*val_iPhi
                        - 0.5*val_dir*(d_sig**-0.5)*val_iPhi
                        + (d_sig**-0.5)*val_dAir*val_iPhi_phi
                    )
                    Ssigma_j[idx_] = c2_*part_
                    idx_ += 1

            factor_j = pi_list[j_] / (mix_pdf_val + 1e-300)
            dPsi_dmu_j *= factor_j
            dPsi_dlambda_j *= factor_j
            Ssigma_j *= factor_j

            offset_j = j_*cluster_params
            S_i[offset_j : offset_j+p] = dPsi_dmu_j
            S_i[offset_j+p : offset_j+2*p] = dPsi_dlambda_j
            fill_sigma_derivs(S_i, offset_j, Ssigma_j)

        if g > 1:
            f_clust = np.zeros(g)
            for j_ in range(g):
                f_clust[j_] = dmvSN_func(X[i:i+1,:], mus[j_], Sigmas[j_], lambdas[j_])[0]
            for j_ in range(g-1):
                dSipi[j_] = (1.0/(mix_pdf_val+1e-300))*(f_clust[j_] - f_clust[g-1])
            pi_offset = g*cluster_params
            S_i[pi_offset : pi_offset+(g-1)] = dSipi

        # store S_i
        if return_scores:
            S[i, :] = S_i

        # accumulate IM
        IM += np.outer(S_i, S_i)

    if ridge and ridge > 0:
        IM = IM + ridge*np.eye(M)

    return (IM, S) if return_scores else IM

def iPhi_slash(w, Ai, di, nu, rng=None, n_mc=2500):
    """
    R's I.Phi for Skew.slash:
    They do something like a for-loop:
       U <- runif(2500)
       V <- pgamma(1, w+nu, di[i]/2)*U
       S <- qgamma(V, w+nu, di[i]/2)
       Esper[i] <- mean( pnorm( sqrt(S) * Ai[i] ) )
    Then the final expression is:
       ( nu*(2^(w + nu)*Gamma(w + nu)) / (di^(w+nu)) ) * pgamma(1, w+nu, di/2 ) * Esper

    We replicate with a Monte Carlo approach.
    'rng' is a np.random.Generator for reproducibility,
    'n_mc' is the number of MC samples (2500 default).

    Ai, di are scalars for a single data point i.
    """
    # We mimic the code in R exactly:
    # 1) sample U ~ Uniform(0,1)
    if rng is None:
        rng = np.random.default_rng(1234)
    U = rng.random(n_mc)
    # 2) V = pgamma(1, shape=(w+nu), rate=di/2) * U
    #    => 'pgamma(1, a, b)' is GammaCDF(1; shape=a, rate=b)
    #    We'll do the SciPy approach: GammaCDF(1, a, scale=1/b)
    from scipy.stats import gamma as gamma_dist
    shape_ = w + nu
    rate_ = di/2.0
    # cdf_val = gamma_dist.cdf(1, a=shape_, scale=1.0/rate_)
    cdf_val = gamma_dist.cdf(1.0, a=shape_, scale=1.0/(rate_+1e-300))
    V = cdf_val*U
    # 3) S = qgamma(V, shape=w+nu, rate=di/2)
    #    => the quantile function of Gamma( shape=w+nu, rate=di/2 )
    #    => we'll invert with ppf
    #    => ppf(V, a=shape_, scale=1/rate_)
    S = gamma_dist.ppf(V, a=shape_, scale=1.0/(rate_+1e-300))
    # 4) Esper = mean( pnorm( sqrt(S)*Ai ) )
    from scipy.stats import norm
    Esper = np.mean(norm.cdf(np.sqrt(S)*Ai))

    # final factor = ( nu*(2^(w+nu)*Gamma(w+nu)) / (di^(w+nu)) ) * pgamma(1, w+nu, di/2 ) * Esper
    from math import gamma as gamma_func, pow
    val_num = nu*( (2.0**(w+nu)) * gamma_func(w + nu ) )
    val_den = (di**(w+nu) + 1e-300)
    pgamma_val = cdf_val  # we already have that above
    factor = (val_num/val_den)*pgamma_val
    return factor*Esper

def iPhi_slash_phi(w, Ai, di, nu):
    """
    R's I.phi for Skew.slash:
      ((nu*2^(w+nu)*Gamma(w+nu)) / [ sqrt(2*pi)*(di + Ai^2)^(w+nu) ]) * pgamma(1, w+nu, (di + Ai^2)/2)

    We'll skip the Monte Carlo part, it uses a closed form?
    The snippet does:
      res2 <- ((nu*2^(w + nu)*gamma(w + nu))/(sqrt(2*pi)*(di + Ai^2)^(w + nu)))*pgamma(1, w + nu, (di + Ai^2)/2)
    """
    from math import gamma as gamma_func, sqrt, pi
    from scipy.stats import gamma as gamma_dist

    val_num = ( nu*(2.0**(w+nu))*gamma_func(w+nu) )
    val_den = ( (math.sqrt(2.0*pi)) * ((di + Ai**2)**(w+nu)) )
    factor_ = val_num/(val_den+1e-300)
    shape_ = w+nu
    rate_  = (di + Ai**2)/2.0
    cdf_val = gamma_dist.cdf(1.0, a=shape_, scale=1.0/(rate_+1e-300))
    return factor_*cdf_val

def deriv_der(Dr, Dr_inv, D):
    """
    Mimic R's deriv.der(Dr, Dr.inv, D).
    = det(Dr) * sum( Dr_inv * t(D) )
    """
    return det(Dr)*np.sum(Dr_inv * D.T)

# def info_matrix_skewslash(
#     X,
#     pi_list,        # shape (g,)
#     mus,            # list of length g: mu_j
#     Sigmas,         # list of length g: Sigma_j
#     lambdas,        # list of length g: shape_j
#     nu,             # single slash param
#     d_mixedmvSS_func,  # mixture PDF for Skew-slash
#     dmvSS_func,        # single-component Skew-slash PDF
#     rng=None,
#     n_mc_nu=8000,
#     g=None,
#     p=None
# ):
#     """
#     Observed information matrix for a mixture of Skew-Slash,
#     as in the R snippet 'if (class(model) == "Skew.slash"){ ... }'.
#
#     Parameter layout per cluster j:
#       mu_j (p),
#       shape_j (p),
#       Sigma_j (p*(p+1)//2),
#     Then (g-1) for pi if g>1,
#     Then 1 for nu.
#
#     We'll replicate the code referencing partial derivatives wrt each param,
#     plus a random approach for partial wrt nu (the snippet does a runif(8000) approach).
#
#     Returns
#     -------
#     IM : (M, M) numpy array
#     """
#     X = np.asarray(X)
#     n = X.shape[0]
#     # cluster_params = 2p + p(p+1)/2
#     cluster_params = 2*p + (p*(p+1)//2)
#     # total dimension
#     M = g*cluster_params + (g-1 if g>1 else 0) + 1
#     IM = np.zeros((M, M), dtype=float)
#
#     # We'll define a helper to place the Sigma derivs:
#     def fill_sigma_derivs(S_i, offset, Ssigma):
#         start_ = offset + 2*p
#         for k_ in range(len(Ssigma)):
#             S_i[start_ + k_] = Ssigma[k_]
#
#     # For partial wrt nu, the snippet does:
#     #   u <- runif(8000)
#     #   dPsi.dnu += mean( 2 * u^(nu -1)*(1 + nu*log(u)) * [ factor * exp(...) ] * pnorm(...) )
#     # We'll do that approach inside the cluster loop, then sum over j.
#
#     if rng is None:
#         rng = np.random.default_rng(12345)
#
#     for i in range(n):
#         S_i = np.zeros(M, dtype=float)
#         # mixture pdf at y[i]
#         mix_pdf_val = d_mixedmvSS_func(X[i:i+1,:], pi_list, mus, Sigmas, lambdas, nu)[0]
#
#         # partial wrt nu, partial wrt pi
#         dPsi_dnu = 0.0
#         dSipi = np.zeros(g-1, dtype=float)
#
#         param_index = 0
#         for j_ in range(g):
#             mu_j = mus[j_]
#             Sigma_j = Sigmas[j_]
#             lam_j = lambdas[j_]
#             diff = X[i] - mu_j
#             Dr = sqrtm(Sigma_j + 1e-15*np.eye(p))
#             Dr_inv = inv(Dr)
#             d_sig = det(Sigma_j)
#
#             Ai = lam_j @ Dr_inv @ diff
#             di = diff @ inv(Sigma_j) @ diff
#
#             # partial wrt mu_j => dPsi.dmu
#             dir_dmu = -2.0*(Dr_inv@Dr_inv)@diff.reshape(-1,1)
#             dir_dmu = dir_dmu.ravel()
#             dAir_dmu = -Dr_inv@lam_j.reshape(-1,1)
#             dAir_dmu = dAir_dmu.ravel()
#
#             # iPhi slash: iPhi_slash_phi => I.phi((p+1)/2), iPhi_slash => I.Phi(p/2+1) => uses MC approach
#             val_iphi = iPhi_slash_phi((p+1)/2.0, Ai, di, nu)
#             val_iPhi = iPhi_slash((p/2.0)+1.0, Ai, di, nu, rng=rng, n_mc=2500)
#
#             c_ = 2.0*(d_sig**-0.5)/((2.0*math.pi)**(p/2.0))
#             dPsi_dmu_j = c_*( dAir_dmu*val_iphi - 0.5*dir_dmu*val_iPhi )
#
#             # partial wrt shape_j => dPsi.dlambda
#             dAir_dlambda = Dr_inv@diff.reshape(-1,1)
#             dAir_dlambda = dAir_dlambda.ravel()
#             dPsi_dlambda_j = c_*( dAir_dlambda*val_iphi )
#
#             # partial wrt Sigma => for each element using a double loop
#             Ssigma_j = np.zeros(p*(p+1)//2, dtype=float)
#             # We'll define iPhi(p/2), iPhi(p/2+1), iPhi slash phi((p+1)/2)
#             val_iPhi_p2   = iPhi_slash(p/2.0, Ai, di, nu, rng=rng, n_mc=2500)
#             val_iPhi_p2p1 = iPhi_slash(p/2.0+1.0, Ai, di, nu, rng=rng, n_mc=2500)
#             val_iphi_p1   = iPhi_slash_phi((p+1)/2.0, Ai, di, nu)
#
#             idx_ = 0
#             for l_ in range(p):
#                 for m_ in range(l_, p):
#                     D = np.zeros((p,p), dtype=float)
#                     D[l_,m_] = 1.0
#                     D[m_,l_] = 1.0
#                     Dr_det = det(Dr)
#                     val_ddet = -(1.0/(Dr_det**2))*deriv_der(Dr, Dr_inv, D)
#                     left_mat = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
#                     val_dir = -(diff.reshape(1,-1)@left_mat@diff.reshape(-1,1))[0,0]
#                     val_dAir = -(lam_j.reshape(1,-1)@Dr_inv@D@Dr_inv@diff.reshape(-1,1))[0,0]
#
#                     c2_ = 2.0/((2.0*math.pi)**(p/2.0))
#                     part_ = ( val_ddet*val_iPhi_p2
#                               - 0.5*val_dir*(d_sig**-0.5)*val_iPhi_p2p1
#                               + (d_sig**-0.5)*val_dAir*val_iphi_p1 )
#                     dPsi_dsigma_k = c2_*part_
#                     Ssigma_j[idx_] = dPsi_dsigma_k
#                     idx_ += 1
#
#             # multiply each partial by factor_j = pi_j / mixture_pdf
#             factor_j = pi_list[j_]/(mix_pdf_val+1e-300)
#             dPsi_dmu_j      = factor_j*dPsi_dmu_j
#             dPsi_dlambda_j  = factor_j*dPsi_dlambda_j
#             Ssigma_j        = factor_j*Ssigma_j
#
#             # partial wrt nu => snippet uses:
#             #  u <- runif(8000)
#             #  dPsi.dnu += mean( 2*u^(nu-1)*(1+nu*log(u))* [ factor * exp(...) ] * pnorm(...) )
#             # We'll replicate.
#             # The R code factors out (det(Sigma_j)^-1/2 / (2*pi)^(p/2)) * exp( -(u^(-1)/2)*di ).
#             # We'll define:
#             #   c3_ = pi_j * mean( ... ) / mixture_pdf
#             # We'll do it after we see the code:
#
#             # R code:
#             # u <- runif(8000)
#             # dPsi.dnu <- dPsi.dnu + pii[j]*mean( 2*u^(nu-1)*(1+nu*log(u))*[ factor * exp(...) ] * pnorm(u^(1/2)*Ai) )
#
#             # We'll define an 8k MC approach:
#             u_arr = rng.random(n_mc_nu)
#             # factor = (d_sig^-0.5 / (2*pi)^(p/2))*exp( -(u^(-1)/2)*di )
#             factor_ = ( (d_sig**-0.5)/((2.0*math.pi)**(p/2.0)) )*np.exp( -0.5*(u_arr**-1)*di )
#             # partial = 2*u^(nu-1)*(1 + nu*log(u)) * factor_ * pnorm(u^(1/2)*Ai)
#             from scipy.stats import norm
#             A_ = np.sqrt(u_arr)*Ai
#             partial_nu = 2.0*(u_arr**(nu - 1.0))*(1.0 + nu*np.log(u_arr))*factor_*norm.cdf(A_)
#             # average
#             mc_val = partial_nu.mean()
#             dPsi_dnu_j = pi_list[j_]*mc_val  # then * (1.0/(mix_pdf_val)) after sum
#             dPsi_dnu += dPsi_dnu_j
#
#             # place partial wrt mu, shape, sigma
#             offset_j = j_*cluster_params
#             S_i[offset_j : offset_j+p]               = dPsi_dmu_j
#             S_i[offset_j + p : offset_j + 2*p]       = dPsi_dlambda_j
#             fill_sigma_derivs(S_i, offset_j, Ssigma_j)
#
#         # partial wrt pi => (g-1)
#         if g>1:
#             # compute single-comp pdf for each cluster j
#             f_clust = np.zeros(g)
#             for j_ in range(g):
#                 pdf_j_ = dmvSS_func(X[i:i+1,:], mus[j_], Sigmas[j_], lambdas[j_], nu)
#                 f_clust[j_] = pdf_j_[0]
#             for j_ in range(g-1):
#                 dSipi[j_] = (1.0/(mix_pdf_val+1e-300))*( f_clust[j_] - f_clust[g-1] )
#             pi_offset = g*cluster_params
#             S_i[ pi_offset : pi_offset+(g-1) ] = dSipi
#
#         # partial wrt nu => place last
#         offset_nu = g*cluster_params + (g-1 if g>1 else 0)
#         S_i[offset_nu] = dPsi_dnu/(mix_pdf_val + 1e-300)
#
#         # accumulate
#         IM += np.outer(S_i, S_i)
#
#     return IM


def info_matrix_skewslash(
    X,
    pi_list,        # shape (g,)
    mus,            # list of length g: mu_j
    Sigmas,         # list of length g: Sigma_j
    lambdas,        # list of length g: shape_j
    nu,             # single slash param
    d_mixedmvSS_func,  # mixture PDF for Skew-slash
    dmvSS_func,        # single-component Skew-slash PDF
    rng=None,
    n_mc_nu=8000,
    g=None,
    p=None,
    *,
    return_scores: bool = False,
    ridge: float = 0.0
):
    """
    Observed information matrix for a mixture of Skew-Slash.

    If return_scores=True:
        returns (IM, S) where S is (n, M) score matrix.
    Otherwise:
        returns IM only (backward compatible).
    """
    X = np.asarray(X)
    n = X.shape[0]

    # cluster_params = 2p + p(p+1)//2
    cluster_params = 2*p + (p*(p+1)//2)
    M = g*cluster_params + (g-1 if g > 1 else 0) + 1

    IM = np.zeros((M, M), dtype=float)
    S = np.zeros((n, M), dtype=float) if return_scores else None

    # place Sigma derivatives
    def fill_sigma_derivs(S_i, offset, Ssigma):
        start_ = offset + 2*p
        S_i[start_ : start_ + len(Ssigma)] = Ssigma

    if rng is None:
        rng = np.random.default_rng(12345)

    for i in range(n):

        S_i = np.zeros(M, dtype=float)
        mix_pdf_val = d_mixedmvSS_func(
            X[i:i+1, :], pi_list, mus, Sigmas, lambdas, nu
        )[0]

        dPsi_dnu = 0.0
        dSipi = np.zeros(g-1, dtype=float)

        for j_ in range(g):

            mu_j = mus[j_]
            Sigma_j = Sigmas[j_]
            lam_j = lambdas[j_]

            diff = X[i] - mu_j
            Dr = sqrtm(Sigma_j + 1e-15*np.eye(p))
            Dr_inv = inv(Dr)
            d_sig = det(Sigma_j)

            Ai = lam_j @ Dr_inv @ diff
            di = diff @ inv(Sigma_j) @ diff

            # ---- mu ----
            dir_dmu = (-2.0*(Dr_inv@Dr_inv)@diff.reshape(-1,1)).ravel()
            dAir_dmu = (-(Dr_inv@lam_j.reshape(-1,1))).ravel()

            val_iphi = iPhi_slash_phi((p+1)/2.0, Ai, di, nu)
            val_iPhi = iPhi_slash((p/2.0)+1.0, Ai, di, nu,
                                   rng=rng, n_mc=2500)

            c_ = 2.0*(d_sig**-0.5)/((2.0*np.pi)**(p/2.0))
            dPsi_dmu_j = c_*(dAir_dmu*val_iphi - 0.5*dir_dmu*val_iPhi)

            # ---- lambda ----
            dAir_dlambda = (Dr_inv@diff.reshape(-1,1)).ravel()
            dPsi_dlambda_j = c_*(dAir_dlambda*val_iphi)

            # ---- Sigma ----
            Ssigma_j = np.zeros(p*(p+1)//2, dtype=float)

            val_iPhi_p2   = iPhi_slash(p/2.0, Ai, di, nu, rng=rng, n_mc=2500)
            val_iPhi_p2p1 = iPhi_slash(p/2.0+1.0, Ai, di, nu, rng=rng, n_mc=2500)
            val_iphi_p1   = iPhi_slash_phi((p+1)/2.0, Ai, di, nu)

            idx_ = 0
            for l_ in range(p):
                for m_ in range(l_, p):

                    D = np.zeros((p,p))
                    D[l_, m_] = 1.0
                    D[m_, l_] = 1.0

                    Dr_det = det(Dr)
                    val_ddet = -(1.0/(Dr_det**2))*deriv_der(Dr, Dr_inv, D)

                    left_mat = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
                    val_dir = -(diff.reshape(1,-1)@left_mat@diff.reshape(-1,1))[0,0]

                    val_dAir = -(lam_j.reshape(1,-1)@Dr_inv@D@Dr_inv@diff.reshape(-1,1))[0,0]

                    c2_ = 2.0/((2.0*np.pi)**(p/2.0))
                    part_ = (
                        val_ddet*val_iPhi_p2
                        - 0.5*val_dir*(d_sig**-0.5)*val_iPhi_p2p1
                        + (d_sig**-0.5)*val_dAir*val_iphi_p1
                    )

                    Ssigma_j[idx_] = c2_*part_
                    idx_ += 1

            # ---- scale by mixture weight ----
            factor_j = pi_list[j_] / (mix_pdf_val + 1e-300)

            dPsi_dmu_j     *= factor_j
            dPsi_dlambda_j *= factor_j
            Ssigma_j       *= factor_j

            # ---- nu (MC) ----
            u_arr = rng.random(n_mc_nu)
            factor_ = ((d_sig**-0.5)/((2.0*np.pi)**(p/2.0))) \
                      * np.exp(-0.5*(u_arr**-1)*di)

            from scipy.stats import norm
            A_ = np.sqrt(u_arr)*Ai

            partial_nu = (
                2.0*(u_arr**(nu-1.0))
                * (1.0 + nu*np.log(u_arr))
                * factor_
                * norm.cdf(A_)
            )

            dPsi_dnu += pi_list[j_] * partial_nu.mean()

            # ---- place block ----
            offset_j = j_*cluster_params

            S_i[offset_j : offset_j+p] = dPsi_dmu_j
            S_i[offset_j+p : offset_j+2*p] = dPsi_dlambda_j
            fill_sigma_derivs(S_i, offset_j, Ssigma_j)

        # ---- pi ----
        if g > 1:
            f_clust = np.zeros(g)
            for j_ in range(g):
                f_clust[j_] = dmvSS_func(
                    X[i:i+1,:], mus[j_], Sigmas[j_], lambdas[j_], nu
                )[0]

            for j_ in range(g-1):
                dSipi[j_] = (1.0/(mix_pdf_val+1e-300)) \
                            * (f_clust[j_] - f_clust[g-1])

            pi_offset = g*cluster_params
            S_i[pi_offset : pi_offset+(g-1)] = dSipi

        # ---- nu slot ----
        offset_nu = g*cluster_params + (g-1 if g>1 else 0)
        S_i[offset_nu] = dPsi_dnu/(mix_pdf_val + 1e-300)

        # ---- store ----
        if return_scores:
            S[i, :] = S_i

        IM += np.outer(S_i, S_i)

    if ridge and ridge > 0:
        IM = IM + ridge*np.eye(M)

    return (IM, S) if return_scores else IM



def iPhi_scn(w, Ai, di, nu):
    """
    R's I.Phi for Skew.cn:
      I.Phi(w, Ai, di, nu) = sqrt(2*pi) * [
         nu[1]*nu[2]^(w - 0.5)* dnorm( sqrt(di), 0, sqrt(1/nu[2]) ) * pnorm( sqrt(nu[2])*Ai )
         + (1 - nu[1])* dnorm( sqrt(di), 0, 1 ) * pnorm(Ai )
      ]

    Typically the code calls this with w = p/2 or w = p/2 + 1, but we keep w general.
    Ai, di are scalars for a single data point i.
    nu = [nu1, nu2].
    """
    nu1, nu2 = nu[0], nu[1]
    term1 = nu1*(nu2**(w-0.5))*norm.pdf( sqrt(di), loc=0, scale=(1.0/sqrt(nu2)) )*norm.cdf( sqrt(nu2)*Ai )
    term2 = (1.0 - nu1)*norm.pdf( sqrt(di), loc=0, scale=1.0 )*norm.cdf(Ai)
    val = sqrt(2.0*pi)*(term1 + term2)
    return val

def iPhi_scn_phi(w, Ai, di, nu):
    """
    R's I.phi for Skew.cn:
      I.phi(w, Ai, di, nu) =
        nu[1]*nu[2]^(w - 0.5)* dnorm( sqrt(di + Ai^2 ), 0, sqrt(1/nu[2]) )
        + (1 - nu[1]) * dnorm( sqrt(di + Ai^2), 0, 1 ).

    Usually called with w=(p+1)/2, but we keep w general.
    """
    nu1, nu2 = nu[0], nu[1]
    term1 = nu1*(nu2**(w-0.5))*norm.pdf( sqrt(di + Ai**2), loc=0, scale=(1.0/sqrt(nu2)) )
    term2 = (1.0 - nu1)*norm.pdf( sqrt(di + Ai**2), loc=0, scale=1.0 )
    return term1 + term2

def deriv_der(Dr, Dr_inv, D):
    """
    Mimic R's deriv.der(Dr, Dr.inv, D) = det(Dr)* sum(Dr.inv * t(D))
    Typically used to compute partial derivatives wrt sqrt(Sigma).
    """
    return det(Dr)*np.sum(Dr_inv * D.T)


def matrix_sqrt(Sig):
    """Compute symmetric matrix sqrt of Sig."""
    return sqrtm(Sig)

def deriv_der(Dr, Dr_inv, D):
    """
    Mimic R's deriv.der(Dr, Dr.inv, D).
    R used:  deriv.der(A,B,C) = det(A)*sum(B * t(C))
    """
    return det(Dr)*np.sum(Dr_inv * D.T)

def iPhi_t(w, Ai, di, nu, p):
    """
    R's  I.Phi <- function(w=0,Ai=NULL,di,nu=0)
    =  2^w * nu^(nu/2)*Gamma(w + nu/2) / [Gamma(nu/2)*(nu + di)^(nu/2 + w)]
       * pt(  (Ai / sqrt(nu+di)) * sqrt(2w + nu), df=(2w + nu) )
    """
    # We replicate exactly:
    # If Ai, di are scalars, we do scalar; if they are arrays, do elementwise.
    # Here, we assume Ai, di are scalars for each data i.
    # w is typically p/2 or (p+2)/2, etc.
    # We import needed from math/special if you want log gamma, but let's do direct:
    from math import gamma
    from scipy.stats import t as student_t

    # This is for a single i, so Ai, di are scalars:
    val1 = (2.0**w)*(nu**(nu/2.0))*gamma(w + nu/2.0)
    # denom
    from math import pow
    denom = (gamma(nu/2.0)*((nu + di)**( (nu/2.0) + w )))
    # factor
    factor = val1/denom
    # next the T cdf:
    # pt( (Ai/sqrt(nu+di))* sqrt(2w + nu), df=(2w + nu))
    df_ = 2.0*w + nu
    scale_ = np.sqrt(2.0*w + nu)
    arg_ = (Ai/np.sqrt(nu + di))*scale_
    cdf_val = student_t.cdf(arg_, df=df_)
    return factor*cdf_val

def iPhi_t_phi(w, Ai, di, nu, p):
    """
    R's I.phi for 't':
    = ( (2^w)*(nu^(nu/2.0)) / [sqrt(2*pi)*Gamma(nu/2)] )
      * (1/(di + Ai^2 + nu))^((nu + 2w)/2) * Gamma((nu+2w)/2)
    """
    from math import gamma, sqrt, pi
    # factor outside
    val1 = ( (2.0**w)*( nu**(nu/2.0) ) / ( sqrt(2.0*pi)*gamma(nu/2.0) ) )
    # next (1/(di + Ai^2 + nu))^((nu + 2w)/2)
    power_ = ((nu + 2.0*w)/2.0)
    denom_ = (di + Ai**2 + nu)
    val2 = (1.0/denom_)**power_
    # gamma((nu + 2w)/2)
    val3 = gamma((nu + 2.0*w)/2.0)
    return val1*val2*val3

# def info_matrix_t(
#     X,
#     pi_list,      # shape (g,)
#     mus,          # list of length g, mus[j] = mu_j (p,)
#     Sigmas,       # list of length g, Sigmas[j] = (p,p)
#     nu,           # single float dof
#     dmvt_ls_func, # function(y, mu, Sigma, shape, nu) => pdf val, shape=0
#     d_mixedmvST_func, # mixture pdf function
#     g, p
# ):
#     """
#     Compute the observed info matrix for a 't' mixture, following the R snippet.
#
#     Returns:
#     --------
#     IM : (M, M) array, sum of S_i S_i^T for i=1..n
#     The parameter order is:
#       for j=1..g:
#         [ mu_j(1..p), Sigma_j( (p*(p+1))/2 ) ],
#       plus if g>1 => (g-1) pi_j's,
#       plus 1 => nu
#     We'll build each S_i in that order.
#     """
#     n = X.shape[0]
#
#     # dimension of parameters per cluster
#     # p for mu_j, no shape, plus p*(p+1)//2 for Sigma_j => cluster_params = p + (p*(p+1)//2)
#     cluster_params = p + (p*(p+1)//2)
#     # plus (g-1) for pi, plus 1 for nu
#     M = g*cluster_params + (g-1 if g>1 else 0) + 1
#
#     IM = np.zeros((M, M), dtype=float)
#
#     # We'll define a small helper to convert partial derivatives of Sigma into the correct slot in S
#     # We'll keep track of param offsets:
#     # cluster_offset_j = j*cluster_params if 0-based indexing. We'll accumulate in S_i.
#
#     def fill_sigma_derivs(S_i, offset, Ssigma):
#         """
#         Place the (p*(p+1)//2) elements from Ssigma into S_i starting at offset+p.
#         Because offset+p is after the p mu-params for cluster j.
#         """
#         for k_ in range(len(Ssigma)):
#             S_i[offset + p + k_] = Ssigma[k_]
#
#     # We will store partial derivatives for each i in a vector S_i, accumulate outer products.
#     from math import log
#     # random generator for gamma draws:
#     rng = np.random.default_rng(12345)
#
#     for i in range(n):
#         # The gradient for observation i, length M
#         S_i = np.zeros(M, dtype=float)
#
#         # We'll compute partial wrt mu_j, Sigma_j for j=1..g, then gather them.
#         # Then at the end, partial wrt pi_j (g-1 of them), partial wrt nu.
#
#         # We'll keep a running offset
#         param_index = 0
#
#         # We'll accumulate dPsi.dnu across j
#         dPsi_dnu = 0.0
#         # We'll store partial wrt pi_j in a separate array, then add them after
#         dSipi = np.zeros(g-1, dtype=float)  # if g==1 we won't use it
#
#         # Evaluate mixture pdf at y[i] once
#         mix_pdf_val = d_mixedmvST_func(X[i:i+1,:], pi_list, mus, Sigmas, [list(np.zeros(p))]*g, nu)[0]
#
#         for j_ in range(g):
#             # partial wrt mu_j, Sigma_j
#             mu_j = mus[j_]
#             Sigma_j = Sigmas[j_]
#             # shape_j = 0 for t
#             # 1) compute needed quantities
#             Dr = matrix_sqrt(Sigma_j)
#             Dr_inv = inv(Dr + 1e-15*np.eye(p))
#             d_sig = det(Sigma_j)
#
#             diff = X[i] - mu_j
#             Ai = float((0.0).__add__(0.0))  # shape=0 => but the code uses `lambda[[j]]%*%Dr.inv%*%diff` ??? Actually the R snippet uses lambda=???
#             # Actually in the R snippet, they do 'Ai <- t(lambda[[j]])...'.
#             # But for "t" family, shape=0, or the code might store lambda but not used.
#             # We'll do Ai=0 if shape=0. The snippet in R does 'Ai=...??'
#             # Actually it references 'lambda[[j]]' but for 't' it might be zero.
#             # We'll do shape= np.zeros(p). So let's define shape_j = 0 * p:
#             shape_j = np.zeros(p)
#             Ai = shape_j @ Dr_inv @ diff  # a scalar
#             di = (diff @ inv(Sigma_j) @ diff)
#
#             # 2) partial wrt mu_j
#             # dir.dmu = -2*(Dr_inv@Dr_inv)@(y[i]-mu_j)
#             dir_dmu = -2.0 * (Dr_inv@Dr_inv)@diff.reshape(-1,1)
#             dir_dmu = dir_dmu.ravel()
#             # dAir.dmu = -Dr_inv@lambda_j => but lambda_j=0 => => 0 vector
#             dAir_dmu = -Dr_inv@(shape_j)  # shape_j=0 => => all zero
#             # Then dPsi.dmu from snippet:
#             # dPsi.dmu = (2*d_sig^(-1/2) / (2*pi)^(p/2)) * [ dAir_dmu * I.phi(...) - (1/2)* dir_dmu * I.Phi(...) ]
#
#             # We'll define iPhi() = iPhi_t(...) for the p/2 or (p/2+1) etc.
#             # iPhi((p+1)/2, Ai, di, nu, p) => iPhi_t((p+1)/2, Ai, di, nu, p)
#             from math import pi, sqrt
#             iPHI_val = iPhi_t_phi((p+1)/2.0, Ai, di, nu, p)  # I.phi
#             iPHIbig_val = iPhi_t((p/2.0)+1.0, Ai, di, nu, p) # I.Phi => p/2+1
#             c_ = (2.0*(d_sig**-0.5))/((2.0*pi)**(p/2.0))
#
#             dPsi_dmu_j = c_*( dAir_dmu*iPHI_val - 0.5*dir_dmu*iPhi_t(p/2.0+1.0, Ai, di, nu, p) )
#             # Actually the snippet is:
#             # dPsi.dmu = c_ * ( dAir.dmu * I.phi((p+1)/2, ...) - (1/2)*dir.dmu * I.Phi((p/2)+1, ...) )
#             # where I.phi = iPhi_t_phi, I.Phi = iPhi_t
#             # shape=0 => Ai=0 => iPHI_val might be simpler but let's keep the general formula.
#
#             # Then we multiply by [pii[j]/ d.mixedmvST(yi, ...)]
#             # But we do that outside to get the final partial. We'll define:
#             denom_ = mix_pdf_val
#             factor_j = (pi_list[j_]/(denom_ + 1e-300))
#             dPsi_dmu_j = factor_j*dPsi_dmu_j  # vector length p
#
#             # 3) partial wrt Sigma_j => we have (p*(p+1)//2) partials
#             Ssigma_j = np.zeros(p*(p+1)//2, dtype=float)
#             # We'll do the same double loop l,m, etc.
#             # The snippet in R:
#             # ddet.ds = -(1/det(Dr)^2)*deriv.der(Dr, Dr_inv, D)
#             # dir.ds  = ...
#             # dAir.ds = ...
#             # dPsi.dsigma = ...
#             # Ssigma[k] = factor_j * dPsi.dsigma
#             # We'll replicate:
#
#             # We'll define:
#             # iPhi_p2  = I.Phi(p/2, Ai, di, nu)
#             iPhi_p2  = iPhi_t(p/2.0, Ai, di, nu, p)
#             iPhi_p2p1= iPhi_t(p/2.0+1, Ai, di, nu, p)
#             iPhi_phi = iPhi_t_phi((p+1)/2.0, Ai, di, nu, p)
#
#             l_ = 0
#             m_ = 0
#             idx = 0
#             for l in range(p):
#                 for m in range(l, p):
#                     # Build D
#                     D = np.zeros((p,p), dtype=float)
#                     D[l,m] = 1.0
#                     D[m,l] = 1.0
#                     ddet_ds = -(1.0/(det(Dr)**2))*deriv_der(Dr, Dr_inv, D)
#
#                     # dir.ds => - (y[i]-mu_j)^T Dr_inv (D Dr_inv + Dr_inv D) Dr_inv (y[i]-mu_j)
#                     # We'll do it carefully:
#                     left_mat = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
#                     dir_ds_val = - diff.reshape(1,-1)@left_mat@diff.reshape(-1,1)
#                     dir_ds_val = dir_ds_val[0,0]
#
#                     # dAir.ds => - t(lambda_j) Dr_inv D Dr_inv (y[i]-mu_j) => shape=0 => => 0
#                     dAir_ds_val = 0.0
#
#                     # dPsi.dsigma
#                     # = (2/(2*pi)^(p/2)) [ ddet.ds * iPhi_t(p/2, ...)
#                     #   - 0.5 * dir_ds_val*d_sig^(-1/2)* iPhi_t(p/2+1, ...)
#                     #   + d_sig^(-1/2)* dAir_ds_val * iPhi_t_phi( (p+1)/2, ...) ]
#
#                     c1 = 2.0 / ((2.0*np.pi)**(p/2.0))
#                     part = ( ddet_ds*iPhi_p2
#                            - 0.5*dir_ds_val*(d_sig**-0.5)*iPhi_p2p1
#                            + (d_sig**-0.5)*dAir_ds_val*iPhi_phi )
#                     dPsi_dsigma_k = c1*part
#                     # multiply by factor_j
#                     dPsi_dsigma_k *= factor_j
#
#                     Ssigma_j[idx] = dPsi_dsigma_k
#                     idx += 1
#
#             # 4) partial wrt nu => accumulate into dPsi_dnu
#             # The snippet uses ~ random draws from Gamma(nu/2, rate=nu/2)
#             # then 'resto' = mean( ui^(p/2)*log(ui)*exp(-ui*di/2)*pnorm( sqrt(ui)* Ai ) )
#             ui = rng.gamma(shape=nu/2.0, scale=1.0/(nu/2.0), size=10000)
#             # shape= nu/2, rate= nu/2 => scale=1/(nu/2)
#             # compute the integrand:
#             val_ = ui**(p/2.0)*np.log(ui)*np.exp(-ui*di/2.0)*norm.cdf(np.sqrt(ui)*Ai)
#             resto = np.mean(val_) if val_.size>0 else 0.0
#
#             # dPsi.dnu_j =  pii[j]* (d_sig^(-1/2)/(2*pi)^(p/2)) * [ (log(nu/2)+1 - digamma(nu/2)) * iPhi_t(p/2,...) - iPhi_t((p+2)/2,...) + resto ]
#             # We'll define iPhi_p2 = iPhi_t(p/2, Ai, di, nu, p)
#             iPhi_p2p2 = iPhi_t((p+2)/2.0, Ai, di, nu, p)
#             from math import log
#             from scipy.special import digamma
#             c_nu = ( pi_list[j_ ]*( (d_sig**-0.5)/((2.0*np.pi)**(p/2.0)) ) )
#
#             dPsi_dnu_j = c_nu * ( (log(nu/2.0)+1.0 - digamma(nu/2.0))*iPhi_p2 - iPhi_p2p2 + resto )
#
#             dPsi_dnu += dPsi_dnu_j
#
#             # 5) add partial wrt mu_j, Ssigma_j to S_i
#             # offset for cluster j
#             offset_j = j_*cluster_params
#             # first p entries => mu_j
#             S_i[offset_j : offset_j+p] = dPsi_dmu_j
#             # next p(p+1)/2 => Ssigma_j
#             fill_sigma_derivs(S_i, offset_j, Ssigma_j)
#
#         # after finishing all clusters, partial wrt mixing proportions => (g-1)
#         # partial wrt pi_j =>
#         # Sipi[j] = (1/mix_pdf_val)*( dmvt.ls(yi, mu[j], Sigma[j], shape=0, nu) - dmvt.ls(yi, mu[g], Sigma[g], shape=0, nu) ) if j < g
#         # from snippet
#         # We'll define an array of length g for the t-likelihood of cluster j:
#         f_clust = np.zeros(g)
#         for j_ in range(g):
#             f_clust[j_] = dmvt_ls_func(X[i:i+1,:], mus[j_], Sigmas[j_], np.zeros(p), nu)[0] #original 0, maybe np.zeros(p)
#
#         if g>1:
#             # we only need first (g-1)
#             for j_ in range(g-1):
#                 dSipi[j_] = (1.0/(mix_pdf_val+1e-300))*( f_clust[j_] - f_clust[g-1] )
#             # place them at the end of the cluster blocks
#             pi_offset = g*cluster_params
#             S_i[ pi_offset : pi_offset+(g-1) ] = dSipi
#
#         # partial wrt nu => place last in S_i
#         # offset_nu = g*cluster_params + (g-1 if g>1 else 0)
#         offset_nu = g*cluster_params + (g-1 if g>1 else 0)
#         S_i[offset_nu] = (1.0/(mix_pdf_val+1e-300))*dPsi_dnu
#
#         # Now accumulate outer product
#         IM += np.outer(S_i, S_i)
#
#     return IM



def info_matrix_t(
    X,
    pi_list,      # shape (g,)
    mus,          # list of length g, mus[j] = mu_j (p,)
    Sigmas,       # list of length g, Sigmas[j] = (p,p)
    nu,           # single float dof
    dmvt_ls_func, # function(y, mu, Sigma, shape, nu) => pdf val, shape=0
    d_mixedmvST_func, # mixture pdf function
    g, p,
    *,
    return_scores: bool = False,
    ridge: float = 0.0
):
    """
    Observed info matrix for a t mixture.

    If return_scores=True:
        returns (IM, S) where S is (n, M)
    Else:
        returns IM only.
    """

    X = np.asarray(X)
    n = X.shape[0]

    cluster_params = p + (p*(p+1)//2)
    M = g*cluster_params + (g-1 if g>1 else 0) + 1

    IM = np.zeros((M, M), dtype=float)
    S = np.zeros((n, M), dtype=float) if return_scores else None

    def fill_sigma_derivs(S_i, offset, Ssigma):
        S_i[offset+p : offset+p+len(Ssigma)] = Ssigma

    rng = np.random.default_rng(12345)

    for i in range(n):

        S_i = np.zeros(M, dtype=float)

        dPsi_dnu = 0.0
        dSipi = np.zeros(g-1, dtype=float)

        mix_pdf_val = d_mixedmvST_func(
            X[i:i+1,:], pi_list, mus, Sigmas,
            [list(np.zeros(p))]*g, nu
        )[0]

        for j_ in range(g):

            mu_j = mus[j_]
            Sigma_j = Sigmas[j_]

            Dr = matrix_sqrt(Sigma_j)
            Dr_inv = inv(Dr + 1e-15*np.eye(p))
            d_sig = det(Sigma_j)

            diff = X[i] - mu_j
            shape_j = np.zeros(p)
            Ai = shape_j @ Dr_inv @ diff
            di = diff @ inv(Sigma_j) @ diff

            # ---- mu ----
            dir_dmu = (-2.0*(Dr_inv@Dr_inv)@diff.reshape(-1,1)).ravel()
            dAir_dmu = -Dr_inv@shape_j

            iPhi_phi = iPhi_t_phi((p+1)/2.0, Ai, di, nu, p)
            iPhi_big = iPhi_t((p/2.0)+1.0, Ai, di, nu, p)

            c_ = 2.0*(d_sig**-0.5)/((2.0*np.pi)**(p/2.0))
            dPsi_dmu_j = c_*(dAir_dmu*iPhi_phi - 0.5*dir_dmu*iPhi_big)

            factor_j = pi_list[j_] / (mix_pdf_val + 1e-300)
            dPsi_dmu_j *= factor_j

            # ---- Sigma ----
            Ssigma_j = np.zeros(p*(p+1)//2)

            iPhi_p2   = iPhi_t(p/2.0, Ai, di, nu, p)
            iPhi_p2p1 = iPhi_t(p/2.0+1, Ai, di, nu, p)
            iPhi_phi2 = iPhi_t_phi((p+1)/2.0, Ai, di, nu, p)

            idx = 0
            for l in range(p):
                for m in range(l,p):

                    D = np.zeros((p,p))
                    D[l,m]=1.0
                    D[m,l]=1.0

                    ddet_ds = -(1.0/(det(Dr)**2))*deriv_der(Dr,Dr_inv,D)

                    left = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
                    dir_ds = -(diff.reshape(1,-1)@left@diff.reshape(-1,1))[0,0]

                    part = (
                        ddet_ds*iPhi_p2
                        -0.5*dir_ds*(d_sig**-0.5)*iPhi_p2p1
                        +0.0*iPhi_phi2
                    )

                    c1 = 2.0/((2.0*np.pi)**(p/2.0))
                    Ssigma_j[idx] = factor_j*c1*part
                    idx+=1

            # ---- nu ----
            ui = rng.gamma(shape=nu/2.0, scale=1.0/(nu/2.0), size=10000)
            val = ui**(p/2.0)*np.log(ui)*np.exp(-ui*di/2.0)*norm.cdf(np.sqrt(ui)*Ai)
            resto = val.mean()

            from scipy.special import digamma
            iPhi_p2p2 = iPhi_t((p+2)/2.0, Ai, di, nu, p)

            c_nu = pi_list[j_]*((d_sig**-0.5)/((2.0*np.pi)**(p/2.0)))
            dPsi_dnu += c_nu * (
                (np.log(nu/2.0)+1-digamma(nu/2.0))*iPhi_p2
                - iPhi_p2p2
                + resto
            )

            # ---- place ----
            offset_j = j_*cluster_params
            S_i[offset_j:offset_j+p] = dPsi_dmu_j
            fill_sigma_derivs(S_i, offset_j, Ssigma_j)

        # ---- pi ----
        f_clust = np.zeros(g)
        for j_ in range(g):
            f_clust[j_] = dmvt_ls_func(
                X[i:i+1,:], mus[j_], Sigmas[j_], np.zeros(p), nu
            )[0]

        if g>1:
            for j_ in range(g-1):
                dSipi[j_] = (f_clust[j_] - f_clust[g-1])/(mix_pdf_val+1e-300)

            pi_offset = g*cluster_params
            S_i[pi_offset:pi_offset+(g-1)] = dSipi

        # ---- nu slot ----
        offset_nu = g*cluster_params + (g-1 if g>1 else 0)
        S_i[offset_nu] = dPsi_dnu/(mix_pdf_val+1e-300)

        if return_scores:
            S[i,:] = S_i

        IM += np.outer(S_i,S_i)

    if ridge and ridge>0:
        IM = IM + ridge*np.eye(M)

    return (IM, S) if return_scores else IM



# def info_matrix_skewt(
#     X,
#     pi_list,       # shape (g,)
#     mus,           # list of length g, each mu_j is (p,)
#     Sigmas,        # list of length g, each Sigma_j is (p,p)
#     lambdas,       # list of length g, each shape_j is (p,)
#     nu,            # single float dof
#     d_mixedmvST_func, # e.g. your mixture PDF function for Skew.t
#     dmvt_ls_func,  # e.g. your Skew.t PDF function for a single component
#     g,
#     p
# ):
#     """
#     Compute the observed information matrix for a mixture of Skew.t distributions,
#     following the R snippet if (class(model) == "Skew.t"){ ... }.
#
#     Parameter ordering per cluster j:
#       mu_j (p),
#       shape_j (p),
#       Sigma_j (p*(p+1)//2),
#     Then if g>1 => (g-1) partials for pi,
#     Then 1 partial for nu.
#
#     Returns
#     -------
#     IM : (M, M) numpy array
#     """
#     n = X.shape[0]
#     # cluster_params = 2p + p(p+1)/2
#     cluster_params = 2*p + (p*(p+1)//2)
#     # total M:
#     M = g*cluster_params + (g-1 if g>1 else 0) + 1
#
#     IM = np.zeros((M, M), dtype=float)
#
#     rng = np.random.default_rng(12345)  # for gamma draws
#
#     def fill_sigma_derivs(S_i, offset, Ssigma):
#         """
#         Place the partial derivatives wrt Sigma (length p*(p+1)//2)
#         into S_i starting at offset+2p (since offset covers cluster j,
#         the first p is mu_j, the next p is shape_j).
#         """
#         start_ = offset + 2*p
#         for k_ in range(len(Ssigma)):
#             S_i[start_ + k_] = Ssigma[k_]
#
#     for i in range(n):
#         # build gradient vector S_i for observation i
#         S_i = np.zeros(M, dtype=float)
#         mix_pdf_val = d_mixedmvST_func(X[i:i+1,:], pi_list, mus, Sigmas, lambdas, nu)[0]
#
#         # accumulators
#         dPsi_dnu = 0.0
#         dSipi = np.zeros(g-1, dtype=float)
#
#         param_index = 0  # pointer in S_i
#         # For each cluster j:
#         for j_ in range(g):
#             mu_j = mus[j_]
#             Sigma_j = Sigmas[j_]
#             lam_j = lambdas[j_]  # shape
#             diff = X[i] - mu_j
#
#             Dr = sqrtm(Sigma_j + 1e-15*np.eye(p))
#             Dr_inv = inv(Dr)
#             d_sig = det(Sigma_j)
#
#             # Ai = lam_j^T Dr_inv (y_i - mu_j)
#             Ai = lam_j @ Dr_inv @ diff
#             # di = (y_i - mu_j)^T Sigma_j^-1 (y_i - mu_j)
#             di = diff @ inv(Sigma_j) @ diff
#
#             # partial wrt mu_j => dPsi.dmu
#             # R code:
#             #   dir.dmu = -2(Dr_inv@Dr_inv)(y_i-mu_j)
#             dir_dmu = -2.0*(Dr_inv@Dr_inv)@diff.reshape(-1,1)
#             dir_dmu = dir_dmu.ravel()
#             #   dAir.dmu = -Dr_inv@lam_j
#             dAir_dmu = -Dr_inv@lam_j.reshape(-1,1)
#             dAir_dmu = dAir_dmu.ravel()
#
#             # iPhi_xxx:
#             # we do iPhi_t_phi((p+1)/2, Ai, di, nu, p) => I.phi
#             # we do iPhi_t(p/2+1, ...) => I.Phi for p/2+1
#             val_iPhi_small = iPhi_t_phi((p+1)/2.0, Ai, di, nu, p)
#             val_iPhi_big   = iPhi_t((p/2.0)+1.0, Ai, di, nu, p)
#
#             c_ = (2.0*(d_sig**-0.5))/((2.0*pi)**(p/2.0))
#             dPsi_dmu_j = c_*( dAir_dmu*val_iPhi_small - 0.5*dir_dmu*val_iPhi_big )
#
#             # partial wrt shape_j => dPsi.dlambda
#             # R code:
#             #   dAir.dlambda = Dr_inv@(y_i - mu_j)
#             dAir_dlambda = Dr_inv@diff.reshape(-1,1)
#             dAir_dlambda = dAir_dlambda.ravel()
#
#             # dPsi.dlambda = c_ * dAir.dlambda * iPhi_t_phi((p+1)/2, Ai, di, nu)
#             # => c_*( dAir_dlambda * val_iPhi_small )
#             dPsi_dlambda_j = c_* ( dAir_dlambda*val_iPhi_small )
#
#             # partial wrt Sigma => same approach as R
#             Ssigma_j = np.zeros(p*(p+1)//2, dtype=float)
#             iPhi_p2     = iPhi_t(p/2.0, Ai, di, nu, p)        # I.Phi(p/2)
#             iPhi_p2p1   = iPhi_t(p/2.0+1.0, Ai, di, nu, p)    # I.Phi(p/2+1)
#             iPhi_phi    = iPhi_t_phi((p+1)/2.0, Ai, di, nu, p)# I.phi((p+1)/2)
#
#             idx_ = 0
#             for l_ in range(p):
#                 for m_ in range(l_, p):
#                     D = np.zeros((p,p), dtype=float)
#                     D[l_,m_] = 1.0
#                     D[m_,l_] = 1.0
#                     # ddet.ds
#                     val_ddet = -(1.0/(det(Dr)**2))*deriv_der(Dr, Dr_inv, D)
#                     # dir.ds
#                     left_mat = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
#                     val_dir = -(diff.reshape(1,-1)@left_mat@diff.reshape(-1,1))[0,0]
#                     # dAir.ds
#                     val_dAir = -(lam_j.reshape(1,-1)@Dr_inv@D@Dr_inv@(diff.reshape(-1,1)))[0,0]
#
#                     # dPsi.dsigma
#                     # = (2/(2*pi)^(p/2))*( val_ddet*iPhi_p2 - (1/2)*val_dir*d_sig^(-1/2)*iPhi_p2p1 + d_sig^(-1/2)*val_dAir*iPhi_phi )
#                     c2_ = 2.0/((2.0*pi)**(p/2.0))
#                     part_ = ( val_ddet*iPhi_p2
#                               - 0.5*val_dir*(d_sig**-0.5)*iPhi_p2p1
#                               + (d_sig**-0.5)*val_dAir*iPhi_phi )
#                     dPsi_dsigma_k = c2_*part_
#                     Ssigma_j[idx_] = dPsi_dsigma_k
#                     idx_ += 1
#
#             # multiply each partial by  (pi_j / mixture_pdf)
#             # factor_j = pi_j / mix_pdf_val
#             factor_j = pi_list[j_]/(mix_pdf_val + 1e-300)
#             dPsi_dmu_j      = factor_j*dPsi_dmu_j
#             dPsi_dlambda_j  = factor_j*dPsi_dlambda_j
#             Ssigma_j        = factor_j*Ssigma_j
#
#             # partial wrt nu => accumulate
#             # we do random approximation
#             ui = rng.gamma(shape=nu/2.0, scale=1.0/(nu/2.0), size=10000)
#             val_rest = ui**(p/2.0)*np.log(ui)*np.exp(-0.5*ui*di)*norm.cdf(np.sqrt(ui)*Ai)
#             resto = val_rest.mean() if val_rest.size>0 else 0.0
#
#             # dPsi.dnu_j => c3 = factor_j but also needs (d_sig^(-1/2)/(2*pi)^(p/2))
#             c3_ = pi_list[j_]*((d_sig**-0.5)/((2.0*pi)**(p/2.0)))/(mix_pdf_val + 1e-300)
#             iPhi_p2val = iPhi_t(p/2.0, Ai, di, nu, p)
#             iPhi_p2p2  = iPhi_t((p+2)/2.0, Ai, di, nu, p)
#             dnu_j = c3_*((log(nu/2.0)+1.0 - digamma(nu/2.0))*iPhi_p2val - iPhi_p2p2 + resto)
#             dPsi_dnu += dnu_j
#
#             # Now place them in S_i
#             # offset for cluster j
#             offset_j = j_*cluster_params
#             # 1) mu_j => p
#             S_i[offset_j : offset_j+p] = dPsi_dmu_j
#             # 2) shape_j => p
#             S_i[offset_j + p : offset_j + 2*p] = dPsi_dlambda_j
#             # 3) Sigma => p(p+1)//2
#             fill_sigma_derivs(S_i, offset_j, Ssigma_j)
#
#         # partial wrt pi => (g-1)
#         f_clust = np.zeros(g)
#         for j_ in range(g):
#             f_val = dmvt_ls_func(X[i:i+1,:], mus[j_], Sigmas[j_], lambdas[j_], nu)
#             f_clust[j_] = f_val[0]
#
#         if g>1:
#             for j_ in range(g-1):
#                 dSipi[j_] = (1.0/(mix_pdf_val+1e-300))*( f_clust[j_] - f_clust[g-1] )
#             # place them after the clusters
#             pi_offset = g*cluster_params
#             S_i[ pi_offset : pi_offset+(g-1) ] = dSipi
#
#         # partial wrt nu => place last
#         offset_nu = g*cluster_params + (g-1 if g>1 else 0)
#         S_i[offset_nu] = dPsi_dnu
#
#         # accumulate outer product
#         IM += np.outer(S_i, S_i)
#
#     return IM

def info_matrix_skewt(
    X,
    pi_list,          # shape (g,)
    mus,              # list length g, each (p,)
    Sigmas,           # list length g, each (p,p)
    lambdas,          # list length g, each (p,)
    nu,               # float dof
    d_mixedmvST_func, # mixture pdf
    dmvt_ls_func,     # single-comp pdf (skew-t)
    g,
    p,
    *,
    return_scores: bool = False,
    ridge: float = 0.0,
    n_mc_nu: int = 10000,
    rng=None,
):
    """
    OPG observed information (outer product of per-observation score).

    If return_scores=True:
        returns (IM, S) where S is (n, M)
    Else:
        returns IM only.
    """
    X = np.asarray(X)
    n = X.shape[0]

    cluster_params = 2 * p + (p * (p + 1) // 2)
    M = g * cluster_params + (g - 1 if g > 1 else 0) + 1  # +1 for nu

    IM = np.zeros((M, M), dtype=float)
    S = np.zeros((n, M), dtype=float) if return_scores else None

    if rng is None:
        rng = np.random.default_rng(12345)

    def fill_sigma_derivs(S_i, offset, Ssigma):
        start = offset + 2 * p
        S_i[start : start + len(Ssigma)] = Ssigma

    for i in range(n):
        S_i = np.zeros(M, dtype=float)

        mix_pdf_val = d_mixedmvST_func(
            X[i:i+1, :], pi_list, mus, Sigmas, lambdas, nu
        )[0]

        dPsi_dnu = 0.0
        dSipi = np.zeros(g - 1, dtype=float)

        for j_ in range(g):
            mu_j = mus[j_]
            Sigma_j = Sigmas[j_]
            lam_j = lambdas[j_]

            diff = X[i] - mu_j

            Dr = sqrtm(Sigma_j + 1e-15 * np.eye(p))
            Dr_inv = inv(Dr)
            d_sig = det(Sigma_j)

            Ai = lam_j @ Dr_inv @ diff
            di = diff @ inv(Sigma_j) @ diff

            # ---- mu ----
            dir_dmu = (-2.0 * (Dr_inv @ Dr_inv) @ diff.reshape(-1, 1)).ravel()
            dAir_dmu = (-Dr_inv @ lam_j.reshape(-1, 1)).ravel()

            val_iPhi_small = iPhi_t_phi((p + 1) / 2.0, Ai, di, nu, p)     # I.phi
            val_iPhi_big   = iPhi_t((p / 2.0) + 1.0, Ai, di, nu, p)       # I.Phi

            c_ = (2.0 * (d_sig ** -0.5)) / ((2.0 * np.pi) ** (p / 2.0))
            dPsi_dmu_j = c_ * (dAir_dmu * val_iPhi_small - 0.5 * dir_dmu * val_iPhi_big)

            # ---- lambda (shape) ----
            dAir_dlambda = (Dr_inv @ diff.reshape(-1, 1)).ravel()
            dPsi_dlambda_j = c_ * (dAir_dlambda * val_iPhi_small)

            # ---- Sigma ----
            Ssigma_j = np.zeros(p * (p + 1) // 2, dtype=float)

            iPhi_p2   = iPhi_t(p / 2.0, Ai, di, nu, p)
            iPhi_p2p1 = iPhi_t(p / 2.0 + 1.0, Ai, di, nu, p)
            iPhi_phi  = iPhi_t_phi((p + 1) / 2.0, Ai, di, nu, p)

            idx = 0
            for l in range(p):
                for m in range(l, p):
                    D = np.zeros((p, p), dtype=float)
                    D[l, m] = 1.0
                    D[m, l] = 1.0

                    val_ddet = -(1.0 / (det(Dr) ** 2)) * deriv_der(Dr, Dr_inv, D)

                    left_mat = Dr_inv @ (D @ Dr_inv + Dr_inv @ D) @ Dr_inv
                    val_dir = -(diff.reshape(1, -1) @ left_mat @ diff.reshape(-1, 1))[0, 0]

                    val_dAir = -(lam_j.reshape(1, -1) @ Dr_inv @ D @ Dr_inv @ diff.reshape(-1, 1))[0, 0]

                    c2_ = 2.0 / ((2.0 * np.pi) ** (p / 2.0))
                    part_ = (
                        val_ddet * iPhi_p2
                        - 0.5 * val_dir * (d_sig ** -0.5) * iPhi_p2p1
                        + (d_sig ** -0.5) * val_dAir * iPhi_phi
                    )
                    Ssigma_j[idx] = c2_ * part_
                    idx += 1

            # ---- scale by factor_j = pi_j / mix_pdf ----
            factor_j = pi_list[j_] / (mix_pdf_val + 1e-300)
            dPsi_dmu_j *= factor_j
            dPsi_dlambda_j *= factor_j
            Ssigma_j *= factor_j

            # ---- nu (MC approx) ----
            ui = rng.gamma(shape=nu / 2.0, scale=1.0 / (nu / 2.0), size=n_mc_nu)
            val_rest = ui ** (p / 2.0) * np.log(ui) * np.exp(-0.5 * ui * di) * norm.cdf(np.sqrt(ui) * Ai)
            resto = float(val_rest.mean()) if val_rest.size else 0.0

            c3_ = pi_list[j_] * ((d_sig ** -0.5) / ((2.0 * np.pi) ** (p / 2.0))) / (mix_pdf_val + 1e-300)
            iPhi_p2val = iPhi_t(p / 2.0, Ai, di, nu, p)
            iPhi_p2p2  = iPhi_t((p + 2) / 2.0, Ai, di, nu, p)
            dPsi_dnu += c3_ * ((log(nu / 2.0) + 1.0 - digamma(nu / 2.0)) * iPhi_p2val - iPhi_p2p2 + resto)

            # ---- place cluster block ----
            offset_j = j_ * cluster_params
            S_i[offset_j : offset_j + p] = dPsi_dmu_j
            S_i[offset_j + p : offset_j + 2 * p] = dPsi_dlambda_j
            fill_sigma_derivs(S_i, offset_j, Ssigma_j)

        # ---- pi block ----
        if g > 1:
            f_clust = np.zeros(g, dtype=float)
            for j_ in range(g):
                f_clust[j_] = dmvt_ls_func(X[i:i+1, :], mus[j_], Sigmas[j_], lambdas[j_], nu)[0]

            for j_ in range(g - 1):
                dSipi[j_] = (f_clust[j_] - f_clust[g - 1]) / (mix_pdf_val + 1e-300)

            pi_offset = g * cluster_params
            S_i[pi_offset : pi_offset + (g - 1)] = dSipi

        # ---- nu slot ----
        offset_nu = g * cluster_params + (g - 1 if g > 1 else 0)
        S_i[offset_nu] = dPsi_dnu

        if return_scores:
            S[i, :] = S_i

        IM += np.outer(S_i, S_i)

    if ridge and ridge > 0:
        IM = IM + ridge * np.eye(M)

    return (IM, S) if return_scores else IM

# def info_matrix_skewcn(
#     X,
#     pi_list,          # (g,) array
#     mus,              # list of length g: mu_j
#     Sigmas,           # list of length g: Sigma_j
#     lambdas,          # list of length g: shape_j
#     nu,               # array-like [nu1, nu2]
#     d_mixedmvSNC_func,# mixture PDF for Skew-cn
#     dmvSNC_func,      # single-component Skew-cn PDF
#     dmvnorm_func,     # single or multivariate normal PDF, used in partial wrt nu
#     g,
#     p
# ):
#     """
#     Approximate the observed info matrix for a mixture of Skew Contaminated Normal (Skew.cn),
#     following the R snippet.
#
#     Parameter order for cluster j:
#       mu_j => p
#       shape_j => p
#       Sigma_j => p*(p+1)//2
#     Then if g>1 => (g-1) for pi
#     Then 2 for (nu1, nu2).
#
#     We do:
#       S_i = [ partial wrt mu_1, shape_1, Sigma_1, ... mu_g, shape_g, Sigma_g, (g-1) pi_j, nu1, nu2 ]
#     Then sum S_i S_i^T over i=1..n.
#
#     Returns
#     -------
#     IM : (M, M) numpy array
#     """
#     X = np.asarray(X)
#     n = X.shape[0]
#     nu1, nu2 = nu[0], nu[1]
#
#     # cluster_params = 2p + p(p+1)//2
#     cluster_params = 2*p + (p*(p+1)//2)
#     # total M
#     M = g*cluster_params + (g-1 if g>1 else 0) + 2  # 2 for nu1, nu2
#     IM = np.zeros((M, M), dtype=float)
#
#     def fill_sigma_derivs(S_i, offset, Ssigma):
#         """
#         Places the p*(p+1)//2 partials for Sigma_j
#         after the first 2p slots for (mu_j, shape_j).
#         offset is the cluster j offset.
#         """
#         start_ = offset + 2*p
#         for k_ in range(len(Ssigma)):
#             S_i[start_ + k_] = Ssigma[k_]
#
#     for i in range(n):
#         # gradient vector for observation i
#         S_i = np.zeros(M, dtype=float)
#
#         # partial wrt nu1, nu2 accumulators
#         dPsi_dnu1 = 0.0
#         dPsi_dnu2 = 0.0
#         # partial wrt pi
#         dSipi = np.zeros(g-1, dtype=float)
#
#         # mixture pdf at y_i
#         mix_pdf_val = d_mixedmvSNC_func(X[i:i+1,:], pi_list, mus, Sigmas, lambdas, [nu1, nu2])[0]
#
#         # For each cluster j
#         for j_ in range(g):
#             mu_j = mus[j_]
#             Sigma_j = Sigmas[j_]
#             lam_j = lambdas[j_]
#             diff = X[i] - mu_j
#
#             # sqrt(Sigma_j)
#             Dr = sqrtm(Sigma_j + 1e-15*np.eye(p))
#             Dr_inv = inv(Dr)
#             d_sig = det(Sigma_j)
#             # Ai = lam_j^T Dr_inv (y_i - mu_j)
#             Ai = lam_j @ Dr_inv @ diff
#             # di = (y_i - mu_j)^T Sigma_j^-1 (y_i - mu_j)
#             di = diff @ inv(Sigma_j) @ diff
#
#             # 1) partial wrt mu_j
#             #   dir.dmu = -2*(Dr_inv@Dr_inv)(y_i - mu_j)
#             dir_dmu = -2.0*(Dr_inv@Dr_inv)@diff.reshape(-1,1)
#             dir_dmu = dir_dmu.ravel()
#             #   dAir.dmu = -Dr_inv lam_j
#             dAir_dmu = -Dr_inv@lam_j.reshape(-1,1)
#             dAir_dmu = dAir_dmu.ravel()
#
#             #   dPsi.dmu = ((2*d_sig^(-1/2))/ (2*pi)^(p/2))* [ dAir_dmu * iPhi_scn_phi(...) - 0.5*dir_dmu*iPhi_scn(...) ]
#             # from snippet:
#             #   dPsi.dmu = c_ [ dAir.dmu * I.phi(...) - (1/2)*dir.dmu*I.Phi(...) ]
#             c_ = 2.0*(d_sig**-0.5)/((2.0*pi)**(p/2.0))
#             val_iphi = iPhi_scn_phi((p+1)/2.0, Ai, di, [nu1, nu2])  # I.phi((p+1)/2)
#             val_iPhi = iPhi_scn((p/2.0)+1.0, Ai, di, [nu1, nu2])    # I.Phi(p/2+1)
#             dPsi_dmu_j = c_*( dAir_dmu*val_iphi - 0.5*dir_dmu*val_iPhi )
#
#             # 2) partial wrt shape_j (lambda_j)
#             #   dAir.dlambda = Dr_inv (y_i - mu_j)
#             dAir_dlambda = Dr_inv@diff.reshape(-1,1)
#             dAir_dlambda = dAir_dlambda.ravel()
#             dPsi_dlambda_j = c_*( dAir_dlambda*val_iphi )
#
#             # 3) partial wrt nu1, nu2
#             #   from snippet:
#             #   dPsi.dnu1 += ...
#             #   dPsi.dnu2 += ...
#             # The snippet does:
#             #   dPsi.dnu1 = dPsi.dnu1 + pii[j]*2*( dmvnorm(...) * pnorm(...) - dmvnorm(...) * pnorm(...) )
#             #   dPsi.dnu2 = dPsi.dnu1 + pii[j]* ...
#             # (Note: the snippet has what appears to be an error: "dPsi.dnu2 <- dPsi.dnu1 + ...". We'll assume it should be `dPsi.dnu2 += ...`.)
#             # We'll need dmvnorm( yi, mu_j, (1/nu2)*Sigma_j ), etc.
#             from math import exp
#             # partial wrt nu1:
#             factor_j = pi_list[j_]/(mix_pdf_val+1e-300)
#             # We'll define single Gaussian PDFs:
#             # dmvnorm_func(yi, mu_j, Sigma_j/nu2)
#             val_dmvnorm_cont = dmvnorm_func(X[i:i+1,:], mu_j, Sigma_j*(1.0/nu2))  # contaminated part
#             val_dmvnorm_main = dmvnorm_func(X[i:i+1,:], mu_j, Sigma_j)           # main part
#             # pnorm( sqrt(nu2)* Ai ), pnorm(Ai)
#             val_pnA1 = norm.cdf(sqrt(nu2)*Ai)
#             val_pnA2 = norm.cdf(Ai)
#             # then:
#             # dPsi.dnu1 => + 2*( val_dmvnorm_cont*pnorm(...) - val_dmvnorm_main*pnorm(...) )
#             # times factor_j
#             dPsi_dnu1_j = 2.0*factor_j*( val_dmvnorm_cont[0]*val_pnA1 - val_dmvnorm_main[0]*val_pnA2 )
#             dPsi_dnu1 += dPsi_dnu1_j
#
#             # partial wrt nu2 => from snippet:
#             # dPsi.dnu2 = dPsi.dnu2 + pii[j]*((nu[1]*d.sig^(-1/2)*nu[2]^(p/2))/(2*pi)^(p/2))* ...
#             # We'll build it carefully:
#             c2_ = pi_list[j_]*((nu1*(d_sig**-0.5)*(nu2**(p/2.0)) ) / ((2.0*pi)**(p/2.0)) )/(mix_pdf_val+1e-300)
#             # multiply by exp(-nu2*di/2)
#             val_exp = exp(-0.5*nu2*di)
#             # the bracket: p*nu2^(-1)*pnorm(...) + dnorm(...) Ai nu2^(-1/2) - pnorm(...) di
#             # => define A_ = sqrt(nu2)*Ai
#             A_ = sqrt(nu2)*Ai
#             part_1 = p*(nu2**-1.0)*norm.cdf(A_)
#             part_2 = norm.pdf(A_)*Ai*(nu2**-0.5)
#             part_3 = - norm.cdf(A_)*di
#             bracket_ = part_1 + part_2 + part_3
#             dPsi_dnu2_j = c2_*val_exp*( bracket_ )
#             dPsi_dnu2 += dPsi_dnu2_j
#
#             # 4) partial wrt Sigma => same approach as other families
#             Ssigma_j = np.zeros(p*(p+1)//2, dtype=float)
#             val_iPhi_p2   = iPhi_scn(p/2.0, Ai, di, [nu1, nu2])
#             val_iPhi_p2p1 = iPhi_scn(p/2.0+1.0, Ai, di, [nu1, nu2])
#             val_iphi_p1   = iPhi_scn_phi((p+1)/2.0, Ai, di, [nu1, nu2])
#
#             idx_ = 0
#             for l_ in range(p):
#                 for m_ in range(l_, p):
#                     D = np.zeros((p,p), dtype=float)
#                     D[l_,m_] = 1.0
#                     D[m_,l_] = 1.0
#                     val_ddet = -(1.0/(det(Dr)**2))*deriv_der(Dr, Dr_inv, D)
#                     # dir.ds
#                     left_mat = Dr_inv@(D@Dr_inv + Dr_inv@D)@Dr_inv
#                     val_dir = - diff.reshape(1,-1)@left_mat@diff.reshape(-1,1)
#                     val_dir = val_dir[0,0]
#                     # dAir.ds = - t(lambda[[j]])%*%Dr.inv%*%D%*%Dr.inv%*%(y[i,] - mu[[j]])
#                     val_dAir = -(lam_j.reshape(1,-1)@Dr_inv@D@Dr_inv@(diff.reshape(-1,1)))[0,0]
#
#                     c3_ = 2.0/((2.0*pi)**(p/2.0))
#                     part_ = ( val_ddet*val_iPhi_p2
#                               - 0.5*val_dir*(d_sig**-0.5)*val_iPhi_p2p1
#                               + (d_sig**-0.5)*val_dAir*val_iphi_p1 )
#                     dPsi_dsigma_k = c3_*part_
#                     Ssigma_j[idx_] = dPsi_dsigma_k
#                     idx_ += 1
#
#             # multiply each partial by factor_j = pi_j / mixture_pdf
#             factor_j2 = pi_list[j_]/(mix_pdf_val+1e-300)
#             dPsi_dmu_j      = factor_j2*dPsi_dmu_j
#             dPsi_dlambda_j  = factor_j2*dPsi_dlambda_j
#             Ssigma_j        = factor_j2*Ssigma_j
#
#             # now place these partials in S_i
#             offset_j = j_*cluster_params
#             # mu_j => p
#             S_i[offset_j : offset_j+p] = dPsi_dmu_j
#             # shape_j => p
#             S_i[offset_j + p : offset_j + 2*p] = dPsi_dlambda_j
#             # Sigma_j => p*(p+1)//2
#             fill_sigma_derivs(S_i, offset_j, Ssigma_j)
#
#         # partial wrt pi => (g-1)
#         # each partial is (1/mix_pdf)*( dmvSNC(...)_j - dmvSNC(...)_g ) for j=1..(g-1)
#         if g>1:
#             f_clust = np.zeros(g)
#             for j_ in range(g):
#                 pdf_j_ = dmvSNC_func(X[i:i+1,:], mus[j_], Sigmas[j_], lambdas[j_], [nu1, nu2])
#                 f_clust[j_] = pdf_j_[0]
#             for j_ in range(g-1):
#                 dSipi[j_] = (1.0/(mix_pdf_val+1e-300))*( f_clust[j_] - f_clust[g-1] )
#             pi_offset = g*cluster_params
#             S_i[ pi_offset : pi_offset+(g-1) ] = dSipi
#
#         # partial wrt nu1, nu2 => place at the end
#         offset_nu = g*cluster_params + (g-1 if g>1 else 0)
#         S_i[offset_nu]   = dPsi_dnu1/(mix_pdf_val+1e-300)
#         S_i[offset_nu+1] = dPsi_dnu2/(mix_pdf_val+1e-300)
#
#         # accumulate outer product
#         IM += np.outer(S_i, S_i)
#
#     return IM

def info_matrix_skewcn(
    X,
    pi_list,
    mus,
    Sigmas,
    lambdas,
    nu,
    d_mixedmvSNC_func,
    dmvSNC_func,
    dmvnorm_func,
    g,
    p,
    *,
    return_scores: bool = False,
    ridge: float = 0.0,
):
    """
    OPG observed information for Skew Contaminated Normal mixture.

    If return_scores=True:
        returns (IM, S) where S is (n, M)
    Else:
        returns IM only.
    """

    X = np.asarray(X)
    n = X.shape[0]
    nu1, nu2 = float(nu[0]), float(nu[1])

    cluster_params = 2 * p + (p * (p + 1) // 2)
    M = g * cluster_params + (g - 1 if g > 1 else 0) + 2

    IM = np.zeros((M, M), dtype=float)
    S = np.zeros((n, M), dtype=float) if return_scores else None

    def fill_sigma_derivs(S_i, offset, Ssigma):
        start = offset + 2 * p
        S_i[start:start + len(Ssigma)] = Ssigma

    for i in range(n):

        S_i = np.zeros(M, dtype=float)

        mix_pdf_val = d_mixedmvSNC_func(
            X[i:i+1, :], pi_list, mus, Sigmas, lambdas, [nu1, nu2]
        )[0] + 1e-300

        dPsi_dnu1 = 0.0
        dPsi_dnu2 = 0.0
        dSipi = np.zeros(g - 1)

        for j_ in range(g):

            mu_j = mus[j_]
            Sigma_j = Sigmas[j_]
            lam_j = lambdas[j_]
            diff = X[i] - mu_j

            Dr = sqrtm(Sigma_j + 1e-15 * np.eye(p))
            Dr_inv = inv(Dr)
            invSigma = inv(Sigma_j)
            d_sig = det(Sigma_j)

            Ai = lam_j @ Dr_inv @ diff
            di = diff @ invSigma @ diff

            # ----- mu -----
            dir_dmu = (-2.0 * (Dr_inv @ Dr_inv) @ diff.reshape(-1, 1)).ravel()
            dAir_dmu = (-Dr_inv @ lam_j.reshape(-1, 1)).ravel()

            c_ = 2.0 * (d_sig ** -0.5) / ((2.0 * np.pi) ** (p / 2.0))
            val_iphi = iPhi_scn_phi((p + 1) / 2.0, Ai, di, [nu1, nu2])
            val_iPhi = iPhi_scn((p / 2.0) + 1.0, Ai, di, [nu1, nu2])

            dPsi_dmu_j = c_ * (dAir_dmu * val_iphi - 0.5 * dir_dmu * val_iPhi)

            # ----- lambda -----
            dAir_dlambda = (Dr_inv @ diff.reshape(-1, 1)).ravel()
            dPsi_dlambda_j = c_ * (dAir_dlambda * val_iphi)

            # ----- Sigma -----
            Ssigma_j = np.zeros(p * (p + 1) // 2, dtype=float)

            val_iPhi_p2 = iPhi_scn(p / 2.0, Ai, di, [nu1, nu2])
            val_iPhi_p2p1 = iPhi_scn(p / 2.0 + 1.0, Ai, di, [nu1, nu2])
            val_iphi_p1 = iPhi_scn_phi((p + 1) / 2.0, Ai, di, [nu1, nu2])

            idx = 0
            for l in range(p):
                for m in range(l, p):

                    D = np.zeros((p, p), dtype=float)
                    D[l, m] = 1.0
                    D[m, l] = 1.0

                    val_ddet = -(1.0 / (det(Dr) ** 2)) * deriv_der(Dr, Dr_inv, D)

                    left_mat = Dr_inv @ (D @ Dr_inv + Dr_inv @ D) @ Dr_inv
                    val_dir = -(diff.reshape(1, -1) @ left_mat @ diff.reshape(-1, 1))[0, 0]

                    val_dAir = -(lam_j.reshape(1, -1) @ Dr_inv @ D @ Dr_inv @ diff.reshape(-1, 1))[0, 0]

                    c3_ = 2.0 / ((2.0 * np.pi) ** (p / 2.0))
                    part_ = (
                        val_ddet * val_iPhi_p2
                        - 0.5 * val_dir * (d_sig ** -0.5) * val_iPhi_p2p1
                        + (d_sig ** -0.5) * val_dAir * val_iphi_p1
                    )

                    Ssigma_j[idx] = c3_ * part_
                    idx += 1

            # scale by posterior weight
            factor_j = pi_list[j_] / mix_pdf_val
            dPsi_dmu_j *= factor_j
            dPsi_dlambda_j *= factor_j
            Ssigma_j *= factor_j

            # ----- nu1 -----
            val_dmvnorm_cont = dmvnorm_func(X[i:i+1, :], mu_j, Sigma_j / nu2)[0]
            val_dmvnorm_main = dmvnorm_func(X[i:i+1, :], mu_j, Sigma_j)[0]

            dPsi_dnu1 += 2.0 * factor_j * (
                val_dmvnorm_cont * norm.cdf(np.sqrt(nu2) * Ai)
                - val_dmvnorm_main * norm.cdf(Ai)
            )

            # ----- nu2 -----
            A_ = np.sqrt(nu2) * Ai
            c2_ = (
                pi_list[j_]
                * (nu1 * (d_sig ** -0.5) * (nu2 ** (p / 2.0)))
                / ((2.0 * np.pi) ** (p / 2.0))
                / mix_pdf_val
            )

            bracket_ = (
                p * (nu2 ** -1.0) * norm.cdf(A_)
                + norm.pdf(A_) * Ai * (nu2 ** -0.5)
                - norm.cdf(A_) * di
            )

            dPsi_dnu2 += c2_ * np.exp(-0.5 * nu2 * di) * bracket_

            # place cluster block
            offset_j = j_ * cluster_params
            S_i[offset_j:offset_j + p] = dPsi_dmu_j
            S_i[offset_j + p:offset_j + 2 * p] = dPsi_dlambda_j
            fill_sigma_derivs(S_i, offset_j, Ssigma_j)

        # ----- pi block -----
        if g > 1:
            f_clust = np.zeros(g)
            for j_ in range(g):
                f_clust[j_] = dmvSNC_func(
                    X[i:i+1, :], mus[j_], Sigmas[j_], lambdas[j_], [nu1, nu2]
                )[0]

            for j_ in range(g - 1):
                dSipi[j_] = (f_clust[j_] - f_clust[g - 1]) / mix_pdf_val

            pi_offset = g * cluster_params
            S_i[pi_offset:pi_offset + (g - 1)] = dSipi

        # ----- nu slots -----
        offset_nu = g * cluster_params + (g - 1 if g > 1 else 0)
        S_i[offset_nu] = dPsi_dnu1 / mix_pdf_val
        S_i[offset_nu + 1] = dPsi_dnu2 / mix_pdf_val

        if return_scores:
            S[i, :] = S_i

        IM += np.outer(S_i, S_i)

    if ridge > 0:
        IM += ridge * np.eye(M)

    return (IM, S) if return_scores else IM

def info_matrix(model):
    """
    Compute an approximation of the observed information matrix for a
    mixture of SMSN distributions (t, Skew.t, Skew.cn, Skew.slash, Skew.normal),
    ignoring the Normal case per your request.

    model : a fitted mixture model object (like 'SkewTMix', 'SkewNormalMix', etc.)
    X     : (n, p) data array

    Returns
    -------
    IM : (M, M) numpy array, the approximate observed information matrix
         (M = total number of parameters).
    """
    from fmvmm.mixtures.skewcontmix_smsn import dmvnorm, dmvSNC, d_mixedmvSNC
    from fmvmm.mixtures.skewslashmix_smsn import dmvSS, d_mixedmvSS
    from fmvmm.mixtures.skewnormmix_smsn import d_mixedmvSN, dmvSN
    # Convert X to array
    # X = np.asarray(X)
    X = model.data
    n, p = X.shape

    # We need p >= 2 per your R code
    if p < 2:
        raise ValueError(f"Need p >= 2, but got p={p}.")

    # Gather the family
    fam = model.family  # or however you store the distribution name
    valid_families = ["t","skew_t","skew_cn","skew_slash","skew_normal", "slash"]
    if fam not in valid_families:
        raise ValueError(f"Family {fam} not recognized or not implemented here.")


    # Number of mixture components
    g = model.n_clusters

    # Retrieve mixing proportions
    # Typically something like:
    pi_list = model.pi_new  # shape (g,) array


    alpha_list = model.alpha_new  # length g



    if fam == "t":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        nu = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        IM = info_matrix_t(
        X, pi_list, mus, Sigmas, nu, dmvt_ls, d_mixedmvST, g, p
    )



    elif fam == "skew_t":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        nu = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        IM = info_matrix_skewt(
        X, pi_list, mus, Sigmas, lambdas, nu,
        d_mixedmvST_func=d_mixedmvST, # your mixture PDF
        dmvt_ls_func=dmvt_ls,   # your single component PDF
        g=g, p=p
    )

    elif fam == "skew_cn":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        nu_vec = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        IM = info_matrix_skewcn(
        X, pi_list, mus, Sigmas, lambdas, nu_vec,
        d_mixedmvSNC_func=d_mixedmvSNC,
        dmvSNC_func=dmvSNC,
        dmvnorm_func=dmvnorm,
        g=g, p=p
    )

    elif (fam == "skew_slash") or (fam ==  "slash"):
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        nu = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        IM = info_matrix_skewslash(
        X, pi_list, mus, Sigmas, lambdas, nu,
        d_mixedmvSS_func=d_mixedmvSS,
        dmvSS_func=dmvSS,
        g=g, p=p
    )

    elif fam == "skew_normal":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        IM = info_matrix_skewnormal(
        X, pi_list, mus, Sigmas, lambdas,
        d_mixedmvSN_func=d_mixedmvSN,
        dmvSN_func=dmvSN,
        g=g,
        p=p
    )



    return IM

def standard_errors_from_info_matrix(info_matrix,
                                     ridge_factor=1e-10,
                                     use_pinv=False):
    """
    Compute standard errors from an estimated information matrix
    by inverting it to get the approximate covariance,
    then taking the square root of the diagonal.

    If the matrix is singular (or nearly so), we try a fallback:
      1. Add a small ridge (ridge_factor) to the diagonal, or
      2. Use the Moore-Penrose pseudoinverse (use_pinv=True).

    Parameters
    ----------
    info_matrix : (M, M) ndarray
        The observed information matrix for M parameters.
    ridge_factor : float
        If > 0, we attempt to invert (info_matrix + ridge_factor*I)
        if the direct inverse fails.
    use_pinv : bool
        If True, we skip the ridge approach and directly compute
        the Moore-Penrose pseudoinverse of info_matrix
        if the direct inverse fails.

    Returns
    -------
    se : (M,) ndarray
        Standard errors (sqrt of the diagonal of the covariance approximation).
    cov : (M, M) ndarray
        The approximate covariance matrix (inverse or pseudoinverse of info_matrix).

    Notes
    -----
    - If the matrix is truly singular or indefinite,
      the standard errors are only approximate or may be unreliable.
    - The order of parameters in 'se' matches the row/column order of info_matrix.

    Raises
    ------
    ValueError
        If none of the fallback methods succeed in producing an invertible result.
    """

    M = info_matrix.shape[0]
    # First attempt direct inverse
    try:
        cov = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        # Direct inverse failed
        if use_pinv:
            # Use the pseudoinverse
            cov = np.linalg.pinv(info_matrix)
        else:
            # Try adding a small ridge on the diagonal
            ridge_mat = info_matrix + ridge_factor*np.eye(M)
            # In case that also fails, we do a nested try:
            try:
                cov = np.linalg.inv(ridge_mat)
            except np.linalg.LinAlgError:
                # As a last fallback, use pseudoinverse of the ridge version
                cov = np.linalg.pinv(ridge_mat)

    # At this point, we have a 'cov' approximation
    diag_vals = np.diag(cov)
    # If any diagonal is negative or zero (from indefinite or near-singular),
    # standard errors are imaginary or zero.
    # We'll warn or just compute:
    se = np.zeros(M, dtype=float)
    for i in range(M):
        if diag_vals[i] > 0:
            se[i] = np.sqrt(diag_vals[i])
        else:
            # Negative or zero diagonal => near-singular or indefinite
            # You could set se[i] = np.nan or a large number to indicate a problem:
            se[i] = np.nan

    return se

def score_matrix(model):
    """
    Compute an approximation of the observed information matrix for a
    mixture of SMSN distributions (t, Skew.t, Skew.cn, Skew.slash, Skew.normal),
    ignoring the Normal case per your request.

    model : a fitted mixture model object (like 'SkewTMix', 'SkewNormalMix', etc.)
    X     : (n, p) data array

    Returns
    -------
    IM : (M, M) numpy array, the approximate observed information matrix
         (M = total number of parameters).
    """
    from fmvmm.mixtures.skewcontmix_smsn import dmvnorm, dmvSNC, d_mixedmvSNC
    from fmvmm.mixtures.skewslashmix_smsn import dmvSS, d_mixedmvSS
    from fmvmm.mixtures.skewnormmix_smsn import d_mixedmvSN, dmvSN
    # Convert X to array
    # X = np.asarray(X)
    X = model.data
    n, p = X.shape

    # We need p >= 2 per your R code
    if p < 2:
        raise ValueError(f"Need p >= 2, but got p={p}.")

    # Gather the family
    fam = model.family  # or however you store the distribution name
    valid_families = ["t","skew_t","skew_cn","skew_slash","skew_normal", "slash"]
    if fam not in valid_families:
        raise ValueError(f"Family {fam} not recognized or not implemented here.")


    # Number of mixture components
    g = model.n_clusters

    # Retrieve mixing proportions
    # Typically something like:
    pi_list = model.pi_new  # shape (g,) array


    alpha_list = model.alpha_new  # length g



    if fam == "t":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        nu = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        _, S= info_matrix_t(
        X, pi_list, mus, Sigmas, nu, dmvt_ls, d_mixedmvST, g, p, return_scores= True
    )



    elif fam == "skew_t":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        nu = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        _, S = info_matrix_skewt(
        X, pi_list, mus, Sigmas, lambdas, nu,
        d_mixedmvST_func=d_mixedmvST, # your mixture PDF
        dmvt_ls_func=dmvt_ls,   # your single component PDF
        g=g, p=p, return_scores= True
    )

    elif fam == "skew_cn":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        nu_vec = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        _, S = info_matrix_skewcn(
        X, pi_list, mus, Sigmas, lambdas, nu_vec,
        d_mixedmvSNC_func=d_mixedmvSNC,
        dmvSNC_func=dmvSNC,
        dmvnorm_func=dmvnorm,
        g=g, p=p, return_scores= True
    )

    elif (fam == "skew_slash") or (fam ==  "slash"):
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        nu = [alpha_list[k][3] for k in range(len(alpha_list))][0]
        _, S = info_matrix_skewslash(
        X, pi_list, mus, Sigmas, lambdas, nu,
        d_mixedmvSS_func=d_mixedmvSS,
        dmvSS_func=dmvSS,
        g=g, p=p, return_scores= True
    )

    elif fam == "skew_normal":
        mus = [alpha_list[k][0] for k in range(len(alpha_list))]
        Sigmas = [alpha_list[k][1] for k in range(len(alpha_list))]
        lambdas = [alpha_list[k][2] for k in range(len(alpha_list))]
        _, S = info_matrix_skewnormal(
        X, pi_list, mus, Sigmas, lambdas,
        d_mixedmvSN_func=d_mixedmvSN,
        dmvSN_func=dmvSN,
        g=g,
        p=p, return_scores= True
    )



    return S
