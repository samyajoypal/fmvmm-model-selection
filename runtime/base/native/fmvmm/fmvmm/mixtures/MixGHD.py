"""
Source codes are from R package MixGHD. 
Converted to python as closely as possible. Many thanks to the authors.
"""

import numpy as np
import math
from scipy.linalg import pinv, det
from scipy.special import kv  # 'kv' is the modified Bessel function of the second kind
from copy import deepcopy


def _as_scalar(value, name):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Parameter '{name}' must be scalar.")
    return float(arr.reshape(-1)[0])


def _cpl_values(par):
    cpl = par.get('cpl', [1e-10, -0.5])
    omega = max(_as_scalar(cpl[0], "omega"), 1e-10)
    lam = _as_scalar(cpl[1], "lambda")
    return omega, lam


# --------------------------------------------------------------------
# 1) HELPER / PLACEHOLDER FUNCTIONS (to be implemented or replaced)
# --------------------------------------------------------------------
def besselK_nuAsym(x, nu, k_max, expon_scaled=False, log=False):
    """
    Asymptotic expansion of the Bessel K_nu(x) function
    when BOTH 'nu' and 'x' are large, as described by
    Abramowitz & Stegun, eqn. 9.7.8 (p.378).

    R references:
      besselK.nuAsym(x, nu, k.max, expon.scaled=FALSE, log=FALSE)

    Parameters
    ----------
    x : float or array-like
        Argument of K_nu(x). Must be > 0 in typical usage.
    nu : float or array-like
        The order 'nu'. Must be large in typical usage.
    k_max : int
        Must be 0 <= k_max <= 5. Determines how many terms of the Debye
        polynomial expansion to use.
    expon_scaled : bool (default False)
        If True, uses an internal shift in 'eta' to emulate the scaling
        K_nu(x)*e^{+x}, as in some Bessel routines. 
    log : bool (default False)
        If True, return log(K_nu(x)) [asymptotic approximation].
        Otherwise return K_nu(x) directly.

    Returns
    -------
    out : float or ndarray
        The asymptotic approximation for K_nu(x). If 'log=True', 
        returns the approximation of log(K_nu(x)).

    Notes
    -----
    - The expansions used here assume x ~ O(nu), i.e. both large.
      Let z = x / nu. Then:
         K_nu(nu*z) ~ sqrt(pi / (2*nu)) * exp(-nu * eta) / (1 + z^2)^(1/4)
                       * {1 + d}  [with expansions up to k_max]
      where 'eta' and 'd' are computed from 'z' and 't = 1/sqrt(1+z^2)'.

    - The code is adapted directly from your R snippet. 
      For references on Debye expansions, see A&S or NIST DLMF.
    """
    # Basic checks
    if not (0 <= k_max <= 5):
        raise ValueError("k.max must be an integer in [0..5].")

    # Convert to arrays for possible broadcast
    x_arr = np.asarray(x, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    # Determine output shape via broadcasting
    shape = np.broadcast_shapes(x_arr.shape, nu_arr.shape)
    # Broadcast to the common shape
    x_b = np.broadcast_to(x_arr, shape)
    nu_b = np.broadcast_to(nu_arr, shape)

    # Avoid division by zero if nu is zero or extremely small
    # (Though the expansion is for large nu, we still guard numeric ops.)
    nu_b_safe = np.where(nu_b == 0.0, 1e-300, nu_b)
    z = x_b / nu_b_safe  # => z = x/nu

    # sz = sqrt(1 + z^2)
    sz = np.sqrt(1.0 + z*z)
    # t = 1 / sz
    t = 1.0 / sz

    # 'eta' depends on whether we want exponent scaling
    #   If expon_scaled => eta <- 1/(z + sz), else => eta <- sz
    # Then we also add log(z / (1 + sz)) to 'eta'.
    if expon_scaled:
        # Abramowitz & Stegun eqn suggests a shift:  (sz - z) <-> 1/(z+sz)
        eta = 1.0 / (z + sz + 1e-300)
    else:
        eta = sz

    # Add the log(...) part:  eta += log(z / (1+sz))
    # We'll do that carefully to avoid log(0).
    with np.errstate(divide='ignore', invalid='ignore'):
        eta += np.log(z / (1.0 + sz) + 1e-300)

    # We'll build 'd' from expansions of the Debye polynomials:
    #   d = sum_{k=1..k_max} [ +/- u_k(t)/nu^k ]
    #   sign is slightly different from the expansions for I_nu.
    # For k_max=0, we skip expansions => d=0
    d = np.zeros(shape, dtype=float)

    if k_max > 0:
        t2 = t*t
        # u1(t) = t*(3 - 5t^2)/24
        u1_t = t*(3.0 - 5.0*t2)/24.0
        # For K_nu, the sign is negative for the first term => - u1_t / nu
        d_kmax1 = -u1_t / nu_b_safe

        if k_max == 1:
            d = d_kmax1
        else:
            # u2(t) = t^2*(81 -462t^2 + 385t^4)/1152
            u2_t = t2*(81.0 + t2*(-462.0 + 385.0*t2))/1152.0
            d_kmax2 = (d_kmax1 + (u2_t / nu_b_safe))  # all over nu => effectively => ( -u1 + u2/nu )/ nu
            d_kmax2 /= nu_b_safe

            if k_max == 2:
                d = d_kmax2
            else:
                # u3(t)
                # from R code: u3.t <- t*t2*(30375 + t2*(-369603 + t2*(765765 - 425425 t2)))/414720
                u3_t = t*t2*(30375.0 + t2*(-369603.0 + t2*(765765.0 - 425425.0*t2))) / 414720.0
                d_kmax3 = d_kmax2 + ((-u3_t) / nu_b_safe)
                d_kmax3 /= nu_b_safe

                if k_max == 3:
                    d = d_kmax3
                else:
                    # u4(t)
                    # from R code: (4465125 + t2*(...)) / 39813120
                    t4 = t2*t2
                    u4_t = t4*(4465125.0 + t2*(-94121676.0 + t2*(349922430.0 +
                            t2*(-446185740.0 + 185910725.0*t2)))) / 39813120.0
                    d_kmax4 = d_kmax3 + (u4_t / nu_b_safe)
                    d_kmax4 /= nu_b_safe

                    if k_max == 4:
                        d = d_kmax4
                    else:
                        # u5(t)
                        # from R code: t*t4*(1519035525 + t2*(-49286948607 + ...
                        u5_t = t*t4*(1519035525.0 + t2*(-49286948607.0 +
                                t2*(284499769554.0 + t2*(-614135872350.0 + 
                                t2*(566098157625.0 -188699385875.0*t2)))))  / 6688604160.0
                        d_kmax5 = d_kmax4 + ((-u5_t) / nu_b_safe)
                        d_kmax5 /= nu_b_safe

                        # k_max == 5
                        d = d_kmax5
    
    
    # We define 'res' as log or non-log
    if log:
        # log(1 + d)
        one_plus_d = 1.0 + d
        # Guard to avoid log(1 + d) negative
        with np.errstate(invalid='ignore'):
            # log1p(d)
            log1p_d = np.log1p(d)
            # If 1+d <= 0 => it might break. In practice for large z,nu, this shouldn't happen.
            mask_neg = (one_plus_d <= 0.0)
            if np.any(mask_neg):
                # fallback: for negative, set them to a large negative or handle
                log1p_d[mask_neg] = np.nan

        # -nu*eta
        neg_nu_eta = - (nu_b * eta)
        # 0.5*(log(pi/(2*nu)) - log(sz)) => note the minus sign in eqn
        # the original formula has
        #  - (log(sz) - log(pi/(2nu))) / 2
        #  = + 0.5 * ( log(pi/(2nu)) - log(sz) )
        # We'll define => halfTerm = 0.5*[log(pi/(2*nu)) - log(sz)] 
        # => Then total = log1p_d + neg_nu_eta + halfTerm
        safe_nu_b = np.where(nu_b <= 0.0, 1e-300, nu_b)
        log_term = np.log(np.pi/(2.0*safe_nu_b) + 1e-300)
        halfTerm = 0.5*(log_term - np.log(sz + 1e-300))

        res = log1p_d + neg_nu_eta + halfTerm
    else:
        # non-log
        one_plus_d = 1.0 + d
        # We do (1+d) * exp(-nu_b * eta) * sqrt(pi/(2*nu_b * sz))
        # i.e. factor = exp(-nu_b * eta) * sqrt(pi/(2*nu_b * sz))
        safe_nu_b = np.where(nu_b <= 0.0, 1e-300, nu_b)
        factor = np.sqrt(np.pi/(2.0*safe_nu_b * sz + 1e-300)) * np.exp(-nu_b*eta)
        res = one_plus_d * factor

    return res


def Rlam(x, lam=None):
    """
    Replicates the R function:
      Rlam(x, lam) = besselK(x, nu=lam+1) / besselK(x, nu=lam)
      with a fallback to besselK.nuAsym if the ratio is inf or zero.
    """
    # Ensure x and lam are NumPy arrays for consistent indexing.
    x_arr = np.atleast_1d(x)
    if lam is None:
        lam = 0.0
    lam_arr = np.atleast_1d(lam)

    # Broadcast to a common shape
    shape = np.broadcast_shapes(x_arr.shape, lam_arr.shape)
    x_b = np.broadcast_to(x_arr, shape)
    lam_b = np.broadcast_to(lam_arr, shape)

    # Evaluate K_{lam+1}(x) and K_{lam}(x)
    v1 = kv(lam_b + 1.0, x_b)
    v0 = kv(lam_b, x_b)

    # ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        val = v1 / v0

    # Identify problematic elements
    set_mask = np.isinf(v1) | np.isinf(v0) | (v0 == 0) | (v1 == 0)
    if np.any(set_mask):
        # Fallback: use besselK_nuAsym or similar
        # from your_module import besselK_nuAsym  # or wherever you defined it

        x_set = x_b[set_mask]
        lam_set = lam_b[set_mask]

        lv1 = besselK_nuAsym(x=x_set, nu=(lam_set + 1.0),
                             k_max=4, expon_scaled=False, log=True)
        lv0 = besselK_nuAsym(x=x_set, nu=lam_set,
                             k_max=4, expon_scaled=False, log=True)
        val_set = np.exp(lv1 - lv0)
        val[set_mask] = val_set

    # If the original input x was a scalar, return a scalar.
    # Otherwise, return the array of values.
    if np.isscalar(x):
        return val.item()  # convert array of shape () to Python float
    return val

def logbesselKvFA(x, y):
    """
    R code:
      logbesselKvFA <- function(x, y) {
        val = log(besselK(x=y, nu=x, expon.scaled=FALSE))
        sun = is.infinite(val)
        val[sun] = besselK.nuAsym(x=y[sun], nu=abs(x[sun]), k.max=4, log=TRUE, expon.scaled=FALSE)
        return(val)
      }

    We interpret as log( K_{x}(y) ) => i.e. K_{nu=x}(y).
    If infinite, fallback to besselK.nuAsym(..., log=TRUE).
    """
    # Evaluate log( K_{x}(y) ) directly with SciPy
    # x => nu, y => argument
    # => kv(nu, y)
    # Because x,y can be arrays, let's handle them carefully.
    x_arr = np.array(x, ndmin=1)
    y_arr = np.array(y, ndmin=1)

    # We broadcast to a common shape:
    shape = np.broadcast_shapes(x_arr.shape, y_arr.shape)
    x_broad = np.broadcast_to(x_arr, shape)
    y_broad = np.broadcast_to(y_arr, shape)

    # Evaluate kv for each element
    kv_vals = kv(x_broad, y_broad)
    val = np.log(kv_vals + 1e-300)  # add small offset to avoid log(0)

    # Check infinite
    sun_mask = ~np.isfinite(val)
    if np.any(sun_mask):
        # fallback: besselK.nuAsym
        # we pass x=y_broad[sun_mask], nu=abs(x_broad[sun_mask])
        # but note your code does: besselK.nuAsym(x=y[sun], nu=abs(x[sun])...)
        # i.e. the argument is 'y' and the order is 'abs(x)'
        y_sub = y_broad[sun_mask]
        nu_sub = np.abs(x_broad[sun_mask])
        val_asym = besselK_nuAsym(x=y_sub, nu=nu_sub,
                                  k_max=4, expon_scaled=False, log=True)
        val[sun_mask] = val_asym

    return val

def besselKv(x, y):
    """
    R code:
      besselKv <- function(x, y) { besselK(x=y, nu=x) }
    i.e. just a direct call to 'besselK'.
    In Python, we do kv(x, y) = K_{x}(y).
    """
    return kv(x, y)



def numeric_grad(f, x, args=(), h=1e-8):
    """
    A simple numerical derivative (central difference)
    of f w.r.t. x, holding other args fixed.
    """
    return (f(x + h, *args) - f(x - h, *args)) / (2.0 * h)




def getall(x):
    """
    In your R code, 'getall(loglik[1:i])' is used in a loop condition, e.g.:
        while ((getall(loglik[1:i]) > 1) & (i < 100)) ...
    We guess it checks the *difference* or *relative difference* of the last two values.
    
    We'll define it as absolute difference of the last two elements.
    """
    x_clean = x[x != 0]  # remove any zeros if needed
    length = len(x_clean)
    if length < 2:
        return 9999  # big number so loop continues
    return abs(x_clean[-1] - x_clean[-2])


def combinewk(weights=None, label=None):
    """
    R code:
    combinewk <- function(weights=NULL, label=NULL){
        if (is.null(label)) stop('label is null')
        kw = label != 0
        for (j in 1:ncol(weights)) weights[kw,j] = (label == j)[kw]
        return(weights)
    }
    
    Translated to Python, noting 1-based cluster labels in R.
    """
    if label is None:
        raise ValueError("label is null")

    

    out = weights.copy()
    nrow, ncol = out.shape

    # Boolean array where label != 0
    known_mask = (label != 0)

    for j in range(1, ncol+1):
        # In R:  weights[kw, j] = (label==j)[kw]
        # Convert j to zero-based in Python -> j_idx = j-1
        j_idx = j - 1
        match_mask = (label == j)
        # wherever known_mask & match_mask is True => set to 1
        rows_to_update = np.where(known_mask & match_mask)[0]
        out[rows_to_update, :] = 0  # zero out entire row first
        out[rows_to_update, j_idx] = 1.0
    
    return out

def rgparGH(data, g=2, w=None, l=None):
    """
    R code:
    rgparGH <- function(data, g=2, w=NULL, l=NULL){
      if (is.null(w)) w = matrix(1/g, nrow=nrow(data), ncol=g)
      val = list()
      for (k in 1:g) val[[k]] = ipar(data=data, wt=w[,k], lc=l[k,])
      val$pi = rep(1/g, g)
      return(val)
    }

    In Python, we'll store them in a dict:
      val[k]   => cluster-specific parameters
      val['pi'] => mixture weights
    """
    X = np.asarray(data)
    n, p = X.shape

    if w is None:
        w = np.full((n, g), 1.0/g)

    val = {}
    
    for k in range(g):
        # l[k, ] => the k-th row of l in R => l is shape (g, p)
        lc_k = l[k, :] if l is not None else None
        val[k] = ipar(data=X, wt=w[:, k], lc=lc_k)

    val['pi'] = np.full(g, 1.0/g)
    return val

def ipar(data, wt=None, lc=None):
    """
    R version:
      ipar <- function(data, wt=NULL, lc=NULL){
         if (is.null(wt)) wt = rep(1,nrow(data))
         p = ncol(data)
         val = list()
         val$mu    = lc
         val$alpha = rep(0,p)
         val$sigma = diag( diag( cov.wt(data, wt=wt)$cov ) )
         ... ensure diag>0.1 ...
         val$cpl   = c(1, -1/2)
         return(val)
      }
    """
    X = np.asarray(data)
    n, p = X.shape
    
    if wt is None:
        wt = np.ones(n)
    
    # We'll build a dictionary for the return value
    val = {}
    val['mu'] = np.array(lc)  # shape (p,), as in R
    val['alpha'] = np.zeros(p)
    
    
    w_sum = np.sum(wt)
    if w_sum < 1e-12:
        w_sum = 1e-12
    
    mean_w = np.einsum('ij,i->j', X, wt) / w_sum
    X_centered = X - mean_w
    cov_w = np.einsum('i,ij,ik->jk', wt, X_centered, X_centered) / w_sum
    
    diag_cov = np.diag(np.diag(cov_w))
    
    # "if (p==1) val$sigma=var(data)"
    if p == 1:
        diag_cov = np.array([[np.var(X)]])
    else:
        # The R code ensures positivity on diagonal, at least 0.1
        for i in range(p):
            if diag_cov[i, i] < 0.1:
                diag_cov[i, i] = 0.1
    
    # "if any eigen(val$sigma)$values <= 0 ) val$sigma= diag(apply(data,2,var))"
    # We'll do a quick check:
    eigvals = np.linalg.eigvals(diag_cov)
    if np.any(eigvals <= 0):
        # fallback
        diag_cov = np.diag(np.var(X, axis=0))
    
    val['sigma'] = diag_cov
    
    # cpl = c(1, -1/2)
    val['cpl'] = [1.0, -0.5]
    return val

def igpar(data=None, g=None, method="kmeans", nr=None, label=None):
    """
    R code:
    igpar <- function(data=NULL, g=NULL, method="kmeans", nr=NULL, label=NULL){
        gpar = igpar3(data=data, g=g, n=10, method=method, nr=nr, label=label)
        return(gpar)
    }
    We'll define igpar3 as a placeholder.
    """
    return igpar3(data=data, g=g, n=10, method=method, nr=nr, label=label)

def igpar3(data=None, g=None, n=10, label=None, method="kmeans", nr=10):
    """
    R version (truncated explanation):
      - If g==1, do a "kmeans(data, 1)" or just get the global mean => then run a small EM
      - Else handle different 'method' values:
          "modelBased" => calls gpcm(...)  [ placeholder ]
          "hierarchical" => calls hclust => cutree => ...
          "random" => random initialization repeated nr times
          "kmedoids" => pam(...)
          else => kmeans(...) 
        then run a small EM loop

    We'll add placeholders for gpcm, pam, hclust, etc. Can be implemented later
    """
    X = np.asarray(data)
    nrows, p = X.shape

    # Helper placeholders for cluster code:
    def hclust_cutree(X, g):
        """
        Minimal placeholder for hierarchical clustering with Ward's method.
        In practice, you'd use e.g. scipy's linkage(..., method='ward') & fcluster(...).
        We just do a random label assignment as a dummy fallback.
        """
        # from scipy.cluster.hierarchy import linkage, fcluster
        # Z = linkage(X, method='ward')
        # clusters = fcluster(Z, t=g, criterion='maxclust')
        # return clusters  # 1-based cluster labels
        labs = np.random.randint(1, g+1, size=len(X))
        return labs
    
    def kmeans_init(X, g):
        """
        Minimal placeholder for k-means -> return (centers, labels).
        You can replace with scikit-learn's KMeans or similar.
        """
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=g, n_init=1, random_state=42).fit(X)
        return (km.cluster_centers_, km.labels_+1)  # scikit-learn gives 0-based => add 1
        # return (km.cluster_centers_, km.labels_)  # scikit-learn gives 0-based => add 1
    
    def pam(X, g):
        """
        Minimal placeholder for 'pam'. In R, pam() returns medoids, clustering, ...
        We'll just reuse k-means centers as a rough stand-in.
        """
        centers, labels = kmeans_init(X, g)
        # We'll pretend centers are "medoids."
        return {
            'medoids': centers,
            'clustering': labels
        }
    
    def gpcm(X, G, mnames):
        """
        Placeholder for 'gpcm(data, G=g, mnames=c("VVV"))' from 'modelBased'.
        We'll just mimic a result with random or KMeans approach.
        We assume it returns something with $map and $gpar[[k]]$mu
        """
        # Just do a k-means approach for demonstration
        centers, labels = kmeans_init(X, G)
        # Build a fake object
        out = {}
        out['map'] = labels  # 1-based
        out['gpar'] = []
        for k in range(G):
            out['gpar'].append({'mu': centers[k]})
        return out

    # Now replicate the logic from R:
    if g == 1:
        # "lk = kmeans(data, 1)" => effectively the global mean
        # We'll do the same with "lc = apply(data, 2, mean)"
        lc = np.mean(X, axis=0).reshape(1, p)
        l = np.ones(nrows, dtype=int)  # all in cluster 1
        # combinewk
        # from . import combinewk, rgparGH, EMgrstepGH  # or adapt import
        z = combinewk(weights=np.zeros((nrows, g)), label=l)
        gpar = rgparGH(data=X, g=g, w=z, l=lc)
        # small EM loop
        for j in range(n):
            try:
                gpar = EMgrstepGH(data=X, gpar=gpar, v=1, label=l, w=z)
            except:
                pass
        return gpar
    else:
        # method-based logic
        if method == "modelBased":
            # call gpcm
            lk = gpcm(X, G=g, mnames=["VVV"])
            l = lk['map']
            lc = lk['gpar'][0]['mu'].reshape(1, p)
            for i in range(2, g+1):
                mu_i = lk['gpar'][i-1]['mu'].reshape(1, p)
                lc = np.vstack([lc, mu_i])
            
            # from . import combinewk, rgparGH, EMgrstepGH
            z = combinewk(weights=np.zeros((nrows, g)), label=l)
            gpar = rgparGH(data=X, g=g, w=z, l=lc)
            
            for j in range(n):
                try:
                    gpar = EMgrstepGH(data=X, gpar=gpar, v=1, label=l, w=z)
                except:
                    pass
            return gpar
        
        elif method == "hierarchical":
            l = hclust_cutree(X, g)
            # compute means
            lc_list = []
            for cluster_id in range(1, g+1):
                cluster_pts = X[l==cluster_id]
                if len(cluster_pts) > 0:
                    lc_list.append(cluster_pts.mean(axis=0))
                else:
                    # fallback
                    lc_list.append(np.mean(X, axis=0))
            lc = np.vstack(lc_list)
            
            # from . import combinewk, rgparGH, EMgrstepGH
            z = combinewk(weights=np.zeros((nrows, g)), label=l)
            gpar = rgparGH(data=X, g=g, w=z, l=lc)
            for j in range(n):
                try:
                    gpar = EMgrstepGH(data=X, gpar=gpar, v=1, label=l, w=z)
                except:
                    pass
            return gpar
        
        elif method == "random":
            # repeated random assignment
            # from . import combinewk, rgparGH, EMgrstepGH, llikGH
            llkO = -np.inf
            gpar_best = None
            
            for r_i in range(nr):
                # random label in 1..g
                l_rand = np.random.randint(1, g+1, size=nrows)
                # compute means
                lc_list = []
                for cluster_id in range(1, g+1):
                    cluster_pts = X[l_rand==cluster_id]
                    if len(cluster_pts) > 0:
                        lc_list.append(cluster_pts.mean(axis=0))
                    else:
                        lc_list.append(np.mean(X, axis=0))
                lc = np.vstack(lc_list)
                
                z = combinewk(weights=np.zeros((nrows, g)), label=l_rand)
                gparO = rgparGH(data=X, g=g, w=z, l=lc)
                
                loglik = np.zeros(100)
                i = 0
                for i in range(3):
                    try:
                        gparO = EMgrstepGH(data=X, gpar=gparO, v=1, label=l_rand, w=z)
                    except:
                        pass
                    loglik[i] = llikGH(X, gparO)
                
                while (getall(loglik[:i+1]) > 1) and (i < 99):
                    i += 1
                    try:
                        gparO = EMgrstepGH(data=X, gpar=gparO, v=1, label=l_rand, w=z)
                    except:
                        pass
                    loglik[i] = llikGH(X, gparO)
                
                llk = llikGH(X, gparO)
                if llk > llkO:
                    llkO = llk
                    gpar_best = deepcopy(gparO)
            
            return gpar_best
        
        elif method == "kmedoids":
            # call pam
            # from . import pam, combinewk, rgparGH, EMgrstepGH
            lk = pam(X, g)
            lc = lk['medoids']
            l = lk['clustering']
            
            z = combinewk(weights=np.zeros((nrows, g)), label=l)
            gpar = rgparGH(data=X, g=g, w=z, l=lc)
            
            for j in range(n):
                try:
                    gpar = EMgrstepGH(data=X, gpar=gpar, v=1, label=l, w=z)
                except:
                    pass
            return gpar
        
        else:
            # default to "kmeans"
            # from . import combinewk, rgparGH, EMgrstepGH
            centers, l = kmeans_init(X, g)
            # centers => shape (g, p), l => shape (n,) w/ 1..g
            z = combinewk(weights=np.zeros((nrows, g)), label=l)
            gpar = rgparGH(data=X, g=g, w=z, l=centers)
            
            for j in range(n):
                try:
                    gpar = EMgrstepGH(data=X, gpar=gpar, v=1, label=l, w=z)
                except:
                    pass
            return gpar


def weightsGH(data=None, gpar=None, v=1):
    """
    R code:
      weightsGH <- function(data=NULL, gpar=NULL, v=1){
        G = length(gpar$pi)
        if (G>1){
          zlog = matrix(0, nrow=nrow(data), ncol=length(gpar$pi))
          for (k in 1:G) zlog[,k] = ddghypGH(..., log=TRUE)
          w = t(apply(zlog, 1, function(z,wt,v){
             fstar = v*(z + log(wt)) - max(v*(z + log(wt)))
             x = exp(fstar)
             if(sum(x)==0) x=rep(1,length(x))
             x = x/sum(x)
             return(x)
          }, wt=gpar$pi, v=v ))
        } else w=matrix(1,nrow(data), ncol=G)
        return(w)
      }

    We'll replicate that in Python, using our `ddghypGH` function for the GH log density.
    """
    X = np.asarray(data)
    n, p = X.shape
    G = len(gpar['pi'])

    if G > 1:
        zlog = np.zeros((n, G))
        for k in range(G):
            # ddghypGH with log=True
            lval = ddghypGH(X, gpar[k], log=True, invS=None)
            zlog[:, k] = lval
        
        # Now replicate 'apply(zlog, 1, function(z, wt, v){...})'
        w_rows = []
        for i in range(n):
            z = zlog[i, :]
            wt = gpar['pi']
            # fstar = v*(z + log(wt)) - max( v*(z + log(wt)) )
            z_plus_logwt = z + np.log(wt + 1e-300)
            temp = v * z_plus_logwt
            fstar = temp - np.max(temp)
            x = np.exp(fstar)
            if np.sum(x) == 0:
                x = np.ones_like(x)
            x /= np.sum(x)
            w_rows.append(x)
        w = np.vstack(w_rows)
    else:
        w = np.ones((n, G))
    return w


def MAPGH(data, gpar, label=None):
    """
    R code:
    MAPGH <- function(data, gpar, label=NULL){
      w = weightsGH(...)
      if(!is.null(label)) w = combinewk(weights=w, label=label)
      z = apply(w, 1, function(z){ z=(1:length(z))[z==max(z)]; return(z[1])})
      z = as.numeric(z)
      return(z)
    }
    """
    w = weightsGH(data=data, gpar=gpar, v=1)
    if label is not None:
        w = combinewk(weights=w, label=label)
    
    # For each row i, find the cluster k with the maximum w[i,k].
    # In R we do 1-based, but let's keep it consistent:
    # z = which max => 1-based
    n = w.shape[0]
    z = np.zeros(n, dtype=int)
    for i in range(n):
        row = w[i, :]
        kmax = np.argmax(row)  # 0-based index
        # z[i] = kmax + 1        # store 1-based
        z[i] = kmax
    return z



# --------------------------------------------------------------------
# 2) UTILITY FUNCTIONS directly translated from your R code
# --------------------------------------------------------------------

def weighted_sum(z, wt):
    """R’s weighted.sum(z, wt=...) -> sum(z*wt)."""
    return np.sum(z * wt)


def ddghypGH(x, par, log=False, invS=None):
    """
    R function ddghypGH(...) for the generalized hyperbolic distribution's density.

    Parameters
    ----------
    x   : (n, p) data array
    par : dictionary with keys ['mu', 'sigma', 'alpha', 'cpl']
          cpl[0] ~ 'omega', cpl[1] ~ 'lambda'
    log : bool, if True return log of pdf
    invS: optional inverse of sigma
    """
    # Convert to np.array if needed
    X = np.asarray(x)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, p = X.shape
    
    mu = par['mu']
    sigma = par['sigma']
    alpha = par['alpha']
    
    # ensure shapes
    mu = np.asarray(mu).ravel()        # shape (p,)
    alpha = np.asarray(alpha).ravel()  # shape (p,)
    
    omega, lam = _cpl_values(par)
    
    if invS is None:
        invS = pinv(sigma)  # pseudo-inverse if needed
    
    # pa = omega + alpha' invS alpha
    alpha_invSa = alpha @ invS @ alpha
    
    # Compute distances
    # mahalanobis(x, center=mu, invS)
    # We'll do it manually:
    X_centered = X - mu
    mx = np.einsum('ij,jk,ik->i', X_centered, invS, X_centered)
    # mx is length n, each row's Mahalanobis distance
    
    pa = omega + alpha_invSa
    pa2 = np.full(n, pa)
    kx = np.sqrt(mx * pa2)
    
    # xmu%*%invS%*%alpha => same approach:
    xmua = np.einsum('ij,jk->ik', X_centered, invS)  # shape (n,p)
    lvx3 = np.einsum('ij,j->i', xmua, alpha)         # shape (n,)
    
    # Now piece together the log of the GH density
    # R code built an array lvx with columns [ (lambda - d/2)*log(kx), log(besselK(...)) - kx, xmu%*%..., 0? ]
    # Then it sums them plus some constants in lv.
    
    lvx = np.zeros((n, 4))
    
    lvx[:, 0] = (lam - p/2.0) * np.log(kx + 1e-300)
    # besselK(kx, nu=lam - d/2) => in python: kv(lam - p/2, kx)
    # The R code used `expon.scaled=TRUE` and then subtracted kx. We replicate that:
    # => log(besselK(..., expon.scaled=TRUE)) - kx
    # We skip "expon.scaled=TRUE" because python's kv does not have that mode directly.
    
    # We can do:
    #   logBessel = np.log(kv(lam - p/2, kx) + 1e-300)
    #   lvx[:,1] = logBessel
    # but the R code subtracts kx:
    logBessel = np.log(kv(lam - p/2.0, kx) + 1e-300)
    lvx[:, 1] = logBessel - kx
    
    lvx[:, 2] = lvx3  # xmu %*% invS %*% alpha
    lvx[:, 3] = 0.0   # the code put 0 for that column, but effectively we might not need it
    
    # Summation of these is lvx_row_sum
    lvx_row_sum = np.sum(lvx, axis=1)
    
    # The 'lv' array in R has length 6 with various constants. We'll collect them similarly:
    lv = np.zeros(6)
    
    # 1) -1/2 * log(det(sigma)) - d/2 * (log(2) + log(pi))
    # watch out for det(sigma) possibly zero. Use a fallback or pinv if needed
    # or we just do math.log(det) if non-singular
    try:
        dsig = det(sigma)
        if dsig <= 0:
            dsig = 1e-300
    except:
        dsig = 1e-300
    
    lv[0] = -0.5 * math.log(dsig) - (p/2.0)*(math.log(2.0) + math.log(math.pi))
    
    # 2)  omega - log(besselK(omega, nu=lambda, expon.scaled=TRUE))
    #    again no direct 'expon.scaled' in Python
    logK_omega = np.log(kv(lam, omega) + 1e-300)
    lv[1] = omega - logK_omega
    
    # 3) -lambda/2*( log(1) )
    #   that is zero, because log(1)=0 => so no effect
    lv[2] = 0.0
    
    # 4) lambda*log(omega)*0 => also zero
    lv[3] = 0.0
    
    # 5) (d/2 - lambda)*log(pa)
    lv[4] = (p/2.0 - lam)*math.log(pa + 1e-300)
    
    # Summation of those 6 constants is sum_lv
    sum_lv = np.sum(lv)
    
    val = lvx_row_sum + sum_lv  # shape (n,)
    if not log:
        val = np.exp(val)
    return val

def gig2GH(x, par, invS=None):
    """
    R function gig2GH(). 
    Returns a matrix of shape (n, 3) with [w, invw, logw].
    Each row i is E[W], E[1/W], E[logW] for observation x_i under GIG.
    """
    X = np.asarray(x)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, d = X.shape
    
    if invS is None:
        invS = pinv(par['sigma'])
    
    alpha = par['alpha']
    alpha = np.asarray(alpha).ravel()
    
    # par$cpl[1] = "omega" in exponent form
    # omega = math.exp(math.log(par['cpl'][0]))
    omega, lam = _cpl_values(par)
    
    # a1 = omega + alpha' invS alpha
    alpha_invSa = alpha @ invS @ alpha
    a1 = omega + alpha_invSa
    
    # b1 = omega + mahalanobis(x, mu, invS)
    # we interpret 'mahalanobis(x, center=par$mu, invS=..., inverted=TRUE)'
    mu = np.asarray(par['mu']).ravel()
    Xc = X - mu
    b1 = np.einsum('ij,jk,ik->i', Xc, invS, Xc) + omega
    
    # v1 = par$cpl[2] - d/2
    v1 = lam - d/2.0
    
    # Now call gigGH(a=a1, b=b1, v=v1)
    val = gigGH(a=a1, b=b1, v=v1)
    return val

def gigGH(a=None, b=None, v=None):
    """
    R function gigGH(). 
    Returns a matrix (n, 3) with columns [w, invw, logw].
    Interprets the formula:
       w    = K_{v+1}( sqrt(a*b) ) / K_v( sqrt(a*b) ) * sqrt(b/a)   ... etc.
       invw = ...
       logw = ...
    """
    # a, b can be scalars or arrays. We’ll treat them as arrays.
    # We want elementwise operations
    if not isinstance(a, np.ndarray):
        a = np.array([a])
    if not isinstance(b, np.ndarray):
        b = np.array([b])
    
    # Ensure same shape
    # The R code expects 'length(a)' to match 'length(b)' => the number of observations
    n = max(a.size, b.size)
    A = np.full(n, a.item()) if a.size==1 else a.ravel()
    B = np.full(n, b.item()) if b.size==1 else b.ravel()
    
    sab = np.sqrt(A*B)  # shape (n,)
    # kv( nu=v+1, x=sab ) / kv( nu=v, x=sab )
    # For numeric stability, add small offset if needed.
    eps = 1e-300
    
    # We replicate R's use of 'besselK(..., expon.scaled=TRUE)' by just normal kv
    # If sab is large, might need log or scaling. For now, do direct kv:
    kv_num = kv(v+1.0, sab+eps)
    kv_den = kv(v,     sab+eps) + eps
    kv_ratio = kv_num / kv_den
    
    sb_a = np.sqrt(B/(A+eps))  # shape (n,)
    
    w    = kv_ratio * sb_a
    invw = kv_ratio*(1.0/sb_a) - 2.0*v/B  # from your R code
    # For logw, the code uses a numeric derivative approach: grad(logbesselKv, x=rep(v,...), y=sab,...) 
    # Currently your code calls 'grad(logbesselKv, ...)'. We do not replicate that here exactly.
    # We'll do a placeholder:
    logw = np.log(sb_a+eps)  # plus a derivative term that we have not fully replicated
    
    # Put together as (n,3)
    out = np.column_stack((w, invw, logw))
    return out

def llikGH(data, gpar, delta=0):
    """
    R function llikGH(). 
    Computes the log-likelihood for mixture of GH distributions in gpar.

    gpar is structured:
      gpar['pi'] : array of shape (G,)
      gpar[k]    : dict with keys ['mu','sigma','alpha','cpl'] for cluster k
    """
    n, p = data.shape
    G = len(gpar['pi'])
    
    logz = np.zeros((n, G))
    for k in range(G):
        logz[:, k] = ddghypGH(data, gpar[k], log=True, invS=None)
    
    
    total = 0.0
    for i in range(n):
        # sum(exp(z)*wt) => sum(exp(logz[i,:])* gpar$pi )
        ex = np.exp(logz[i, :])
        weighted = ex * gpar['pi']
        total += math.log(np.sum(weighted) + 1e-300)
    return total


# --------------------------------------------------------------------
# 3) The CORE EM functions: mainMGHD & EMgrstepGH & updatemaScpl
# --------------------------------------------------------------------

def mainMGHD(data=None, gpar0=None, G=None, n=None, label=None, eps=1e-5, method=None, nr=None):
    """
    R function mainMGHD().
    data  : (N, p) array
    gpar0 : initial gpar (dictionary) or None
    G     : number of clusters
    n     : maximum number of iterations? (R uses n for the loop limit)
    label : optional cluster labels
    eps   : tolerance
    method: method name
    nr    : not used except possibly for 'igpar'
    """
    # pcol = ncol(data)
    N, pcol = data.shape
    
    # If label is given, compute cluster means for initialization
    if label is not None:
        # in R code: we do an apply(data[label==1,],2,mean), etc.
        # We'll store those in lc
        unique_labels = np.unique(label)
        # Make sure labeling is consistent with G
        # The R code just picks the means for each i in 1..G
        lc = []
        for i in range(1, G+1):
            cluster_pts = data[label == i]
            if cluster_pts.shape[0] > 0:
                mean_i = cluster_pts.mean(axis=0)
            else:
                mean_i = np.zeros(pcol)
            lc.append(mean_i)
        lc = np.vstack(lc)
        
        # combinewk
        z = combinewk(weights=np.full((N, G), 1.0/G), label=label)
        # If gpar0 not given, then call rgparGH
        if gpar0 is None:
            gpar = rgparGH(data=data, g=G, w=z, l=lc)
        else:
            gpar = gpar0
    else:
        # no label
        if gpar0 is None:
            # call igpar
            gpar = igpar(data=data, g=G, method=method, nr=nr)
        else:
            gpar = gpar0
    
    loglik = np.zeros(n, dtype=float)
    
    # initial steps
    i = 0
    for i in range(3):
        gpar = EMgrstepGH(data=data, gpar=gpar, v=1, label=label, w=None)
        loglik[i] = llikGH(data, gpar)
    
    # main loop
    # while( ( getall(loglik[1:i]) > eps) & (i < n ) )
    # "getall" presumably checks difference in consecutive loglik or something. Let's replicate that logic:
    def rel_change(x):
        if len(x) < 2:
            return 999.0
        return abs(x[-1] - x[-2]) / (abs(x[-2]) + 1e-300)
    
    i = 2  # after the first 3 steps, i is at index 2
    while True:
        if rel_change(loglik[:i+1]) < eps:
            break
        if (i+1) >= (n-1):
            break
        
        i += 1
        gpar = EMgrstepGH(data=data, gpar=gpar, v=1, label=label, w=None)
        loglik[i] = llikGH(data, gpar)
    
    # if i<n => shorten loglik
    if i < (n-1):
        loglik = loglik[:i+1]
    
    # compute BIC, ICL, AIC, AIC3
    # BIC=2*loglik[i]-log(nrow(data))*((G-1)+G*(2*pcol+2+pcol*(pcol-1)/2))
    # We replicate that:
    ndata = data.shape[0]
    # number of parameters for GH mixture:
    #   (G-1) + G*(2*pcol+2 + pcol*(pcol-1)/2)
    #   = (G-1) + G*(2pcol + 2 + pcol(pcol-1)/2)
    nparams = (G - 1) + G*(2*pcol + 2 + pcol*(pcol-1)/2.0)
    
    ll_final = loglik[i]
    BIC = 2*ll_final - math.log(ndata)*nparams
    # z = weightsGH(...)
    z = weightsGH(data=data, gpar=gpar, v=1)
    # ICL = BIC + 2*sum(log(apply(z,1,max)))
    row_max = np.max(z, axis=1)
    ICL = BIC + 2.0*np.sum(np.log(row_max + 1e-300))
    
    # AIC=2*ll_final - 2*nparams
    AIC  = 2*ll_final - 2.0*nparams
    AIC3 = 2*ll_final - 3.0*nparams
    
    # map = MAPGH(...)
    map_labels = MAPGH(data=data, gpar=gpar, label=label)
    
    val = {
        'loglik': loglik,
        'gpar': gpar,
        'z': z,
        'map': map_labels,
        'BIC': BIC,
        'ICL': ICL,
        'AIC': AIC,
        'AIC3': AIC3
    }
    return val

def EMgrstepGH(data=None, gpar=None, v=1, label=None, w=None):
    """
    R function EMgrstepGH(...).
    - E-step: compute w=weightsGH(...) unless it's given
    - If label given, combinewk
    - M-step: for each cluster k, call updatemaScpl
    - Then update gpar$pi
    """
    if w is None:
        w = weightsGH(data=data, gpar=gpar, v=v)
    if label is not None:
        w = combinewk(weights=w, label=label)
    
    G = len(gpar['pi'])
    d = len(gpar[0]['mu'])  # dimension from the first component
    
    # For each cluster k in 1..G:
    for k in range(G):
        invS = None
        alpha_known = None
        gpar[k] = updatemaScpl(x=data, par=gpar[k], weights=w[:, k], invS=invS, alpha_known=alpha_known, v=v)
    
    # update gpar$pi = colMeans of w
    gpar['pi'] = np.mean(w, axis=0)
    return gpar

def updatemaScpl(x, par, weights=None, invS=None, alpha_known=None, v=None):
    """
    R function updatemaScpl(...).
    - Compute 'abc = gig2GH(...)'
    - Then update mu, alpha, sigma, cpl
    """
    X = np.asarray(x)
    n, p = X.shape
    if weights is None:
        weights = np.ones(n)
    if invS is None:
        invS = pinv(par['sigma'])
    
    abc = gig2GH(x=X, par=par, invS=invS)  # shape (n, 3) => columns [w, invw, logw]
    # Weighted average of each column:
    sumw = np.sum(weights)
    
    # ABC = apply(abc, 2, weighted.sum, wt=weights)/sumw => array( [A, B, C], shape=(3,) )
    ABC = np.array([np.sum(abc[:, j]*weights) for j in range(3)]) / sumw
    A = ABC[0]
    B = ABC[1]
    
    
    
    w_col = abc[:, 0]
    invw_col = abc[:, 1]
    # logw_col = abc[:, 2]
    
    # u = (B - abc[,2])*weights => in R, "abc[,2]" was 'invw' => so (B - invw[i])
    u = (B - invw_col) * weights
    # t = (A*abc[,2] - 1)*weights => (A*invw_col - 1)*weights
    t_ = (A*invw_col - 1.0) * weights
    T_ = np.sum(t_)
    
    if alpha_known is None:
        # mu.new
        # Weighted sum of X columns with weights t_
        mu_num = np.einsum('ij,i->j', X, t_)
        if abs(T_) < 1e-12:
            mu_new = par['mu']
        else:
            mu_new = mu_num / T_
        
        # alpha.new
        alpha_num = np.einsum('ij,i->j', X, u)
        if abs(T_) < 1e-12:
            alpha_new = par['alpha']
        else:
            alpha_new = alpha_num / T_
    else:
        alpha_new = alpha_known
        # mu.new = weighted.mean(X, w=abc[,2]*weights) - alpha.new/ABC[2]
        
        invw_weights = invw_col * weights
        sum_invw_weights = np.sum(invw_weights)
        mu_mean = np.einsum('ij,i->j', X, invw_weights) / (sum_invw_weights + 1e-12)
        mu_new = mu_mean - alpha_new / B   # since B=ABC[2] in R code? Actually B=ABC[2], but R code = invw col. Let's keep it consistent.
    
    # alpha.new = alpha.new * v => from the R code: alpha.new = alpha.new * v
    alpha_new = alpha_new * v
    
    
    w_for_cov = invw_col * weights
    sum_w_for_cov = np.sum(w_for_cov)
    
    X_centered = X - mu_new
    # Weighted cov
    if sum_w_for_cov > 1e-12:
        covML = np.einsum('i,ij,ik->jk', w_for_cov, X_centered, X_centered) / sum_w_for_cov
    else:
        covML = np.eye(p)*1e-5
    A_ = covML * B  # "A" in R code
    
    
    R = A_
    
    for i_col in range(p):
        if R[i_col, i_col] < 1e-5:
            R[i_col, i_col] = 1e-5
    
    # par.old = c( log(par$cpl[1]) , par$cpl[2] )
    # We do the same:
    cpl1, cpl2 = _cpl_values(par)
    # try:
    #     cpl1 = np.maximum(cpl1[0],1)
    # except:
    #     cpl1 = np.maximum(cpl1,1)
    # print(cpl1)
    par_old = [math.log1p(cpl1), cpl2] #log giving error when it becomes zero!
    
    
    def updateol(ol=None, ABC=None, n=1):
  
    
        if ol is None or ABC is None:
            return ol
    
        ol = list(ol)  # convert to mutable list if needed
    
        for i in range(n):
            # If ABC[2] == 0 => ol[1] = 0
            if abs(ABC[2]) < 1e-15:
                ol[1] = 0.0
            else:
                
    
                # We'll define a local function f(lam_value) => logbesselKvFA(lam_value, omega=ol[0])
                def f_for_grad(lam_value, omega_val):
                    return logbesselKvFA(lam_value, omega_val)
    
                # numeric derivative wrt lam_value
                bv = numeric_grad(f_for_grad, ol[1], args=(ol[0],))
    
                # Then: ol[2] = ABC[3]*(ol[2]/bv)
                # i.e. "lambda_new = ABC[3] * (lambda_old / bv)"
                # in R: ol[2] => lam, ABC[3] => ABC[2] in 0-based
                lam_old = ol[1]
                if abs(bv) < 1e-15:
                    # avoid division by zero
                    pass
                else:
                    ol[1] = ABC[2] * (lam_old / bv)
    
            # R code calls:
            #   Rp = Rlam(ol[1], lam=+ol[2])
            #   Rn = Rlam(ol[1], lam=-ol[2])
            # But remember in R: ol[1] is the first element => we interpret as "omega"?
            # Actually the snippet suggests "ol[1]" is "omega" and "ol[2]" is "lambda".
            # Let's keep that:
            omega_now = ol[0]
            lam_now   = ol[1]
    
            Rp = Rlam(omega_now, lam=+lam_now)
            Rn = Rlam(omega_now, lam=-lam_now)
    
            # f1 = Rp + Rn - (ABC[1] + ABC[2]) => in R => ABC[1], ABC[2]
            f1 = Rp + Rn - (ABC[0] + ABC[1])
    
            
            f2_part1 = Rp**2 - ((2*lam_now + 1)/ (omega_now+1e-15)) * Rp - 1
            f2_part2 = Rn**2 - ((2*(-1*lam_now) + 1)/ (omega_now+1e-15)) * Rn - 1
            f2_val = f2_part1 + f2_part2
    
            
            if (ol[0] - f1/(f2_val+1e-15)) > 0:
                ol[0] = ol[0] - f1/(f2_val+1e-15)
    
        return ol

    
    
    a_ = updateol(par_old, ABC=ABC, n=2)
    cpl_new = [_as_scalar(a_[0], "omega"), _as_scalar(a_[1], "lambda")]
    
    new_par = {
        'mu': mu_new,
        'alpha': alpha_new,
        'sigma': R,
        'cpl': cpl_new
    }
    return new_par
