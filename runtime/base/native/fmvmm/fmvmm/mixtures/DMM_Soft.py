import numpy as np
from scipy.special import gammaln,logsumexp
import sys

import numpy as np
from numpy import (
    asanyarray,
    exp
)
from numpy.linalg import norm
from scipy.special import polygamma, psi
from scipy.special import digamma
import pandas as pd
import copy
from fmvmm.utils.utils_dmm import (kmeans_init, gmm_init, kmeans_init_adv, gmm_init_adv, random_init,
                             dmm_loglikelihood, dmm_responsibilities, dmm_pi_estimate)

from fmvmm.utils.utils_mixture import (mixture_clusters)
import time
from multiprocessing import Pool
from fmvmm.mixtures._base import BaseMixture
from fmvmm.utils.utils_dmm import combined_info_and_se, mean_precision_inference

MAXINT = sys.maxsize
euler = -1 * psi(1)  # Euler-Mascheroni constant



def get_precision_es(data):
    # Convert the list of lists to a NumPy array
    data_array = np.array(data)
    # print(data_array.shape)

    # Calculate mean (mu_1) and variance (var_1) for the first column (axis 0)
    mu_1 = np.mean(data_array[:, 0])
    var_1 = np.var(data_array[:, 0])

    # Compute the expression
    result = (mu_1 - (var_1 + mu_1**2)) / var_1

    return result

def get_means(data):
    # Convert the list of lists to a NumPy array
    data_array = np.array(data)
    # print(data_array.shape)

    # Calculate mean
    mu = [np.mean(data_array[:, i]) for i in range(data_array.shape[1])]


    return mu


class NotConvergingError(Exception):
    """Error when a successive approximation method doesn't converge
    """
    pass


def _ipsi(y, tol=1.48e-9, maxiter=10):
    """Inverse of psi (digamma) using Newton's method. For the purposes
    of Dirichlet MLE, since the parameters a[i] must always
    satisfy a > 0, we define ipsi :: R -> (0,inf).

    Parameters
    ----------
    y : (K,) shape array
        y-values of psi(x)
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 10.

    Returns
    -------
    (K,) shape array
        Approximate x for psi(x)."""
    y = asanyarray(y, dtype="float")
    x0 = np.piecewise(
        y,
        [y >= -2.22, y < -2.22],
        [(lambda x: exp(x) + 0.5), (lambda x: -1 / (x + euler))],
    )
    for i in range(maxiter):
        x1 = x0 - (psi(x0) - y) / _trigamma(x0)
        if norm(x1 - x0) < tol:
            return x1
        x0 = x1
    # raise NotConvergingError(
    #     f"Failed to converge after {maxiter} iterations, " f"value is {x1}")
    return x1


def _trigamma(x):
    return polygamma(1, x)


def dirichlet_mix_mle(x: np.ndarray, gamma: np.ndarray, alpha_init):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    alpha_not = alpha_init
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    alpha_old = alpha_not
    alpha_all = []
    for o in range(k):

        log_x = []
        for f in range(p):
            log_x_temp = (
                1/N_js[o])*np.sum([np.log(x_lol[j][f])*gamma[j, o] for j in range(n)])
            log_x.append(log_x_temp)
        alpha_new = []
        for i in range(p):
            diff = 5
            t = copy.deepcopy(alpha_old[o][i])
            t1 = copy.deepcopy(alpha_old[o][i])
            di_alpha_not = digamma(t)
            di_alpha_old = di_alpha_not
            beta = np.sum(alpha_old[o])-t
            while diff > 0.0001:
                di_alpha_sum = digamma(np.sum(alpha_old[o])-t + t1)
                di_alpha_new = di_alpha_sum + log_x[i]
                t1 = _ipsi(di_alpha_new, tol=1.48e-9, maxiter=1000)
                diff = abs(di_alpha_new - di_alpha_old)
                di_alpha_old = di_alpha_new
                alpha_new_temp = t1
            alpha_new.append(alpha_new_temp)
        alpha_all.append(alpha_new)

    return alpha_all

def parallel_alpha_computation(args):
    t, alpha_old_i, log_x_i = args
    beta = np.sum(alpha_old_i) - t
    alpha_new_temp = _ipsi(np.log(beta) + log_x_i, tol=1.48e-9, maxiter=10000)
    return alpha_new_temp




def dirichlet_mix_mle_highdimensional(x: np.ndarray, gamma: np.ndarray, alpha_init):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    alpha_not = alpha_init
    N_js = np.sum(gamma, axis=0)
    log_x = (1 / N_js) * np.dot(np.log(x).T, gamma)
    alpha_old = alpha_not
    alpha_all = []
    for o in range(k):
        alpha_new = []
        with Pool() as pool:
            # Prepare arguments for parallel computation
            args_list = [(copy.deepcopy(alpha_old[o][i]), copy.deepcopy(alpha_old[o]), log_x[i,o]) for i in range(p)]

            # Use the Pool to parallelize the computation
            alpha_new = pool.map(parallel_alpha_computation, args_list)
        # for i in range(p):
        #     t = copy.deepcopy(alpha_old[o][i])
        #     beta = np.sum(alpha_old[o])-t
        #     alpha_new_temp = _ipsi(
        #         np.log(beta)+log_x[i,o], tol=1.48e-9, maxiter=10000)
        #     alpha_new.append(alpha_new_temp)
        alpha_all.append(alpha_new)

    return alpha_all


def parallel_alpha_computation2(args):
    t,beta, log_x_i = args
    beta=beta-t
    alpha_new_temp = _ipsi(np.log(beta) + log_x_i, tol=1.48e-9, maxiter=10000)
    return alpha_new_temp





def dirichlet_mix_mle_approx(x: np.ndarray, gamma: np.ndarray, alpha_init):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    alpha_not = alpha_init
    N_js = np.sum(gamma, axis=0)
    log_x = (1 / N_js) * np.dot(np.log(x).T, gamma)
    # x_lol = x.tolist()
    # alpha_not = alpha_init
    # N_js = [np.sum(gamma[:, l]) for l in range(k)]
    alpha_old = alpha_not
    alpha_all = []
    _,data_cwise=mixture_clusters(gamma, x.tolist())
    for o in range(k):

        if np.array(data_cwise[o]).ndim == 1:
            alpha_new = alpha_init[o]
        else:
            beta=get_precision_es(data_cwise[o])
            alpha_new = []
            with Pool(processes=16) as pool:
                # Prepare arguments for parallel computation
                args_list = [(copy.deepcopy(alpha_old[o][i]), beta, log_x[i,o]) for i in range(p)]

                # Use the Pool to parallelize the computation
                alpha_new = pool.map(parallel_alpha_computation2, args_list)

        alpha_all.append(alpha_new)

    return alpha_all







def dirichlet_mean_precision_mle(x: np.ndarray, gamma: np.ndarray, alpha_init):
    alpha_old = alpha_init
    max_iter=100
    diff = 5
    norm0 = 5
    it=0
    while diff > 0.0001:
        alpha_1 = _fit_s(x, gamma, alpha_old)
        alpha_2 = _fit_m(x, gamma, alpha_1)
        norm1 = norm(np.array(alpha_2)-np.array(alpha_old))
        diff = abs(norm1-norm0)
        #print("alpha_diff",diff)
        norm0 = norm1
        alpha_old = alpha_2
    #     it=it+1
    #     if it==max_iter:
    #         return alpha_2
    # print(it)
    return alpha_2


def dirichlet_mean_identical_precision_mle(x: np.ndarray, gamma: np.ndarray, alpha_init):
    alpha_old = alpha_init
    diff = 5
    norm0 = 5
    while diff > 0.0001:
        alpha_1 = _fit_s_identical(x, gamma, alpha_old)
        alpha_2 = _fit_m(x, gamma, alpha_1)
        norm1 = norm(np.array(alpha_2)-np.array(alpha_old))
        diff = abs(norm1-norm0)
        #print("alpha_diff",diff)
        norm0 = norm1
        alpha_old = alpha_2
    return alpha_2

def dirichlet_mean_precision_mle_approx(x: np.ndarray, gamma: np.ndarray, alpha_init):
    alpha_old = alpha_init
    diff = 5
    norm0 = 5
    while diff > 0.0001:
        alpha_1 = _fit_s_approx(x, gamma, alpha_old)
        alpha_2 = _fit_m(x, gamma, alpha_1)
        norm1 = norm(np.array(alpha_2)-np.array(alpha_old))
        diff = abs(norm1-norm0)
        #print("alpha_diff",diff)
        norm0 = norm1
        alpha_old = alpha_2
    return alpha_2

def dirichlet_mean_identical_precision_mle_approx(x: np.ndarray, gamma: np.ndarray, alpha_init):
    alpha_old = alpha_init
    diff = 5
    norm0 = 5
    while diff > 0.0001:
        alpha_1 = _fit_s_ideantical_approx(x, gamma, alpha_old)
        alpha_2 = _fit_m(x, gamma, alpha_1)
        norm1 = norm(np.array(alpha_2)-np.array(alpha_old))
        diff = abs(norm1-norm0)
        #print("alpha_diff",diff)
        norm0 = norm1
        alpha_old = alpha_2
    return alpha_2



def dirichlet_mean_known_precision_mle(x: np.ndarray, gamma: np.ndarray, alpha_init,true_s):

    alpha_1 = _fit_m_known_s(x, gamma, alpha_init,true_s)
    return alpha_1


def dirichlet_known_mean_precision_mle(x: np.ndarray, gamma: np.ndarray, alpha_init,true_m):

    alpha_1 = _fit_s_known_m(x, gamma, alpha_init,true_m)
    return alpha_1


##
##def _fit_s(x: np.ndarray, gamma: np.ndarray, alpha_old):
##
##    p = x.shape[1]
##    n = x.shape[0]
##    k = gamma.shape[1]
##    x_lol = x.tolist()
##    N_js = [np.sum(gamma[:, l]) for l in range(k)]
##    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
##    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
##              for i in range(len(alpha_old))]
##    alpha_new_all = []
##    for o in range(k):
##        diff = 5
##        sjold = sj_old[o]
##        max_iter=100
##        it=0
##        while diff > 0.0001:
##            cs = []
##            for i in range(n):
##                for j in range(p):
##                    c_temp = mj_old[o][j]*gamma[i, o]*np.log(x_lol[i][j])
##                    cs.append(c_temp)
##            c = np.sum(cs)
##            bs = []
##            for i in range(p):
##                b_temp = mj_old[o][i]*digamma(sjold*mj_old[o][i])
##                bs.append(b_temp)
##            b = N_js[o]*np.sum(bs)
##            a = N_js[o]*digamma(sjold)
##            firststdif = a-b+c
##            d = N_js[o]*_trigamma(sjold)
##            es = []
##            for i in range(p):
##                es_temp = (mj_old[o][i]**2)*_trigamma(sjold*mj_old[o][i])
##                es.append(es_temp)
##            e = N_js[o]*np.sum(es)
##            seconddif = d-e
##            # if seconddif < 0:
##            #     sj_new = sjold - firststdif/seconddif
##            # elif firststdif + sjold*seconddif < 0:
##            #     sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
##            # else:
##            #     raise NotConvergingError(f"Unable to update s from {sjold}")
##            if firststdif + sjold*seconddif < 0:
##                sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
##            elif seconddif < 0:
##                sj_new = sjold - firststdif/seconddif
##            else:
##                # raise NotConvergingError(f"Unable to update s from {sjold}")
##                sj_new = max(1e-6, sjold * 0.9)
##            diff = abs(sj_new-sjold)
##            #print("s_diff",diff)
##            sjold = sj_new
##            it=it+1
##            if it==max_iter:
##                break
##        # print(it)
##        alphaj_new = [sj_new*mj_old[o][i] for i in range(p)]
##        alpha_new_all.append(alphaj_new)
##    return alpha_new_all
##
##
##def _fit_m(x: np.ndarray, gamma: np.ndarray, alpha_old):
##    p = x.shape[1]
##    n = x.shape[0]
##    k = gamma.shape[1]
##    x_lol = x.tolist()
##    N_js = [np.sum(gamma[:, l]) for l in range(k)]
##    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
##    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
##              for i in range(len(alpha_old))]
##    alpha_new_all = []
##
##    for o in range(k):
##
##        sjold = sj_old[o]
##        mjold = mj_old[o]
##        alpha_j_old = alpha_old[o]
##        m6 = []
##        for s in range(p):
##            m5 = np.sum([gamma[i, o]*np.log(x_lol[i][s])
##                        for i in range(n)])/N_js[o]
##            m6.append(m5)
##        diff = 5
##        norm0 = 5
##        while diff > 0.0001:
##            alpha_j_new = []
##            for j in range(p):
##                m1 = m6[j]
##                m2 = [mjold[q]*m6[q]for q in range(p)]
##                m3 = np.sum(m2)
##                m4 = np.sum([mjold[r]*digamma(sjold*mjold[r])
##                            for r in range(p)])
##                alpha_jm_new = _ipsi(m1-m3+m4, tol=1.48e-9, maxiter=1000)
##                alpha_j_new.append(alpha_jm_new)
##            alpha_j_new = np.array(alpha_j_new) / \
##                np.array(alpha_j_new).sum() * sjold
##            norm1 = norm(alpha_j_new-alpha_j_old)
##            diff = abs(norm1-norm0)
##            #print("m_diff",diff)
##            norm0 = norm1
##            mjold = alpha_j_new/alpha_j_new.sum()
##        alpha_new_all.append(alpha_j_new.tolist())
##    return alpha_new_all


def _fit_s(x: np.ndarray, gamma: np.ndarray, alpha_old):
    """
    Numerically-stable version of your _fit_s.
    Formulas are unchanged. We only:
      - avoid Python lists + repeated summations (vectorize the same sums)
      - guard divisions / logs with tiny eps
      - guard 1/0 and inf/nan propagation
      - keep sj positive and finite
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    alpha_old = np.asarray(alpha_old, dtype=float)

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]

    # --- numeric guards (do not change formulas, only avoid nan/inf) ---
    eps_x = 1e-300          # for log(x) only; does not alter positive x, only clips zeros
    eps_s = 1e-12           # for safe reciprocals / denom checks
    s_min = 1e-8            # keep precision strictly positive
    s_max = 1e12            # prevent overflow in digamma/trigamma (purely numeric safeguard)

    # N_js as in your code; but guard zeros
    N_js = np.sum(gamma, axis=0)
    N_js_safe = np.maximum(N_js, eps_s)

    # sj_old and mj_old as in your code; guard zero sums
    sj_old = np.sum(alpha_old, axis=1)
    sj_old_safe = np.maximum(sj_old, s_min)
    mj_old = alpha_old / sj_old_safe[:, None]  # same as alpha_old/sj_old

    # log(x) with clipping to avoid -inf when x hits 0 due to numeric issues
    logx = np.log(np.clip(x, eps_x, None))

    alpha_new_all = []
    for o in range(k):
        diff = 5.0
        sjold = float(np.clip(sj_old_safe[o], s_min, s_max))
        max_iter = 100
        it = 0

        # cache component weights
        go = gamma[:, o]

        while diff > 0.0001:
            mjo = mj_old[o]  # (p,)

            # c = sum_{i,j} m_j * gamma_i * log(x_ij)
            # vectorized: sum_j m_j * sum_i gamma_i * logx_ij
            # -> mjo @ (logx.T @ go)
            c = float(mjo @ (logx.T @ go))

            # b = N_js[o] * sum_j m_j * digamma(s * m_j)
            sm = sjold * mjo
            # guard sm > 0 (should be; but clip for safety)
            sm = np.clip(sm, s_min, s_max)

            b = float(N_js[o] * np.sum(mjo * digamma(sm)))

            # a = N_js[o] * digamma(s)
            a = float(N_js[o] * digamma(np.clip(sjold, s_min, s_max)))

            firststdif = a - b + c  # unchanged

            # d = N_js[o] * trigamma(s)
            d = float(N_js[o] * _trigamma(np.clip(sjold, s_min, s_max)))

            # e = N_js[o] * sum_j m_j^2 * trigamma(s*m_j)
            e = float(N_js[o] * np.sum((mjo ** 2) * _trigamma(sm)))

            seconddif = d - e  # unchanged

            # --- update step: SAME formulas, with safe evaluation ---
            sj_new = None

            # If any nan/inf occurred, damp (doesn't change formulas; avoids propagating nan)
            if (not np.isfinite(firststdif)) or (not np.isfinite(seconddif)) or (not np.isfinite(sjold)):
                sj_new = sjold * 0.9

            else:
                cond1 = firststdif + sjold * seconddif  # unchanged condition
                if cond1 < 0:
                    # sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
                    # evaluate safely without changing algebra:
                    inv_s = 1.0 / max(sjold, s_min)
                    inv_s2 = inv_s * inv_s
                    inv_sec = 1.0 / seconddif if abs(seconddif) > eps_s else np.sign(seconddif) / eps_s
                    denom = inv_s + inv_s2 * inv_sec * firststdif
                    if (not np.isfinite(denom)) or abs(denom) < eps_s:
                        sj_new = sjold * 0.9
                    else:
                        sj_new = 1.0 / denom

                elif seconddif < 0:
                    # sj_new = sjold - firststdif/seconddif
                    denom = seconddif if abs(seconddif) > eps_s else -eps_s
                    sj_new = sjold - (firststdif / denom)

                else:
                    # your damping fallback
                    sj_new = sjold * 0.9

            # enforce positivity/finite (numeric safeguard only)
            sj_new = float(np.clip(sj_new, s_min, s_max))

            diff = abs(sj_new - sjold)
            sjold = sj_new
            it += 1
            if it == max_iter:
                break

        alphaj_new = (sjold * mj_old[o]).tolist()  # same formula
        alpha_new_all.append(alphaj_new)

    return alpha_new_all


def _fit_m(x: np.ndarray, gamma: np.ndarray, alpha_old):
    """
    Numerically-stable version of your _fit_m.
    Formulas are unchanged.
    Added max-iteration safeguard to prevent infinite loops.
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    alpha_old = np.asarray(alpha_old, dtype=float)

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]

    eps_x = 1e-300
    eps_w = 1e-12
    s_min = 1e-8
    s_max = 1e12
    max_iter = 200   # ← NEW safeguard

    logx = np.log(np.clip(x, eps_x, None))

    N_js = np.sum(gamma, axis=0)
    N_js_safe = np.maximum(N_js, eps_w)

    sj_old = np.sum(alpha_old, axis=1)
    sj_old_safe = np.maximum(sj_old, s_min)

    mj_old = alpha_old / sj_old_safe[:, None]

    alpha_new_all = []

    for o in range(k):
        sjold = float(np.clip(sj_old_safe[o], s_min, s_max))
        mjold = mj_old[o].copy()
        alpha_j_old = alpha_old[o].copy()

        go = gamma[:, o]

        m6 = (logx.T @ go) / N_js_safe[o]

        diff = 5.0
        norm0 = 5.0
        it = 0   # ← iteration counter

        while diff > 0.0001 and it < max_iter:

            # m3 = sum_q mjold[q]*m6[q]
            m3 = float(np.sum(mjold * m6))

            # m4 = sum_r mjold[r]*digamma(sjold*mjold[r])
            sm = np.clip(sjold * mjold, s_min, s_max)
            m4 = float(np.sum(mjold * digamma(sm)))

            # alpha_jm_new = _ipsi(m6 - m3 + m4)
            y = (m6 - m3 + m4)
            alpha_j_new = _ipsi(y, tol=1.48e-9, maxiter=1000)
            alpha_j_new = np.asarray(alpha_j_new, dtype=float)

            # normalize (unchanged formula)
            ssum = float(np.sum(alpha_j_new))
            if (not np.isfinite(ssum)) or ssum <= 0:
                alpha_j_new = np.clip(mjold, eps_w, None)
                alpha_j_new = alpha_j_new / np.sum(alpha_j_new) * sjold
            else:
                alpha_j_new = (alpha_j_new / ssum) * sjold

            # convergence metric (unchanged)
            norm1 = norm(alpha_j_new - alpha_j_old)
            diff = abs(norm1 - norm0)
            norm0 = norm1

            # update mjold
            denom = float(np.sum(alpha_j_new))
            if (not np.isfinite(denom)) or denom <= 0:
                denom = eps_w
            mjold = alpha_j_new / denom

            it += 1

        # If max_iter reached, we silently accept last iterate.
        # (We do NOT change formulas; only stop iteration.)

        alpha_new_all.append(alpha_j_new.tolist())

    return alpha_new_all

def _fit_m_known_s(x: np.ndarray, gamma: np.ndarray, alpha_old,true_s):
    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    _,data_cwise=mixture_clusters(gamma.tolist(), x.tolist())
    if true_s==None:
        sj_old=[get_precision_es(data_cwise[o]) for o in range(k)]
    else:
        sj_old=true_s
    # sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
              for i in range(len(alpha_old))]
    alpha_new_all = []

    for o in range(k):

        sjold = sj_old[o]
        mjold = mj_old[o]
        alpha_j_old = alpha_old[o]
        m6 = []
        for s in range(p):
            m5 = np.sum([gamma[i, o]*np.log(x_lol[i][s])
                        for i in range(n)])/N_js[o]
            m6.append(m5)
        diff = 5
        norm0 = 5
        while diff > 0.0001:
            alpha_j_new = []
            for j in range(p):
                m1 = m6[j]
                m2 = [mjold[q]*m6[q]for q in range(p)]
                m3 = np.sum(m2)
                m4 = np.sum([mjold[r]*digamma(sjold*mjold[r])
                            for r in range(p)])
                alpha_jm_new = _ipsi(m1-m3+m4, tol=1.48e-9, maxiter=1000)
                alpha_j_new.append(alpha_jm_new)
            alpha_j_new = np.array(alpha_j_new) / \
                np.array(alpha_j_new).sum() * sjold
            norm1 = norm(alpha_j_new-alpha_j_old)
            diff = abs(norm1-norm0)
            #print("m_diff",diff)
            norm0 = norm1
            mjold = alpha_j_new/alpha_j_new.sum()
        alpha_new_all.append(alpha_j_new.tolist())
    return alpha_new_all





def _fit_s_identical(x: np.ndarray, gamma: np.ndarray, alpha_old):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
              for i in range(len(alpha_old))]
    alpha_new_all = []
    diff = 5
    sjold = sj_old[0]
    while diff > 0.0001:
        cs = []
        for i in range(n):
            for j in range(p):
                c_temp = mj_old[0][j]*gamma[i, 0]*np.log(x_lol[i][j])
                cs.append(c_temp)
        c = np.sum(cs)
        bs = []
        for i in range(p):
            b_temp = mj_old[0][i]*digamma(sjold*mj_old[0][i])
            bs.append(b_temp)
        b = N_js[0]*np.sum(bs)
        a = N_js[0]*digamma(sjold)
        firststdif = a-b+c
        d = N_js[0]*_trigamma(sjold)
        es = []
        for i in range(p):
            es_temp = (mj_old[0][i]**2)*_trigamma(sjold*mj_old[0][i])
            es.append(es_temp)
        e = N_js[0]*np.sum(es)
        seconddif = d-e
        # if seconddif < 0:
        #     sj_new = sjold - firststdif/seconddif
        # elif firststdif + sjold*seconddif < 0:
        #     sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
        if firststdif + sjold*seconddif < 0:
            sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
        elif seconddif < 0:
            sj_new = sjold - firststdif/seconddif
        else:
            raise NotConvergingError(f"Unable to update s from {sjold}")
        diff = abs(sj_new-sjold)
        #print("s_diff",diff)
        sjold = sj_new
    for o in range(k):
        alphaj_new = [sj_new*mj_old[o][i] for i in range(p)]
        alpha_new_all.append(alphaj_new)
    return alpha_new_all


def _fit_s_approx(x: np.ndarray, gamma: np.ndarray, alpha_old):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
              for i in range(len(alpha_old))]
    alpha_new_all = []
    _,data_cwise=mixture_clusters(gamma.tolist(), x.tolist())
    for o in range(k):
        beta=get_precision_es(data_cwise[o])
        # sjold = sj_old[o]
        cs = []
        for i in range(n):
            for j in range(p):
                c_temp = mj_old[o][j]*gamma[i, o]*np.log(x_lol[i][j])
                cs.append(c_temp)
        c = np.sum(cs)/N_js[o]
        bs = []
        for i in range(p):
            b_temp = mj_old[o][i]*digamma(beta*mj_old[o][i])
            bs.append(b_temp)
        b = np.sum(bs)
        sj_new=_ipsi(
            b-c, tol=1.48e-9, maxiter=10000)
        alphaj_new = [sj_new*mj_old[o][i] for i in range(p)]
        alpha_new_all.append(alphaj_new)
    return alpha_new_all

def _fit_s_ideantical_approx(x: np.ndarray, gamma: np.ndarray, alpha_old):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
    mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
              for i in range(len(alpha_old))]
    alpha_new_all = []
    _,data_cwise=mixture_clusters(gamma.tolist(), x.tolist())
    beta=get_precision_es(data_cwise[0])
    # sjold = sj_old[o]
    cs = []
    for i in range(n):
        for j in range(p):
            c_temp = mj_old[0][j]*gamma[i, 0]*np.log(x_lol[i][j])
            cs.append(c_temp)
    c = np.sum(cs)/N_js[0]
    bs = []
    for i in range(p):
        b_temp = mj_old[0][i]*digamma(beta*mj_old[0][i])
        bs.append(b_temp)
    b = np.sum(bs)
    sj_new=_ipsi(
        b-c, tol=1.48e-9, maxiter=10000)
    for o in range(k):
        alphaj_new = [sj_new*mj_old[o][i] for i in range(p)]
        alpha_new_all.append(alphaj_new)
    return alpha_new_all


def _fit_s_known_m(x: np.ndarray, gamma: np.ndarray, alpha_old,true_m):

    p = x.shape[1]
    n = x.shape[0]
    k = gamma.shape[1]
    x_lol = x.tolist()
    N_js = [np.sum(gamma[:, l]) for l in range(k)]
    sj_old = [np.sum(alpha_old[i]) for i in range(len(alpha_old))]
    # sj_old = [28, 92, 60, 4.8]
    # mj_old = [(np.array(alpha_old[i])/sj_old[i]).tolist()
    #           for i in range(len(alpha_old))]
    # print(mj_old)
    _,data_cwise=mixture_clusters(gamma.tolist(), x.tolist())
    if true_m ==None:
        mj_old=[get_means(data_cwise[i]) for i in range(len(alpha_old))]
    else:
        mj_old=true_m
    # print(mj_old)
    alpha_new_all = []
    for o in range(k):
        diff = 5
        sjold = sj_old[o]
        while diff > 0.0001:
            cs = []
            for i in range(n):
                for j in range(p):
                    c_temp = mj_old[o][j]*gamma[i, o]*np.log(x_lol[i][j])
                    cs.append(c_temp)
            c = np.sum(cs)
            bs = []
            for i in range(p):
                b_temp = mj_old[o][i]*digamma(sjold*mj_old[o][i])
                bs.append(b_temp)
            b = N_js[o]*np.sum(bs)
            a = N_js[o]*digamma(sjold)
            firststdif = a-b+c
            d = N_js[o]*_trigamma(sjold)
            es = []
            for i in range(p):
                es_temp = (mj_old[o][i]**2)*_trigamma(sjold*mj_old[o][i])
                es.append(es_temp)
            e = N_js[o]*np.sum(es)
            seconddif = d-e
            # if seconddif < 0:
            #     sj_new = sjold - firststdif/seconddif
            # elif firststdif + sjold*seconddif < 0:
            #     sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
            if firststdif + sjold*seconddif < 0:
                sj_new = 1/((1/sjold)+(1/sjold**2)*(1/seconddif)*firststdif)
            elif seconddif < 0:
                sj_new = sjold - firststdif/seconddif
            else:
                raise NotConvergingError(f"Unable to update s from {sjold}")
            diff = abs(sj_new-sjold)
            #print("s_diff",diff)
            sjold = sj_new
        alphaj_new = [sj_new*mj_old[o][i] for i in range(p)]
        alpha_new_all.append(alphaj_new)
    return alpha_new_all




def estimate_alphas(
    data: np.ndarray,
    gamma_temp_ar,
    alpha_temp,
    method,
    true_m,
    true_s,
    post_m_step=None,
):
    if method == "fixediteration":
        alpha_new = dirichlet_mix_mle(data, gamma_temp_ar, alpha_temp)
    elif method == "meanprecision":
        alpha_new = dirichlet_mean_precision_mle(
            data, gamma_temp_ar, alpha_temp)
    # elif method == "highdimensional":
    #     alpha_new = dirichlet_mix_mle_highdimensional(data, gamma_temp_ar, alpha_temp)
    elif method == "highdimensional":
        alpha_new = dirichlet_mix_mle_approx(data, gamma_temp_ar, alpha_temp)
    elif method == "meanprecision_identical":
        alpha_new = dirichlet_mean_identical_precision_mle(data, gamma_temp_ar, alpha_temp)
    elif method == "meanprecision_approx":
        alpha_new = dirichlet_mean_precision_mle_approx(
            data, gamma_temp_ar, alpha_temp)
    elif method == "meanprecision_identical_approx":
        alpha_new = dirichlet_mean_identical_precision_mle_approx(data, gamma_temp_ar, alpha_temp)
    elif method == "meanprecision_known_mean":
        alpha_new = dirichlet_known_mean_precision_mle(data, gamma_temp_ar, alpha_temp,true_m)
    elif method == "meanprecision_known_precision":
        alpha_new = dirichlet_mean_known_precision_mle(data, gamma_temp_ar, alpha_temp,true_s)

    return alpha_new



class DMM_Soft(BaseMixture):
    def __init__(self,n_clusters,tol=0.0001,initialization="kmeans",method="meanprecision",print_log_likelihood=False,max_iter=25, verbose=True):
        super().__init__(n_clusters=n_clusters, EM_type="Soft", mixture_type="identical", tol=tol, print_log_likelihood=print_log_likelihood, max_iter=max_iter, verbose=verbose)
        self.k=n_clusters
        self.initialization=initialization
        self.method = method
        self.family = "dirichlet"

    def _log_pdf_dirichlet(self,X,alphas):
        threshold = 1e-10
        N,p=X.shape
        k=len(alphas)
        probs=np.empty((N, k))
        for j in range(k):
            alpha=alphas[j]
            for i in range(N):
                with np.errstate(under="ignore"):
                    x=X[i, :]
                    t1=[(alphm-1)*np.log(xm) for alphm,xm in zip(alpha,x)]
                    # t1 = [(alphm - 1) * (np.log1p(xm - 1) if xm < threshold else np.log(xm)) for alphm, xm in zip(alpha, x)]
                    t2=np.sum(gammaln(alpha))
                    t3=gammaln(np.sum(alpha))
                    probs[i,j]=np.sum(t1)-t2+t3
        return probs

    def _estimate_weighted_log_prob_identical(self, X, alpha, pi):
        return self._log_pdf_dirichlet(X,alpha) + np.log(pi)

    def fit(self,sample,true_m=None,true_s=None, post_m_step=None):
        start_time = time.time()
        self.data = self._process_data(sample)
        self.n, self.p = self.data.shape
        self.total_parameters = (self.k -1) + self.k* (self.p)
        np.random.seed(0)
        # if self.initialization == "kmeans":
        #     self.pi_not, self.alpha_not = kmeans_init(self.data, self.k)
        # elif self.initialization == "gmm":
        #     self.pi_not, self.alpha_not = gmm_init(self.data, self.k)
        if self.initialization == "kmeans":
            self.pi_not, self.alpha_not = kmeans_init_adv(self.data, self.k)
        elif self.initialization == "gmm":
            self.pi_not, self.alpha_not = gmm_init_adv(self.data, self.k)
        elif self.initialization == "random":
            self.pi_not, self.alpha_not = random_init(self.data, self.k)

        self.alpha_temp = self.alpha_not
        self.pi_temp = self.pi_not
        pi_new,alpha_new, log_likelihood_new,log_gamma_new=self._fit(self.data,self.pi_temp,self.alpha_temp,estimate_alphas,method=self.method,true_m=true_m,true_s=true_s, post_m_step=post_m_step)
        estimated_mean = []
        for a in alpha_new:
            mean_temp = [b/np.sum(a) for b in a]
            estimated_mean.append(mean_temp)
        self.pi_new = pi_new
        self.alpha_new = alpha_new
        self.estimated_mean = estimated_mean
        self.cluster = log_gamma_new.argmax(axis=1)
        # self.data_cwise = data_cwise
        self.gamma_temp_ar = np.exp(log_gamma_new)
        # self.gamma_matrix = gamma_matrix
        self.log_likelihood_new = log_likelihood_new
        if self.verbose:
            print("Soft DMM Fitting Done Successfully")
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

    def get_mean(self):

        return self.estimated_mean

    def get_precision(self):
        preci=[np.sum(al) for al in self.alpha_new]

        return preci

    def responsibilities(self):

        return self.gamma_temp_ar

    def n_iter(self):
        return len(self.log_likelihoods)

    def get_info_mat(self, method="score"):
        IM, SE = combined_info_and_se(self.pi_new, np.array(self.alpha_new), self.gamma_temp_ar, self.data, method, mode = "soft")

        return IM, SE

    def get_mean_precision_inference(self, method="score", alpha=0.05):
        I, _ = self.get_info_mat(method=method)
        cov = np.linalg.pinv(I)

        K, p = self.alpha_new.shape
        offset = (K-1)

        results = []

        for j in range(K):
            idx = slice(offset + j*p, offset + (j+1)*p)
            cov_alpha = cov[idx, idx]
            alpha_hat = self.alpha_new[j]

            res = mean_precision_inference(alpha_hat, cov_alpha, alpha)
            results.append(res)

        return results





    # def clustered_data(self):
    #     return pd.DataFrame(self.data_cwise).transpose()

    # def bic(self):

    #     return ((self.k-1)+(self.k*self.p))*np.log(self.n) - 2*(self.log_likelihood_new)

    # def aic(self):

    #     return 2*((self.k-1)+(self.k*self.p)) - 2*(self.log_likelihood_new)

    # def icl(self):
    #     entropy_s=[]
    #     for i in self.gamma_matrix:
    #         for j in i:
    #                 ent_temp=j*np.log1p(j)
    #                 entropy_s.append(ent_temp)



    #     entropy=np.sum(entropy_s)

    #     return ((self.k-1)+(self.k*self.p))*np.log(self.n) - 2*(self.log_likelihood_new) - entropy
