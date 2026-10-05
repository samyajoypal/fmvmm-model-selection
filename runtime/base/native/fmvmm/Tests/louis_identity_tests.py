"""Numerical validation of the FMVMM Louis identity."""

import numpy as np
from scipy.special import logsumexp

from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.mixtures.FMVMM import fmvmm
from fmvmm.utils.utils_dist import pack_mvn_unconstrained, unpack_mvn_unconstrained


def _negative_hessian(fun, theta, step=1e-4):
    theta = np.asarray(theta, dtype=float)
    h = step * np.maximum(1.0, np.abs(theta))
    f0 = fun(theta)
    result = np.zeros((theta.size, theta.size))
    for a in range(theta.size):
        plus = theta.copy(); plus[a] += h[a]
        minus = theta.copy(); minus[a] -= h[a]
        result[a, a] = -(fun(plus) - 2.0 * f0 + fun(minus)) / h[a] ** 2
        for b in range(a):
            pp = theta.copy(); pp[a] += h[a]; pp[b] += h[b]
            pm = theta.copy(); pm[a] += h[a]; pm[b] -= h[b]
            mp = theta.copy(); mp[a] -= h[a]; mp[b] += h[b]
            mm = theta.copy(); mm[a] -= h[a]; mm[b] -= h[b]
            value = -(fun(pp) - fun(pm) - fun(mp) + fun(mm)) / (4.0 * h[a] * h[b])
            result[a, b] = value
            result[b, a] = value
    return 0.5 * (result + result.T)


def main():
    rng = np.random.default_rng(42)
    x = np.vstack([
        rng.multivariate_normal([-1.2, -0.3], [[0.7, 0.1], [0.1, 0.6]], size=45),
        rng.multivariate_normal([1.3, 0.4], [[0.6, -0.05], [-0.05, 0.8]], size=55),
    ])
    model = fmvmm(
        n_clusters=2,
        list_of_dist=["mvn"],
        specific_comb=False,
        em_type="soft",
        max_iter=80,
        tol=1e-7,
        verbose=False,
    )
    model.fit(x)
    idx = int(np.argmin(model.list_bic))
    pi = np.asarray(model.list_pi[idx], dtype=float)
    alpha = model.list_alpha[idx]
    theta = np.concatenate([
        [np.log(pi[0] / pi[1])],
        pack_mvn_unconstrained(p=2, mu=alpha[0][0], sigma=alpha[0][1]),
        pack_mvn_unconstrained(p=2, mu=alpha[1][0], sigma=alpha[1][1]),
    ])

    def observed_loglik(value):
        eta = value[0]
        pi0 = 1.0 / (1.0 + np.exp(-eta))
        params0 = unpack_mvn_unconstrained(value[1:6], p=2)
        params1 = unpack_mvn_unconstrained(value[6:11], p=2)
        weighted = np.column_stack([
            np.log(pi0) + mvn.logpdf(x, *params0),
            np.log1p(-pi0) + mvn.logpdf(x, *params1),
        ])
        return float(np.sum(logsumexp(weighted, axis=1)))

    direct = _negative_hessian(observed_loglik, theta)
    louis, _se, details = model.get_info_mat(
        method="louis", parameterization="internal", return_details=True, ridge=0.0
    )
    louis_best = np.asarray(louis[idx], dtype=float)
    relative_error = np.linalg.norm(louis_best - direct) / np.linalg.norm(direct)
    assert relative_error < 0.03, relative_error
    assert np.linalg.norm(details[idx]["missing_information"]) > 0
    print(f"Louis identity test passed (relative error {relative_error:.3g}).")


if __name__ == "__main__":
    main()
