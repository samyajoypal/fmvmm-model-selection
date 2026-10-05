import numpy as np

from fmvmm.distributions import dirichlet
from fmvmm.inference import as_adapter, fixed_value, score_test, wald_test
from fmvmm.mixtures.DMM_Soft import DMM_Soft
from fmvmm.utils.utils_mixture import sample_mixture_distribution


def main():
    np.random.seed(7)

    pis = [0.45, 0.55]
    alphas = [
        [[6.0, 2.0, 3.0]],
        [[2.0, 7.0, 4.0]],
    ]
    data, _labels = sample_mixture_distribution(80, dirichlet.rvs, pis, alphas)

    model = DMM_Soft(n_clusters=2, max_iter=50, tol=1e-4, verbose=False)
    model.fit(data)

    adapter = as_adapter(model)
    pv_internal = adapter.parameter_vector(
        parameterization="internal",
        info_method="score",
    )
    pv_user = adapter.parameter_vector(
        parameterization="user",
        info_method="score",
    )

    assert "eta[0]" in pv_internal.names
    assert "pi[0]" in pv_user.names
    assert pv_internal.covariance.shape[0] == pv_internal.theta.size
    assert pv_user.covariance.shape[0] == pv_user.theta.size

    H = fixed_value(pv_user, "pi[0]", 0.5)
    wald = wald_test(
        model,
        H,
        parameterization="user",
        info_method="score",
    )
    score = score_test(
        model,
        ["eta[0]"],
        parameterization="internal",
        info_method="score",
    )

    assert np.isfinite(wald.statistic)
    assert np.isfinite(score.statistic)
    assert wald.df == 1
    assert score.df == 1

    print("DMM generic inference example passed.")


if __name__ == "__main__":
    main()
