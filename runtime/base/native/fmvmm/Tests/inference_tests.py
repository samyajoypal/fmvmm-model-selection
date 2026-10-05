import numpy as np

from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.inference import as_adapter, fixed_value, lrt, score_test, wald_test
from fmvmm.mixtures.FMVMM import fmvmm


def _data():
    rng = np.random.default_rng(321)
    np.random.seed(321)
    x1 = mvn.rvs(
        np.array([0.0, 0.0]),
        np.array([[0.45, 0.05], [0.05, 0.35]]),
        size=18,
        random_state=rng,
    )
    x2 = mvsl.rvs(
        np.array([2.3, 2.0]),
        np.array([[0.5, 0.0], [0.0, 0.45]]),
        np.array([0.5, -0.25]),
        size=18,
    )
    return np.vstack([x1, x2])


def main():
    x = _data()

    model = fmvmm(
        n_clusters=2,
        list_of_dist=["mvn", "mvsl"],
        specific_comb=True,
        assignment_permutations=True,
        max_iter=3,
        verbose=False,
        debug=False,
    )
    model.fit(x)

    adapter = as_adapter(model)
    pv_user = adapter.parameter_vector(parameterization="user")
    pv_internal = adapter.parameter_vector(parameterization="internal")

    assert "pi[0]" in pv_user.names
    assert "eta[0]" in pv_internal.names
    assert pv_user.covariance.shape[0] == pv_user.theta.size
    assert pv_internal.covariance.shape[0] == pv_internal.theta.size

    H = fixed_value(pv_user, "pi[0]", 0.5)
    wald = wald_test(model, H, parameterization="user")
    assert np.isfinite(wald.statistic)
    assert wald.df == 1

    null_model = fmvmm(
        n_clusters=2,
        list_of_dist=["mvn", "mvsl"],
        specific_comb=True,
        assignment_permutations=True,
        fixed_pi=[0.5, 0.5],
        max_iter=3,
        verbose=False,
        debug=False,
    )
    null_model.fit(x)
    score = score_test(null_model, ["eta[0]"], parameterization="internal")
    assert np.isfinite(score.statistic)
    assert score.df == 1

    lrt_result = lrt(model, null_model, df=1)
    assert np.isfinite(lrt_result.statistic)
    assert lrt_result.df == 1

    print("Generic inference tests passed.")


if __name__ == "__main__":
    main()
