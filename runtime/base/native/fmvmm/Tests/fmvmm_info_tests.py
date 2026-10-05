import numpy as np

from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.mixtures.FMVMM import fmvmm


def main():
    rng = np.random.default_rng(123)

    x1 = mvn.rvs(
        np.array([0.0, 0.0]),
        np.array([[0.45, 0.08], [0.08, 0.35]]),
        size=20,
        random_state=rng,
    )
    x2 = mvsl.rvs(
        np.array([2.5, 2.0]),
        np.array([[0.55, -0.05], [-0.05, 0.45]]),
        np.array([0.7, -0.3]),
        size=20,
    )
    x = np.vstack([x1, x2])

    model = fmvmm(
        n_clusters=2,
        list_of_dist=["mvn", "mvsl"],
        max_iter=4,
        verbose=False,
        debug=False,
        em_type="soft",
    )
    model.fit(x)

    info, se, details = model.get_info_mat(method="louis", return_details=True)
    info_internal, se_internal, details_internal = model.get_info_mat(
        method="louis",
        parameterization="eta",
        return_details=True,
    )

    assert len(info) == len(model.worked_dist)
    assert len(se) == len(model.worked_dist)
    assert info[0].shape[0] == info[0].shape[1]
    assert info[0].shape[0] == len(se[0])
    assert info[0].shape[0] == 12
    assert info_internal[0].shape[0] == 11
    assert info_internal[0].shape[0] == len(se_internal[0])
    assert np.all(np.isfinite(se[0]))
    assert details[0]["eta_dim"] == 1
    assert details[0]["cov_pi"].shape == (2, 2)

    lhs = details_internal[0]["complete_score_opg"] - details_internal[0]["missing_information"]
    rhs = details_internal[0]["info_internal"]
    assert np.linalg.norm(lhs - rhs) < 1e-5

    print("FMVMM information matrix test passed.")


if __name__ == "__main__":
    main()
