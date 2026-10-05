"""Regression checks for bounded soft M-steps and GEM acceptance."""

import numpy as np

from fmvmm.mixtures.FMVMM import (
    _gem_accept_component_update,
    _weighted_mvn_update,
    fmvmm,
)


def test_component_gem_acceptance_rejects_worse_update():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(40, 2))
    weights = np.ones(x.shape[0])
    alpha_prev = _weighted_mvn_update(x, weights, None)
    bad_alpha = (
        alpha_prev[0] + np.array([100.0, -100.0]),
        alpha_prev[1],
    )

    accepted = _gem_accept_component_update("mvn", x, weights, alpha_prev, bad_alpha)

    assert np.allclose(accepted[0], alpha_prev[0])
    assert np.allclose(accepted[1], alpha_prev[1])


def test_bounded_soft_mstep_keeps_observed_loglik_non_decreasing():
    rng = np.random.default_rng(321)
    x = np.vstack([
        rng.normal(loc=(-2.0, 0.0), scale=0.45, size=(45, 2)),
        rng.normal(loc=(2.0, 0.4), scale=0.70, size=(45, 2)),
    ])

    model = fmvmm(
        n_clusters=2,
        list_of_dist=["mvn", "mvt"],
        specific_comb=True,
        candidate_combinations=[("mvn", "mvt")],
        assignment_permutations=True,
        em_type="soft",
        max_iter=5,
        tol=1e-8,
        mstep_max_iter=2,
        mstep_maxfun=25,
        verbose=False,
    )
    model.fit(x)

    assert model.worked_dist
    for loglikes in model.list_all_log_likelihood:
        diffs = np.diff(np.asarray(loglikes, dtype=float))
        assert np.all(diffs >= -1e-7), loglikes


if __name__ == "__main__":
    test_component_gem_acceptance_rejects_worse_update()
    test_bounded_soft_mstep_keeps_observed_loglik_non_decreasing()
    print("fmvmm m-step tests passed")
