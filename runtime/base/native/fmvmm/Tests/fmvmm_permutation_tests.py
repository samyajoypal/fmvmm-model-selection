"""Regression checks for family-to-cluster assignment starts."""

import numpy as np

from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.distributions import multivariate_t as mvt
from fmvmm.mixtures.FMVMM import (
    _assignment_equivalence_key,
    _expand_assignment_permutations,
    dist_name_by_module,
    fmvmm,
)


def test_distinct_assignment_expansion():
    starts = _expand_assignment_permutations([(mvn, mvt), (mvn, mvn)])
    assert starts == [(mvn, mvt), (mvt, mvn), (mvn, mvn)]
    assert _assignment_equivalence_key((mvn, mvt)) == _assignment_equivalence_key((mvt, mvn))


def test_permutations_are_collapsed_after_fit():
    rng = np.random.default_rng(912)
    x = np.vstack([
        rng.normal(loc=(-2.0, -0.5), scale=0.45, size=(80, 2)),
        rng.normal(loc=(2.0, 0.5), scale=0.65, size=(80, 2)),
    ])
    model = fmvmm(
        n_clusters=2,
        list_of_dist=["mvn", "mvt"],
        specific_comb=True,
        candidate_combinations=[("mvn", "mvt")],
        assignment_permutations=True,
        em_type="soft",
        max_iter=8,
        tol=1e-5,
        verbose=False,
    )
    model.fit(x)

    assert model.n_assignment_starts == 2
    assert len(model.worked_dist) == 1
    assert sorted(dist_name_by_module[module] for module in model.worked_dist[0]) == [
        "mvn", "mvt"
    ]


def test_component_constraints_reject_permutation_expansion():
    try:
        fmvmm(
            n_clusters=2,
            list_of_dist=["mvn", "mvt"],
            specific_comb=True,
            assignment_permutations=True,
            fixed_mu=[{"component": 0, "coordinate": 0, "value": 0.0}],
        )
    except ValueError as exc:
        assert "fixed_mu" in str(exc)
    else:
        raise AssertionError("Expected component-indexed constraints to be rejected.")


if __name__ == "__main__":
    test_distinct_assignment_expansion()
    test_permutations_are_collapsed_after_fit()
    test_component_constraints_reject_permutation_expansion()
    print("fmvmm permutation tests passed")
