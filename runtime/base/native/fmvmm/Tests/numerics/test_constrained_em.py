"""Portable tests of constrained component updates and the public soft-EM API."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from fmvmm.mixtures.FMVMM import fmvmm, dist_map
from fmvmm.mixtures._base import BaseMixture
from fmvmm.numerics.em_policy import EMPolicy, bounded_weights
from fmvmm.numerics.constrained_mstep import (
    prepare_component, fit_component, _bounds, _pack, _unpack, _analytic,
)
from fmvmm.numerics.gh_policy import mean_gig


class ForcedBadProposal(fmvmm):
    """Exercise the rejection branch with a deliberately poor valid proposal."""
    def _m_step(self, *args, **kwargs):
        pi, alpha = super()._m_step(*args, **kwargs)
        alpha = copy.deepcopy(alpha)
        native = list(alpha[0])
        location_index = -3 if len(native) == 6 else 0
        native[location_index] = np.asarray(native[location_index]) + 8.
        if len(native) == 3:
            native[2] = np.array([5.])
        alpha[0] = tuple(native)
        return pi, alpha


class SyntheticLegacy(BaseMixture):
    def __init__(self):
        super().__init__(1, 'Soft', 'nonidentical', 0., False, 2, False)

    def _e_step(self, X, alpha, pi, dist_comb=None):
        return -(float(alpha[0])-1.)**2, np.zeros((len(X), 1))

    def _m_step(self, *args, **kwargs):
        return np.array([1.]), [3.]


class RecordStationarity(fmvmm):
    def _post_em_iteration(self, pi, alpha, log_gamma):
        super()._post_em_iteration(pi, alpha, log_gamma)
        if not hasattr(self, 'seen'):
            self.seen = []
        scores, gap = self._final_em_stationarity()
        self.seen.append((max(r['projected_gradient'] for r in scores), gap))


class TestConstrainedEM(unittest.TestCase):
    def test_rejected_proposal_preserves_gh_and_infinite_df(self):
        rng = np.random.default_rng(9094)
        for name in ('mvt', 'mghp'):
            with self.subTest(name=name):
                x = rng.normal(size=(80, 2))
                mu, sigma = x.mean(0), np.cov(x.T, bias=True)
                a = ((mu, sigma, np.array([np.inf])) if name == 'mvt'
                     else (1., 2., 3., mu, sigma, np.zeros(2)))
                expected, _ = prepare_component(name, a, EMPolicy())
                model = ForcedBadProposal(
                    1, list_of_dist=[name], numerical_policy={}, max_iter=4,
                    candidate_initializations={(name,): ([1.], [a])}, verbose=False)
                with patch('fmvmm.mixtures._base._blend_parameter',
                           side_effect=AssertionError('Constrained EM must not interpolate')):
                    model.fit(x)
                self.assertFalse(model.candidate_failures)
                diagnostics = model.em_diagnostics[0]
                self.assertEqual(diagnostics['stop_reason'], 'rejected_nonmonotone_proposal')
                self.assertEqual(diagnostics['nonmonotone_proposal_rejections'], 1)
                self.assertFalse(diagnostics['converged'])
                for old, new in zip(expected, model.list_alpha[0][0]):
                    np.testing.assert_allclose(old, new, rtol=1e-12, atol=1e-12)
                self.assertEqual(np.diff(model.list_all_log_likelihood[0])[0], 0.)
                if name == 'mghp':
                    self.assertAlmostEqual(mean_gig(*model.list_alpha[0][0][:3]), 1.)
                else:
                    self.assertTrue(np.isposinf(model.list_alpha[0][0][-1]).all())

    def test_legacy_damping_unchanged(self):
        model = SyntheticLegacy()
        _, alpha, likelihood, _ = model._fit(
            np.zeros((1, 1)), np.array([1.]), [0.], lambda: None)
        self.assertEqual(alpha, [1.5])
        self.assertEqual(likelihood, -.25)
        self.assertEqual(model._nonmonotone_proposal_rejections, 0)

    def test_small_likelihood_change_requires_final_stationarity(self):
        rng = np.random.default_rng(61)
        x = np.r_[rng.normal([-1.5, 0], .8, size=(100, 2)),
                  rng.normal([1.5, 0], .8, size=(100, 2))]
        alpha = [(np.array([-.25, 0]), np.diag([2., 1.])),
                 (np.array([.25, 0]), np.diag([2., 1.]))]
        model = RecordStationarity(
            2, list_of_dist=['mvn'], candidate_combinations=[('mvn', 'mvn')],
            candidate_initializations={('mvn', 'mvn'): ([.5, .5], alpha)},
            numerical_policy={}, tol=1., max_iter=4, verbose=False)
        model.fit(x)
        self.assertFalse(model.candidate_failures)
        self.assertGreater(model.seen[0][0], 1e-4)
        self.assertEqual(model.em_diagnostics[0]['iterations'], 4)
        self.assertEqual(model.em_diagnostics[0]['stop_reason'], 'max_iter')
        self.assertFalse(model.em_diagnostics[0]['converged'])

    def test_slash_domain_is_explicit(self):
        self.assertEqual(EMPolicy().slash_df_max, 450.)
        for name in ('msl', 'mssl'):
            _bounds(name, 5, EMPolicy())
            with self.assertRaisesRegex(ValueError, 'slash_df_max'):
                _bounds(name, 5, EMPolicy(slash_df_max=500.))

    def test_weight_floor_kkt(self):
        counts = np.array([.001, 1., 10.])
        pi = bounded_weights(counts, .1)
        np.testing.assert_allclose(pi, [.1, .1, .8])
        multiplier = counts[2] / pi[2]
        self.assertTrue(np.all(counts[:2] / pi[:2] <= multiplier))

    def test_analytic_component_gradients_and_zero_weights(self):
        rng = np.random.default_rng(934)
        for name in ('mvn', 'mvt', 'mvsn'):
            for p in (1, 2, 5):
                with self.subTest(name=name, p=p):
                    x, w = rng.normal(size=(35, p)), rng.uniform(.1, 1., size=35)
                    alpha = (np.zeros(p), np.eye(p))
                    if name == 'mvt':
                        alpha += (np.array([5.]),)
                    elif name == 'mvsn':
                        alpha += (np.linspace(-.3, .4, p),)
                    native, _ = prepare_component(name, alpha, EMPolicy())
                    u, _ = _pack(name, native, EMPolicy(), False)
                    _, gradient = _analytic(x, u, name, EMPolicy(), w)
                    def objective(v):
                        a, _ = _unpack(v, name, p, EMPolicy())
                        return float(w @ dist_map[name].logpdf(x, *a))
                    numerical = approx_derivative(objective, u, method='3-point').reshape(-1)
                    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-6)
                    w[0] = 0.
                    x[0] = np.nan
                    _, diagnostics = fit_component(name, x, w, native, EMPolicy(component_maxiter=5))
                    self.assertTrue(diagnostics['accepted'])
                    self.assertTrue(diagnostics['finite'])
                    self.assertGreaterEqual(diagnostics['weighted_loglik'], diagnostics['weighted_loglik_before']-1e-8)

    def test_public_parallel_matches_sequential(self):
        rng = np.random.default_rng(211)
        x = np.r_[rng.normal([-2, 0], .7, size=(40, 2)), rng.normal([2, 0], .9, size=(40, 2))]
        candidates = [('mvn', 'mvn'), ('mvn', 'mvt')]
        starts = {}
        for candidate in candidates:
            alpha = []
            for j, name in enumerate(candidate):
                a = (np.array([-2. if j == 0 else 2., 0.]), np.eye(2))
                if name == 'mvt':
                    a += (np.array([5.]),)
                alpha.append(a)
            starts[candidate] = ([.5, .5], alpha)
        answers = []
        for workers in (1, 2):
            model = fmvmm(2, candidate_combinations=candidates,
                          candidate_initializations=starts, numerical_policy={},
                          max_iter=5, n_jobs=workers, verbose=False)
            model.fit(x)
            self.assertFalse(model.candidate_failures)
            answers.append(model.list_bic)
            for trace in model.list_all_log_likelihood:
                self.assertGreaterEqual(np.min(np.diff(trace)), -1e-8)
        np.testing.assert_allclose(*answers, rtol=0., atol=1e-10)


if __name__ == '__main__':
    unittest.main()
