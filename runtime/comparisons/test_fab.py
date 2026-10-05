"""Independent formula checks and optional real frozen-backend FAB smoke."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import minimize

from fab import FABUnresolved, _verify_cached_profile, family_objectives, lower_bound, numerical_error, run_fab, v_step


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.q = np.array([[.8, .2], [.3, .7], [.6, .4]])
        self.logs = np.array([[-1.2, -2.3], [-.7, -.4], [-2., -.9]])
        self.pi = np.array([.65, .35])
        self.dim = np.array([2., 5.])

    def test_explicit_complete_data_formula(self):
        actual = lower_bound(self.q, self.logs, self.pi, self.dim)
        expected = 0.
        for i in range(3):
            for j in range(2):
                expected += self.q[i, j] * (self.logs[i, j] + math.log(self.pi[j]) - math.log(self.q[i, j]))
        expected -= .5 * math.log(3)
        for j in range(2):
            expected -= .5 * self.dim[j] * math.log(sum(self.q[:, j]))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_v_step_matches_independent_numerical_maximization(self):
        q_new = v_step(self.q, self.logs, self.pi, self.dim)
        def loss(first):
            trial = np.column_stack([first, 1 - first])
            return -lower_bound(trial, self.logs, self.pi, self.dim, tangent_q=self.q)
        optimum = minimize(loss, [.5] * 3, method='L-BFGS-B', bounds=[(1e-8, 1 - 1e-8)] * 3,
                           options={'ftol': 1e-14, 'gtol': 1e-8})
        np.testing.assert_allclose(q_new[:, 0], optimum.x, atol=3e-6, rtol=0)
        self.assertGreaterEqual(lower_bound(q_new, self.logs, self.pi, self.dim, tangent_q=self.q),
                                lower_bound(self.q, self.logs, self.pi, self.dim) - 1e-12)

    def test_tangent_reset_is_ascent(self):
        q_new = v_step(self.q, self.logs, self.pi, self.dim)
        self.assertGreaterEqual(lower_bound(q_new, self.logs, self.pi, self.dim),
                                lower_bound(q_new, self.logs, self.pi, self.dim, tangent_q=self.q) - 1e-12)

    def test_uses_actual_counts_with_bounded_weights(self):
        q_new = v_step(self.q, self.logs, self.pi, self.dim)
        unnormalized = np.empty_like(self.q)
        for i in range(3):
            for j in range(2):
                unnormalized[i, j] = self.pi[j] * math.exp(self.logs[i, j] - self.dim[j] / (2 * sum(self.q[:, j])))
        expected = unnormalized / unnormalized.sum(1)[:, None]
        np.testing.assert_allclose(q_new, expected, rtol=1e-13)
        wrong = np.exp(self.logs + np.log(self.pi) - self.dim / (2 * 3 * self.pi))
        wrong /= wrong.sum(1)[:, None]
        self.assertGreater(np.max(np.abs(wrong - q_new)), 1e-3)

    def test_component_penalties_use_log_effective_count(self):
        ll = np.array([[-4., -3.], [-5., -4.]])
        dim = np.array([[2., 4.], [2., 4.]])
        count = np.array([1., 100.])
        scores = family_objectives(ll, dim, count)
        self.assertEqual(tuple(scores.argmax(1)), (1, 0))
        for j in range(2):
            for h in range(2):
                self.assertAlmostEqual(scores[j, h], ll[j, h] - .5 * dim[j, h] * math.log(count[j]))

    def test_empty_component_and_invalid_memberships_fail(self):
        with self.assertRaises(FABUnresolved):
            v_step(np.array([[1., 0.], [1., 0.]]), np.zeros((2, 2)), self.pi, self.dim)
        with self.assertRaises(FABUnresolved):
            lower_bound(self.q * 2, self.logs, self.pi, self.dim)

    def test_programming_errors_are_not_numerical_failures(self):
        class Deadline(BaseException):
            pass
        backend = SimpleNamespace(DeadlineExceeded=Deadline)
        for error in (KeyError('missing'), NameError('typo'), TypeError('wrong type'),
                      ImportError('dependency'), ValueError('wrong shape'),
                      RuntimeError('unexpected'), FloatingPointError('Stationarity objective disagrees with native density.')):
            self.assertFalse(numerical_error(error, backend))
        for error in (Deadline(), FABUnresolved('missing family'), np.linalg.LinAlgError('singular'),
                      OverflowError('overflow'), FloatingPointError('Nonfinite weighted objective/gradient.'),
                      RuntimeError('Unresolved weighted profile x')):
            self.assertTrue(numerical_error(error, backend))

    def test_cached_profile_omits_zero_weight_underflow(self):
        calls = []
        def logpdf(x):
            calls.append(x.copy())
            return np.where(x[:, 0] == 0, -2., -np.inf)
        backend = SimpleNamespace(dist_map={'test': SimpleNamespace(logpdf=logpdf)})
        x = np.array([[0.], [1.]])
        profile = dict(alpha=[], weighted_loglik=-2., dimension=3)
        score = -4 - 3 * math.log(2)
        self.assertEqual(_verify_cached_profile(x, np.array([1., 0.]), 'test', profile, backend, score), -2.)
        np.testing.assert_array_equal(calls[0], x[:1])
        with self.assertRaises(FABUnresolved):
            _verify_cached_profile(x, np.array([.5, .5]), 'test', profile, backend, score)

    def test_cached_profile_nonfinite_values_cannot_pass_tolerance(self):
        backend = SimpleNamespace(dist_map={'test': SimpleNamespace(logpdf=lambda x: np.full(len(x), -2.))})
        x, w = np.array([[0.], [1.]]), np.array([1., 0.])
        good = dict(alpha=[], weighted_loglik=-2., dimension=3)
        for bad_value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(FABUnresolved):
                _verify_cached_profile(x, w, 'test', dict(good, weighted_loglik=bad_value), backend)
            with self.assertRaises(FABUnresolved):
                _verify_cached_profile(x, w, 'test', good, backend, bad_value)
            with self.assertRaises(FABUnresolved):
                _verify_cached_profile(x, w, 'test', dict(good, dimension=bad_value), backend)


def native_smoke(output, max_rounds, reuse_operations=None):
    root = Path(__file__).resolve().parents[2]
    bundle = root / 'Study/paper_study_20260917/prepared/fmvmm-paper-l13-20260916-v2/bundle'
    spec = importlib.util.spec_from_file_location('fab_frozen_runner', bundle / 'runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    ctx = runner.context(bundle)
    backend = ctx['backend']
    source = root / 'Study/paper_study_20260917/local_smoke_v2/tables/remote_smoke_l13.json'
    table = backend.loads(source.read_text())
    exp = runner.restore(ctx, table)
    reused = []
    if reuse_operations is not None:
        previous = backend.loads(Path(reuse_operations).read_text())
        previous = previous.get('result', previous)
        for key, value in previous.get('operations', {}).items():
            if key.startswith('fab:profile:') and value.get('qualified'):
                exp.ops[key] = value
                reused.append(key)
    sources = {str(path): runner.sha256(path) for path in (Path(__file__), Path(__file__).with_name('fab.py'))}
    original = backend.dumps(exp.pilots['student'], sort_keys=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save(progress):
        runner.atomic_write(output.with_suffix('.progress.json'),
                            dict(active_operation=exp.active_operation,
                                 result=progress), backend)
    result = run_fab(exp, backend, evaluate=lambda row: runner.evaluate(exp, row, backend),
                     initial_scores=table['scores'], max_rounds=max_rounds, checkpoint=save)
    result['native_smoke_source'] = str(source)
    result['native_smoke_source_sha256'] = runner.sha256(source)
    result['implementation_sha256'] = sources
    result['reused_development_profile_keys'] = reused
    if sources != {path: runner.sha256(path) for path in sources}:
        raise AssertionError('FAB source changed during native validation')
    result['operations'] = {k: v for k, v in exp.ops.items() if k.startswith('fab:')}
    if backend.dumps(exp.pilots['student'], sort_keys=True) != original:
        raise AssertionError('Original MVT changed')
    if not np.isfinite(result['selected']['bic']):
        raise AssertionError('Nonfinite BIC diagnostic')
    if not all(item['ascent_verified'] for item in result['ascent_checks']):
        raise AssertionError('Ascent check failed')
    runner.atomic_write(output, result, backend)
    print(json.dumps({key: result.get(key) for key in ('status', 'qualified', 'completed_rounds', 'objective',
                                                     'elapsed_seconds', 'weighted_fit_seconds', 'error')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-smoke', type=Path)
    parser.add_argument('--max-rounds', type=int, default=30)
    parser.add_argument('--reuse-operations', type=Path)
    args, remaining = parser.parse_known_args()
    if args.native_smoke:
        native_smoke(args.native_smoke, args.max_rounds, args.reuse_operations)
    else:
        unittest.main(argv=[sys.argv[0]] + remaining)
