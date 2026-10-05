"""Behavioural tests for selection, stopping, accounting and beam information use."""
import copy
import itertools
import unittest
from types import SimpleNamespace

import numpy as np

from adaptive import replay_beam, run_greedy


class DeadlineExceeded(BaseException):
    pass


def qualified(row):
    d = row.get('diagnostics', {})
    return bool(row.get('qualified') and d.get('converged')
                and d.get('final_stationarity_acceptable') and not d.get('hit_cap'))


def fitted(names, bic, q=None, ok=True):
    names = tuple(names)
    return dict(names=names, structure=tuple(sorted(names)), bic=bic,
                alpha=[(np.array([0.]),) for _ in names], pi=np.ones(len(names)) / len(names),
                qualified=ok, diagnostics=dict(converged=ok,
                final_stationarity_acceptable=ok, hit_cap=False),
                q=np.tile(np.ones(len(names)) / len(names), (4, 1)) if q is None else q)


class FakeDistribution:
    def logpdf(self, x, *alpha):
        return np.full(len(x), -100.)

    def total_params(self, *alpha):
        return 1


class FakeExperiment:
    def __init__(self, fitted_results, initial_gaussian_bic=101., failure=None):
        self.x = np.arange(4.)[:, None]
        self.k, self.library = 2, ['mvn', 'mvt', 'mvsn']
        self.pilots = dict(student=fitted(('mvt', 'mvt'), 100.),
                           gaussian=fitted(('mvn', 'mvn'), initial_gaussian_bic))
        self.ops = {'pilot:' + k: v for k, v in self.pilots.items()}
        self.pilot_construction = {'additional_incumbents': {}}
        self.results, self.profile_calls, self.fit_calls = list(fitted_results), [], []
        self.failure = failure
        for j in range(self.k):
            for name in self.library:
                self.ops[f'profile:student:{j}:{name}'] = self.profile_row(name)

    def profile_row(self, name):
        return dict(name=name, alpha=(np.array([0.]),), score=10. + self.library.index(name),
                    diagnostics=dict(gradient_acceptable=True))

    def profile(self, label, j, name):
        key = f'profile:{label}:{j}:{name}'
        self.profile_calls.append((label, j, name, self.pilots[label]['bic']))
        if self.failure is not None and j == 0 and name == 'mvn':
            raise self.failure
        row = self.profile_row(name)
        self.ops[key] = row
        return key, row

    def full_fit(self, names, pi, alpha):
        self.fit_calls.append(tuple(names))
        answer = self.results.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        answer = copy.deepcopy(answer)
        answer['names'], answer['structure'] = tuple(names), tuple(sorted(names))
        return answer


def backend_for(exp):
    return SimpleNamespace(np=np, DeadlineExceeded=DeadlineExceeded,
        mixture=lambda x, row: (np.full(len(x), -2.), row['q']),
        dist_map={name: FakeDistribution() for name in exp.library})


def run(exp, **kw):
    return run_greedy(exp, backend_for(exp), qualified=qualified,
        evaluate=lambda exp, row, backend: {'test_log_density': -row['bic']},
        numerical_error=lambda error, backend: isinstance(error, (DeadlineExceeded, RuntimeError)), **kw)


class GreedyTests(unittest.TestCase):
    def test_accepts_only_improvement_and_uses_new_memberships(self):
        q = np.array([[.9, .1], [.8, .2], [.3, .7], [.2, .8]])
        exp = FakeExperiment([fitted(('mvsn',) * 2, 90., q), fitted(('mvsn',) * 2, 91., q)])
        original = copy.deepcopy(exp.ops['profile:student:0:mvt'])
        answer = run(exp)
        self.assertEqual(answer['stop_reason'], 'no_bic_improvement')
        self.assertEqual(answer['bic'], 90.)
        self.assertEqual(answer['accepted_rounds'], 1)
        self.assertEqual(answer['incremental_fit_attempts'], 2)
        self.assertEqual(answer['incremental_profile_attempts'], 6)
        self.assertTrue(all(call[3] == 90. for call in exp.profile_calls))
        self.assertGreater(answer['rounds'][0]['posterior_mean_tv_change'], 0.)
        self.assertEqual(exp.ops['profile:student:0:mvt']['score'], original['score'])
        self.assertGreaterEqual(answer['incremental_accounted_seconds'], answer['incremental_operation_seconds'])

    def test_current_family_can_be_refitted_and_round_cap_is_not_convergence(self):
        exp = FakeExperiment([fitted(('mvsn',) * 2, 90.), fitted(('mvsn',) * 2, 80.)])
        answer = run(exp, max_rounds=2)
        self.assertEqual(exp.fit_calls, [('mvsn', 'mvsn')] * 2)
        self.assertEqual(answer['status'], 'round_cap')
        self.assertEqual(answer['accepted_rounds'], 2)

    def test_failed_profiles_all_count_and_preserve_last_accepted_fit(self):
        exp = FakeExperiment([fitted(('mvsn',) * 2, 90.)], failure=DeadlineExceeded('timeout'))
        answer = run(exp)
        self.assertEqual(answer['stop_reason'], 'unresolved_profile_table')
        self.assertEqual(answer['incremental_profile_attempts'], 6)
        self.assertEqual(answer['failed_profile_attempts'], 1)
        self.assertEqual(answer['bic'], 90.)
        failed = [event for event in answer['events'] if event['status'] == 'deadline']
        self.assertGreater(failed[0]['seconds'], 0.)

    def test_unqualified_better_bic_never_accepted(self):
        exp = FakeExperiment([fitted(('mvsn',) * 2, 1., ok=False)])
        answer = run(exp)
        self.assertEqual(answer['bic'], 100.)
        self.assertEqual(answer['unqualified_fit_attempts'], 1)
        self.assertEqual(answer['accepted_rounds'], 0)

    def test_original_initial_incumbents_still_compete(self):
        exp = FakeExperiment([fitted(('mvsn',) * 2, 90.)], initial_gaussian_bic=80.)
        exp.pilot_construction['additional_incumbents']['pilot_attempt:1'] = fitted(('mvn',) * 2, 75.)
        answer = run(exp, max_rounds=1)
        self.assertEqual(answer['bic'], 75.)
        self.assertEqual(answer['greedy_bic'], 90.)
        self.assertEqual(answer['winning_key'], 'pilot_attempt:1')
        self.assertEqual(answer['quality']['test_log_density'], -75.)

    def test_infrastructure_error_is_checkpointed_and_raised(self):
        exp = FakeExperiment([TypeError('bug')])
        snapshots = []
        with self.assertRaises(TypeError):
            run(exp, checkpoint=lambda result: snapshots.append(copy.deepcopy(result)))
        self.assertEqual(snapshots[-1]['status'], 'infrastructure_error')
        self.assertEqual(snapshots[-1]['incremental_fit_attempts'], 1)

    def test_incumbent_alpha_is_a_feasible_score_candidate(self):
        exp = FakeExperiment([fitted(('mvt',) * 2, 100.)])
        for j in range(exp.k):
            for family in exp.library:
                exp.ops[f'profile:student:{j}:{family}']['score'] = -1000.
        answer = run(exp)
        self.assertEqual(exp.fit_calls[0], ('mvt', 'mvt'))
        self.assertTrue(all(v['selected'] for v in answer['rounds'][0]['incumbent_component_candidates']))


class BeamTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = [{'structure': names} for names in
                          itertools.combinations_with_replacement(['mvn', 'mvt', 'mvsn'], 2)]
        self.records = {i: dict(status='qualified', qualified=True, row=fitted(row['structure'], 100. - i),
                               seconds=3., fit_seconds=2., quality={'test_log_density': -10.})
                        for i, row in enumerate(self.catalogue)}

    def test_default_budget_one_uses_mvn_only_and_charges_it(self):
        answer = replay_beam(self.catalogue, self.records, budget=1, qualified=qualified)
        self.assertEqual(answer['evaluated_indices'], [0])
        self.assertEqual(answer['attempted_complete_fits'], 1)
        self.assertEqual(answer['candidate_seconds'], 3.)

    def test_failures_consume_budget_and_cannot_win(self):
        self.records[0] = dict(status='deadline', seconds=10.)
        answer = replay_beam(self.catalogue, self.records, budget=3, seed=8, qualified=qualified)
        self.assertEqual(answer['attempted_complete_fits'], 3)
        self.assertEqual(answer['fit_status_counts']['deadline'], 1)
        self.assertNotEqual(answer['winning_index'], 0)
        self.assertEqual(answer['candidate_seconds'], 16.)

    def test_beam_can_cross_a_worse_intermediate_bic(self):
        catalogue = [{'structure': s} for s in [('mvn', 'mvn'), ('mvn', 'mvt'), ('mvt', 'mvt')]]
        records = {i: dict(status='qualified', row=fitted(row['structure'], bic))
                   for i, (row, bic) in enumerate(zip(catalogue, [0., 20., 10.]))}
        answer = replay_beam(catalogue, records, budget=3, width=1,
                             initial_indices=[2], qualified=qualified)
        self.assertEqual(answer['evaluated_indices'], [2, 1, 0])
        self.assertEqual(answer['winning_index'], 0)

    def test_missing_and_infrastructure_results_are_not_statistical_failures(self):
        del self.records[0]
        with self.assertRaisesRegex(ValueError, 'missing'):
            replay_beam(self.catalogue, self.records, budget=1, qualified=qualified)
        self.records[0] = {'status': 'infrastructure_error'}
        with self.assertRaisesRegex(ValueError, 'infrastructure'):
            replay_beam(self.catalogue, self.records, budget=1, qualified=qualified)

    def test_unvisited_outcomes_are_not_inspected(self):
        class Watched(dict):
            def __getitem__(self, key):
                accesses.append(key)
                return super().__getitem__(key)
        accesses = []
        answer = replay_beam(self.catalogue, Watched(self.records), budget=3,
                             qualified=qualified, seed=12)
        self.assertEqual(set(accesses), set(answer['evaluated_indices']))
        self.assertEqual(len(answer['evaluated_indices']), len(set(answer['evaluated_indices'])))

    def test_qualified_pilot_rescues_failed_homogeneous_vertex(self):
        self.records[0] = dict(status='deadline', seconds=9.)
        incumbent = dict(status='qualified', row=fitted(('mvn', 'mvn'), 1.),
                         quality={'test_log_density': -3.})
        answer = replay_beam(self.catalogue, self.records, budget=2, width=1,
                             initial_indices=[0], qualified=qualified,
                             vertex_incumbents={0: incumbent})
        self.assertEqual(answer['attempted_complete_fits'], 2)
        self.assertEqual(answer['qualified_complete_fits'], 1)
        self.assertEqual(answer['qualified_vertices'], 2)
        self.assertEqual(answer['candidate_seconds'], 12.)
        self.assertEqual(answer['winning_index'], 0)
        self.assertEqual(answer['winning_source'], 'initial_incumbent')
        self.assertEqual(answer['quality']['test_log_density'], -3.)

    def test_unvisited_incumbents_do_not_enter_search(self):
        class Watched(dict):
            def __getitem__(self, key):
                accesses.append(key)
                return super().__getitem__(key)
        accesses = []
        incumbents = Watched({i: dict(row=fitted(row['structure'], 1.))
                              for i, row in enumerate(self.catalogue)})
        answer = replay_beam(self.catalogue, self.records, budget=1,
                             qualified=qualified, vertex_incumbents=incumbents)
        self.assertEqual(accesses, [0])
        self.assertEqual(answer['evaluated_indices'], [0])


if __name__ == '__main__':
    unittest.main()
