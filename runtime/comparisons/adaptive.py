"""Explicit computational alternatives to one-table family screening.

The greedy method updates memberships using the frozen native FMVMM adapter.
It is a BIC family-update comparator, not an implementation of EMDL or FAB.
The beam method searches a finite graph of previously fitted combinations and
uses attained BIC to select its beam. Neither method claims global optimality.
No function writes to an input checkpoint or alters the native fitting code.
"""
from __future__ import annotations

import copy
import math
import random
import time
import traceback


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer.')


def run_greedy(exp, backend, *, evaluate, qualified, numerical_error,
               max_rounds=10, bic_tolerance=1e-6, checkpoint=None):
    """Update families from accepted fitted memberships, with full accounting.

    ``exp`` must be a private experiment restored from a qualified old table.
    All of its original Student profiles are reused in round zero. Subsequent
    rounds fit all families with the current posterior by the *unchanged*
    ``exp.profile`` method. For the currently fitted family, its current
    parameter value is also a feasible weighted-score candidate. This prevents
    a worse numerical profile from discarding a known feasible component.

    The rowwise highest scores propose one ordered assignment. Its weighted
    component estimates and current mixing weights initialize a native full
    mixture fit. Only a qualified decrease in observed BIC greater than
    ``bic_tolerance`` is accepted. Identical family names do not alone stop an
    update: a new parameter fit may improve BIC. Every termination reason,
    including the round cap and numerical failure, remains explicit.

    ``checkpoint(result)`` is called at operation boundaries. Initial table and
    initial mixture costs are excluded from incremental costs and must be added
    once by the caller. Failed new operations are included. ``quality`` is
    evaluated after selection, never used to choose families, and separately
    timed. As for every fixed-table comparator, all
    qualified initial-mixture and retry incumbents stored on ``exp`` are already
    included in the returned final winner. ``greedy_row`` separately records
    the final accepted update trajectory.
    """
    _positive_integer(max_rounds, 'max_rounds')
    if not math.isfinite(bic_tolerance) or bic_tolerance < 0:
        raise ValueError('bic_tolerance must be finite and nonnegative.')
    np = backend.np
    current = copy.deepcopy(exp.pilots['student'])
    if not qualified(current) or not math.isfinite(current['bic']):
        raise ValueError('Greedy updates require a qualified initial MVT fit.')
    if len(current['names']) != exp.k:
        raise ValueError('Initial fit has the wrong component count.')
    if any(name not in exp.library for name in current['names']):
        raise ValueError('Initial family is outside the comparison library.')
    initial_incumbents = {key: copy.deepcopy(row) for key, row in exp.ops.items()
                          if key.startswith('pilot:') and qualified(row)
                          and math.isfinite(row.get('bic', math.nan))}
    construction = getattr(exp, 'pilot_construction', None) or {}
    initial_incumbents.update({key: copy.deepcopy(row) for key, row in
                              construction.get('additional_incumbents', {}).items()
                              if qualified(row) and math.isfinite(row.get('bic', math.nan))})
    for j in range(exp.k):
        for family in exp.library:
            key = f'profile:student:{j}:{family}'
            row = exp.ops[key]
            if not row['diagnostics'].get('gradient_acceptable'):
                raise ValueError('Initial table contains an unresolved profile.')
            if not math.isfinite(float(row['score'])):
                raise ValueError('Initial table contains a nonfinite score.')
    events, rounds = [], []
    result = dict(method='greedy_family_update', status='running',
                  parameters=dict(max_rounds=max_rounds, bic_tolerance=bic_tolerance),
                  initial_bic=float(current['bic']), row=current, rounds=rounds,
                  events=events, accepted_rounds=0, stop_reason=None,
                  reused_initial_profiles=exp.k * len(exp.library),
                  initial_cost_scope='Add the shared initial-mixture and full score-table cost once.',
                  timing_scope='Incremental weighted and complete fitting operations; quality evaluation separate.')
    started = time.perf_counter()

    def save():
        result['greedy_row'] = current
        if result['status'] == 'running':
            result['row'] = current
        result['incremental_profile_attempts'] = sum(e['kind'] == 'profile' for e in events)
        result['incremental_fit_attempts'] = sum(e['kind'] == 'fit' for e in events)
        result['incremental_profile_seconds'] = sum(e.get('seconds', 0.) for e in events if e['kind'] == 'profile')
        result['incremental_fit_seconds'] = sum(e.get('seconds', 0.) for e in events if e['kind'] == 'fit')
        result['incremental_operation_seconds'] = (result['incremental_profile_seconds']
                                                 + result['incremental_fit_seconds'])
        result['failed_profile_attempts'] = sum(e['kind'] == 'profile' and e['status'] in {'failed', 'deadline', 'infrastructure_error'} for e in events)
        result['unqualified_fit_attempts'] = sum(e['kind'] == 'fit' and e['status'] == 'unqualified' for e in events)
        result['failed_fit_attempts'] = sum(e['kind'] == 'fit' and e['status'] in {'failed', 'deadline', 'infrastructure_error'} for e in events)
        result['elapsed_seconds'] = time.perf_counter() - started
        result['incremental_accounted_seconds'] = max(result['incremental_operation_seconds'],
            result['elapsed_seconds'] - result.get('quality_evaluation_seconds', 0.))
        result['incremental_overhead_seconds'] = (result['incremental_accounted_seconds']
                                                 - result['incremental_operation_seconds'])
        if checkpoint is not None:
            checkpoint(result)

    def operation(kind, round_index, key, call):
        event = dict(kind=kind, round=round_index, key=key, status='running')
        events.append(event)
        save()
        begin = time.perf_counter()
        try:
            value = call()
            row = value[1] if kind == 'profile' else value
            event['row'] = copy.deepcopy(row)
            event['status'] = ('qualified' if kind == 'profile' or qualified(row)
                               else 'unqualified')
            return value, event
        except BaseException as error:
            recoverable = numerical_error(error, backend)
            event.update(status='deadline' if isinstance(error, backend.DeadlineExceeded) else
                         'failed' if recoverable else 'infrastructure_error',
                         error=str(error), exception_type=type(error).__name__,
                         traceback=traceback.format_exc())
            if kind == 'profile' and key in exp.ops:
                event['row'] = copy.deepcopy(exp.ops[key])
            if not recoverable:
                result.update(status='infrastructure_error', stop_reason='infrastructure_error')
                raise
            return None, event
        finally:
            event['seconds'] = time.perf_counter() - begin
            save()

    try:
        for round_index in range(max_rounds):
            # Component labels are carried forward unchanged by the native fit.
            _, q = backend.mixture(exp.x, current)
            q = np.asarray(q, dtype=float)
            if (q.shape != (len(exp.x), exp.k) or not np.isfinite(q).all()
                    or np.any(q < 0) or not np.allclose(q.sum(1), 1., atol=1e-12, rtol=0)):
                raise ValueError('Current native posterior is invalid.')
            label = 'student' if round_index == 0 else f'comparison_greedy_round_{round_index}'
            if round_index:
                if label in exp.pilots or any(key.startswith('profile:' + label + ':') for key in exp.ops):
                    raise ValueError('Greedy experiment is not a fresh private restore.')
                exp.pilots[label] = copy.deepcopy(current)
            record = dict(round=round_index, bic_before=float(current['bic']),
                          names_before=list(current['names']), profile_label=label,
                          accepted=False, incumbent_component_candidates=[])
            rounds.append(record)
            profiles, scores, unresolved = {}, np.full((exp.k, len(exp.library)), np.nan), []
            for j in range(exp.k):
                for h, name in enumerate(exp.library):
                    key = f'profile:{label}:{j}:{name}'
                    if round_index == 0:
                        row = exp.ops[key]
                    else:
                        value, event = operation('profile', round_index, key,
                                                 lambda: exp.profile(label, j, name))
                        if value is None:
                            unresolved.append(key)
                            continue
                        row = value[1]
                    # Preserve the existing component as an explicit feasible
                    # score candidate, without changing the cached profile row.
                    selected = copy.deepcopy(row)
                    if not selected['diagnostics'].get('gradient_acceptable') or not math.isfinite(float(selected['score'])):
                        unresolved.append(key)
                        continue
                    if name == current['names'][j]:
                        weights = q[:, j]
                        positive = weights > 0
                        ll = float(np.dot(weights[positive], backend.dist_map[name].logpdf(
                            exp.x[positive], *current['alpha'][j])))
                        dimension = int(backend.dist_map[name].total_params(*current['alpha'][j]))
                        incumbent_score = 2 * ll - dimension * np.log(len(exp.x))
                        if not math.isfinite(incumbent_score):
                            raise ValueError('Current qualified component has a nonfinite weighted score.')
                        use_incumbent = bool(incumbent_score > selected['score'])
                        record['incumbent_component_candidates'].append(dict(
                            component=j, family=name, fitted_profile_score=float(selected['score']),
                            incumbent_score=float(incumbent_score), selected=use_incumbent))
                        if use_incumbent:
                            selected.update(alpha=copy.deepcopy(current['alpha'][j]),
                                            score=float(incumbent_score), weighted_loglik=ll,
                                            dimension=dimension, source='accepted_mixture_component')
                    profiles[j, name] = selected
                    scores[j, h] = selected['score']
            record['scores'] = scores
            if unresolved:
                record['unresolved_profiles'] = unresolved
                result.update(status='numerical_failure', stop_reason='unresolved_profile_table')
                break
            # Lexicographic family names break exact ties, independently of truth.
            names = tuple(min(exp.library, key=lambda name: (-scores[j, exp.library.index(name)], name))
                          for j in range(exp.k))
            alpha = [copy.deepcopy(profiles[j, name]['alpha']) for j, name in enumerate(names)]
            record['proposed_names'] = list(names)
            proposed, event = operation('fit', round_index, 'greedy_fit:' + str(round_index),
                                       lambda: exp.full_fit(names, copy.deepcopy(current['pi']), alpha))
            if proposed is None:
                result.update(status='numerical_failure', stop_reason='complete_fit_failure')
                break
            record['proposed_bic'] = float(proposed['bic'])
            record['proposal_qualified'] = bool(qualified(proposed))
            if not qualified(proposed) or not math.isfinite(float(proposed['bic'])):
                result.update(status='numerical_failure', stop_reason='unqualified_complete_fit')
                break
            improvement = float(current['bic'] - proposed['bic'])
            record['bic_improvement'] = improvement
            if improvement <= bic_tolerance:
                result.update(status='complete', stop_reason='no_bic_improvement')
                break
            _, new_q = backend.mixture(exp.x, proposed)
            record['posterior_mean_tv_change'] = float(.5 * np.abs(q - new_q).sum(1).mean())
            record['accepted'] = True
            current = copy.deepcopy(proposed)
            result['accepted_rounds'] += 1
            save()
        else:
            result.update(status='round_cap', stop_reason='maximum_rounds')
        candidates = dict(initial_incumbents, greedy_accepted=current)
        winning_key = min(candidates, key=lambda key: (candidates[key]['bic'], key))
        winner = candidates[winning_key]
        result['winning_key'] = winning_key
        result['row'] = copy.deepcopy(winner)
        result['greedy_bic'] = float(current['bic'])
        quality_started = time.perf_counter()
        result['quality'] = evaluate(exp, winner, backend)
        result['quality_evaluation_seconds'] = time.perf_counter() - quality_started
        result['winner_qualified'] = bool(qualified(winner))
        result['bic'] = float(winner['bic'])
        result['structure'] = list(winner['structure'])
    except BaseException as error:
        result.update(status='infrastructure_error', stop_reason='infrastructure_error',
                      error=str(error), exception_type=type(error).__name__,
                      traceback=traceback.format_exc())
        raise
    finally:
        save()
    return result


def replay_beam(catalogue, fit_records, *, budget, qualified, width=3, seed=0,
                initial_indices=None, vertex_incumbents=None):
    """Replay a BIC beam search on the one-family-replacement graph.

    Each catalogue vertex is an unordered family combination with its frozen,
    score-assigned native complete fit. An edge replaces one family. Starting
    vertices default to the all-MVN combination followed by all-MVT if the
    budget permits. Every new vertex evaluated,
    including a numerical failure, consumes one unit of ``budget``. Unseen
    neighbours are ordered by a seeded shuffle, independently of their scores
    and fitted outcomes. At each layer the best ``width`` qualified *new*
    vertices form the next beam; the best overall result is retained separately.
    Consequently the beam can cross worse intermediate combinations and does
    not repeatedly expand the same vertex. No BIC is inspected until a vertex
    enters the evaluated set. Missing records are an incomplete input error,
    never silently treated as an unsuccessful numerical fit.

    ``vertex_incumbents`` may map vertex indices to wrappers containing a
    qualified common initial-mixture fit under ``row`` and its ``quality``.
    Once that vertex is visited, its lower BIC among the catalogue fit and the
    initial fit determines beam ranking. Thus an available qualified initial
    fit can support expansion even when its catalogue refit fails. The latter
    attempt and its full cost are still counted. Incumbents at unvisited
    vertices do not influence expansion.

    Initial mixture costs, weighted-score-table costs (used to initialize the
    frozen fits), and common initial-mixture incumbents are handled by the
    caller. Cached reuse saves physical work but is charged at original fit
    timings, so this is an operation-cost comparison, not a fresh wall-clock
    experiment.
    """
    _positive_integer(budget, 'budget')
    _positive_integer(width, 'width')
    if not catalogue:
        raise ValueError('The beam catalogue must be nonempty.')
    structures = [tuple(sorted(row['structure'])) for row in catalogue]
    k = len(structures[0])
    if not k or len(set(structures)) != len(structures) or any(len(s) != k for s in structures):
        raise ValueError('Catalogue structures must be unique with common positive component count.')
    lookup = {s: i for i, s in enumerate(structures)}
    library = sorted(set(name for s in structures for name in s))
    if initial_indices is None:
        initial_indices = [lookup[(name,) * k] for name in ('mvn', 'mvt')
                           if (name,) * k in lookup][:budget]
    initial_indices = list(dict.fromkeys(initial_indices))
    if not initial_indices or any(i not in range(len(catalogue)) for i in initial_indices):
        raise ValueError('Invalid starting catalogue indices.')
    if budget < len(initial_indices):
        raise ValueError('Budget does not cover the declared starting vertices.')
    rng = random.Random(int(seed))
    sequence, layers, seen, accepted = [], [], set(), {}
    accepted_records, qualified_fits = {}, set()
    vertex_incumbents = {} if vertex_incumbents is None else vertex_incumbents
    terminal = {'qualified', 'unqualified', 'deadline', 'failed', 'interrupted', 'infrastructure_error'}

    def inspect(index):
        if index not in fit_records:
            raise ValueError(f'Beam needs missing fitted record {index}.')
        value = fit_records[index]
        if value.get('status') not in terminal:
            raise ValueError(f'Beam record {index} is not terminal.')
        # Infrastructure failures are not statistical algorithm outcomes.
        if value['status'] in {'interrupted', 'infrastructure_error'}:
            raise ValueError(f'Beam record {index} has infrastructure failure: {value["status"]}.')
        row = value.get('row')
        if row is not None and tuple(sorted(row['names'])) != structures[index]:
            raise ValueError('Beam fitted structure differs from its catalogue vertex.')
        seen.add(index)
        sequence.append(index)
        if (value['status'] == 'qualified' and value.get('qualified', True) is True
                and row is not None and qualified(row) and math.isfinite(row['bic'])):
            accepted[index] = row
            accepted_records[index] = dict(value, winning_source='catalogue_fit')
            qualified_fits.add(index)
        if index in vertex_incumbents:
            incumbent = vertex_incumbents[index]
            initial_row = incumbent.get('row')
            if (initial_row is None or not qualified(initial_row)
                    or not math.isfinite(initial_row['bic'])
                    or tuple(sorted(initial_row['names'])) != structures[index]
                    or incumbent.get('qualified', True) is not True
                    or incumbent.get('status', 'qualified') != 'qualified'):
                raise ValueError('Beam vertex incumbent must be a qualified fit of the same structure.')
            if index not in accepted or initial_row['bic'] < accepted[index]['bic']:
                accepted[index] = initial_row
                accepted_records[index] = dict(incumbent, winning_source='initial_incumbent')

    def rank(indices):
        return sorted((i for i in indices if i in accepted),
                      key=lambda i: (accepted[i]['bic'], structures[i], i))

    for index in initial_indices:
        inspect(index)
    beam = rank(initial_indices)[:width]
    stop_reason = 'no_qualified_initial_vertex' if not beam else None
    while beam and len(sequence) < min(budget, len(catalogue)):
        neighbours = set()
        for centre in beam:
            for j in range(k):
                for name in library:
                    structure = list(structures[centre])
                    structure[j] = name
                    target = lookup.get(tuple(sorted(structure)))
                    if target is not None and target not in seen:
                        neighbours.add(target)
        if not neighbours:
            stop_reason = 'no_unvisited_neighbour'
            break
        order = sorted(neighbours)
        rng.shuffle(order)
        planned = order[:budget - len(sequence)]
        layer = dict(beam=list(beam), unevaluated_neighbours=len(order), evaluated_indices=planned,
                     budget_truncated=len(planned) < len(order))
        for index in planned:
            inspect(index)
        beam = rank(planned)[:width]
        layer['next_beam'] = list(beam)
        layers.append(layer)
        if not beam:
            stop_reason = 'no_qualified_neighbour'
    if stop_reason is None:
        stop_reason = 'catalogue_exhausted' if len(sequence) == len(catalogue) else 'fit_budget'
    winner = rank(accepted)[0] if accepted else None
    statuses = {}
    for index in sequence:
        status = fit_records[index]['status']
        statuses[status] = statuses.get(status, 0) + 1
    return dict(method='bic_beam_search', status='complete', parameters=dict(
        budget=budget, width=width, seed=seed, initial_indices=initial_indices),
        evaluated_indices=sequence, layers=layers, stop_reason=stop_reason,
        attempted_complete_fits=len(sequence), qualified_complete_fits=len(qualified_fits),
        qualified_vertices=len(accepted),
        fit_status_counts=statuses, winning_index=winner,
        winning_source=accepted_records[winner]['winning_source'] if winner is not None else None,
        row=copy.deepcopy(accepted_records[winner]['row']) if winner is not None else None,
        quality=copy.deepcopy(accepted_records[winner].get('quality', {})) if winner is not None else {},
        candidate_seconds=sum(fit_records[i].get('seconds', 0.) for i in sequence),
        candidate_fit_seconds=sum(fit_records[i].get('fit_seconds', 0.) for i in sequence),
        cost_scope='Add shared initial-mixture and table costs once; all attempted candidate records charged.')
