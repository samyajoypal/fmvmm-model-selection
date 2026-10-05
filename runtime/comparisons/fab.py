"""Constrained, fixed-K FAB using the frozen FMVMM component likelihoods.

Fujimaki and Morinaga (2012), equations (11), (15), (17)--(19), (21).
This is the fixed-K inner iteration of FABtwo (no component deletion), with
the study's existing parameter domains and bounded mixing weights. It is not
BIC structural EM. No ordinary mixture EM is called after initialization.
The paper's unrestricted regularity/consistency conclusions are not asserted
for these constrained numerical fits.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np
from scipy.special import logsumexp, xlogy


class FABUnresolved(RuntimeError):
    """The full family comparison or an ascent check could not be resolved."""


def numerical_error(error, backend):
    """Do not turn programming/import errors into scientific fit failures."""
    if isinstance(error, (FABUnresolved, backend.DeadlineExceeded, np.linalg.LinAlgError, OverflowError)):
        return True
    if isinstance(error, FloatingPointError):
        return str(error).startswith(('Nonfinite ', 'Overflow ', 'GH boundary encoding would change '))
    return isinstance(error, RuntimeError) and str(error).startswith((
        'No fitted candidate ', 'Unresolved weighted profile ', 'All profile starts invalid ',
        'No eligible pilot among '))


def _memberships(q):
    q = np.asarray(q, dtype=float)
    if (q.ndim != 2 or not np.isfinite(q).all() or np.any(q < 0)
            or not np.allclose(q.sum(1), 1., rtol=0, atol=1e-12)
            or np.any(q.sum(0) <= 0)):
        raise FABUnresolved('Memberships must be normalized with positive column counts.')
    return q


def lower_bound(q, log_components, pi, dimensions, tangent_q=None):
    """G(q, q_tilde, theta), including entropy and all dimension penalties."""
    q = _memberships(q)
    n, k = q.shape
    logs = np.asarray(log_components, dtype=float)
    pi = np.asarray(pi, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    if (logs.shape != q.shape or not np.isfinite(logs).all()
            or pi.shape != (k,) or not np.isfinite(pi).all() or np.any(pi <= 0)
            or not np.isclose(pi.sum(), 1., rtol=0, atol=1e-12)
            or dimensions.shape != (k,) or np.any(dimensions <= 0)
            or not np.isfinite(dimensions).all()):
        raise FABUnresolved('Invalid FAB density, weight, or dimension arrays.')
    counts = q.sum(0)
    if tangent_q is not None and np.shape(tangent_q) != q.shape:
        raise FABUnresolved('Tangent memberships have a different shape.')
    tangent_counts = counts if tangent_q is None else _memberships(tangent_q).sum(0)
    tangent_log_count = np.log(tangent_counts) + (counts - tangent_counts) / tangent_counts
    complete = np.sum(q * (logs + np.log(pi)[None, :]))
    entropy = -np.sum(xlogy(q, q))
    return float(complete + entropy - .5 * (k - 1) * np.log(n)
                 - .5 * np.dot(dimensions, tangent_log_count))


def v_step(q_previous, log_components, pi, dimensions):
    """Equation (15), using actual counts even when weight bounds are active."""
    q_previous = _memberships(q_previous)
    counts = q_previous.sum(0)
    logits = (np.asarray(log_components, dtype=float) + np.log(pi)[None, :]
              - np.asarray(dimensions, dtype=float)[None, :] / (2 * counts[None, :]))
    q = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return _memberships(q)


def family_objectives(weighted_loglik, dimensions, counts):
    """Equation (18): no BIC/log(n) or EMDL/log-star penalty."""
    ll = np.asarray(weighted_loglik, dtype=float)
    d = np.asarray(dimensions, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if ll.ndim != 2 or d.shape != ll.shape or counts.shape != (len(ll),):
        raise FABUnresolved('Incompatible family comparison arrays.')
    if not np.isfinite(ll).all() or np.any(counts <= 0) or not np.isfinite(d).all():
        raise FABUnresolved('Unresolved family objective.')
    return ll - .5 * d * np.log(counts)[:, None]


def _fingerprint(q):
    return hashlib.sha256(np.ascontiguousarray(q, dtype='<f8').tobytes()).hexdigest()


def _verify_cached_profile(x, weights, name, profile, backend, initial_score=None):
    """Recompute a cached weighted likelihood without evaluating zero weights.

    A zero responsibility contributes zero even if the density at that point
    underflows. Nonfinite values at positive weights, in the stored likelihood,
    or in the score are unresolved; NaN cannot pass a tolerance comparison.
    """
    weights = np.asarray(weights, dtype=float)
    positive = weights > 0
    ld = np.asarray(backend.dist_map[name].logpdf(x[positive], *profile['alpha']), dtype=float)
    if ld.shape != weights[positive].shape:
        raise ValueError('Cached profile density has an unexpected shape.')
    checked_ll = float(np.dot(weights[positive], ld))
    recorded_ll = float(profile['weighted_loglik'])
    expected_score = float(2 * checked_ll - profile['dimension'] * np.log(len(weights)))
    if not (np.isfinite(ld).all() and math.isfinite(checked_ll)
            and math.isfinite(recorded_ll) and math.isfinite(expected_score)):
        raise FABUnresolved('Nonfinite cached profile likelihood or score: ' + name)
    if abs(checked_ll - recorded_ll) > 2e-6:
        raise FABUnresolved('Cached profile does not match initial MVT memberships: ' + name)
    if initial_score is not None:
        initial_score = float(initial_score)
        if not math.isfinite(initial_score) or abs(initial_score - expected_score) > 4e-6:
            raise FABUnresolved('Initial score verification failed: ' + name)
    return checked_ll


def _state(exp, backend, row):
    logs = np.column_stack([backend.dist_map[name].logpdf(exp.x, *alpha)
                            for name, alpha in zip(row['names'], row['alpha'])])
    dimensions = np.asarray([int(backend.dist_map[name].total_params(*alpha))
                             for name, alpha in zip(row['names'], row['alpha'])])
    return logs, dimensions


def _check_ascent(label, previous, current, trace):
    # Fixed floating-point allowance; it does not alter the convergence rule.
    allowance = 1e-8 * max(1., abs(previous))
    good = bool(np.isfinite(current) and current >= previous - allowance)
    trace.append(dict(step=label, before=float(previous), after=float(current),
                      change=float(current - previous), numerical_allowance=allowance,
                      ascent_verified=good))
    if not good:
        raise FABUnresolved('FAB objective decreased in ' + label)


def _weighted_profile(exp, backend, q, j, name, iteration, current, checkpoint):
    """Same declared starts/budgets as native profiles, plus current incumbent.

    Each attempted start, including failures, is retained in exp.ops. The
    current parameter is also evaluated without optimization, so the M-step
    never loses a feasible incumbent when an optimizer returns a worse point.
    """
    from fmvmm.numerics.constrained_mstep import component_stationarity
    key = f'fab:profile:{iteration}:{j}:{name}'
    fingerprint = _fingerprint(q[:, j])
    if key in exp.ops:
        cached = exp.ops[key]
        if cached.get('membership_sha256') != fingerprint:
            raise FABUnresolved('Cached FAB profile has different membership weights.')
        if not cached.get('qualified'):
            raise FABUnresolved('Previously unresolved FAB profile: ' + key)
        return key, cached
    w = q[:, j]
    start_time = time.perf_counter()
    operation = dict(name=name, membership_sha256=fingerprint, starts=[],
                     qualified=False, status='running', seconds=0.)
    exp.ops[key] = operation
    exp.active_operation = key
    if checkpoint is not None:
        checkpoint()
    candidates = []
    expected_errors = (Exception, backend.DeadlineExceeded)
    try:
        mu, covariance = backend.moments(exp.x, w)
        covariance += exp.policy.covariance_floor * np.eye(exp.x.shape[1])
        standardized = (exp.x - mu) / np.sqrt(np.diag(covariance))
        direction = (w[:, None] * standardized ** 3).sum(0) / w.sum()
        direction /= max(np.linalg.norm(direction), 1e-12)
        if np.linalg.norm(direction) < .5:
            direction = np.eye(exp.x.shape[1])[0]
        if name == 'mvt':
            starts = [backend.initial_parameters(name, mu, covariance, tail=df) for df in (4., 20.)]
        else:
            starts = [backend.initial_parameters(name, mu, covariance)]
            if name in {'mvsn', 'mvst', 'msnc', 'mssl'}:
                starts += [backend.initial_parameters(name, mu, covariance, sign * 2 * direction)
                           for sign in (-1., 1.)]
        incumbent = current['alpha'][j] if current['names'][j] == name else None
        if incumbent is not None:
            starts.insert(0, copy.deepcopy(incumbent))
        with backend.Deadline(exp.settings['profile_deadline_seconds']):
            if incumbent is not None:
                inc_start = time.perf_counter()
                inc_ll = float(np.dot(w[w > 0], backend.dist_map[name].logpdf(exp.x[w > 0], *incumbent)))
                inc_diag = component_stationarity(name, exp.x, w, incumbent, exp.policy)
                rec = dict(kind='incumbent_evaluation', weighted_loglik=inc_ll,
                           diagnostics=inc_diag, seconds=time.perf_counter() - inc_start,
                           status='complete')
                operation['starts'].append(rec)
                if np.isfinite(inc_ll):
                    candidates.append((inc_ll, copy.deepcopy(incumbent), inc_diag, len(operation['starts']) - 1))
            for initial in starts:
                exp.check_stop()
                rec = dict(kind='optimizer_start', status='running')
                operation['starts'].append(rec)
                started = time.perf_counter()
                try:
                    initial, preparation = backend.prepare_component(name, initial, exp.policy)
                    alpha, diag = backend.fit_component(name, exp.x, w, initial, exp.policy,
                                                        maxiter=exp.settings['profile_iterations'])
                    ll = float(np.dot(w[w > 0], backend.dist_map[name].logpdf(exp.x[w > 0], *alpha)))
                    rec.update(status='complete', weighted_loglik=ll, diagnostics=diag,
                               initialization=preparation)
                    if np.isfinite(ll):
                        candidates.append((ll, alpha, diag, len(operation['starts']) - 1))
                except expected_errors as error:
                    recoverable = numerical_error(error, backend)
                    rec.update(status='failed' if recoverable else 'infrastructure_error',
                               error=str(error), error_type=type(error).__name__)
                    if not recoverable:
                        raise
                    if isinstance(error, backend.DeadlineExceeded):
                        raise
                finally:
                    rec['seconds'] = time.perf_counter() - started
        if not candidates:
            raise FABUnresolved('No finite weighted fit: ' + key)
        # Same objective selects the profile. An unresolved highest value cannot
        # be replaced by an inferior qualifying fit to hide a numerical problem.
        best_ll, alpha, diag, selected = max(candidates, key=lambda item: item[0])
        dimension = int(backend.dist_map[name].total_params(*alpha))
        operation.update(alpha=alpha, weighted_loglik=best_ll, dimension=dimension,
                         diagnostics=diag, selected_start=selected,
                         qualified=bool(diag.get('gradient_acceptable')),
                         status='qualified' if diag.get('gradient_acceptable') else 'unresolved')
        if not operation['qualified']:
            raise FABUnresolved('Weighted component stationary check failed: ' + key)
        return key, operation
    except expected_errors as error:
        recoverable = numerical_error(error, backend)
        operation.update(status='unresolved' if recoverable else 'infrastructure_error',
                         qualified=False, error=str(error),
                         error_type=type(error).__name__)
        if not recoverable:
            raise
        raise FABUnresolved(str(error)) from error
    except BaseException as error:
        operation.update(status='interrupted', qualified=False, error=str(error),
                         error_type=type(error).__name__)
        raise
    finally:
        operation['seconds'] = time.perf_counter() - start_time
        exp.active_operation = None
        if checkpoint is not None:
            checkpoint()


def run_fab(exp, backend, *, evaluate, initial_scores=None, max_rounds=30,
            objective_tolerance=1e-6, checkpoint=None):
    """Return one constrained FAB trajectory; selection is exclusively by G.

    ``evaluate(row)`` computes ordinary likelihood/BIC/prediction diagnostics;
    these never choose the FAB iterate. ``checkpoint(result)`` receives the
    last complete state and the complete new weighted-operation ledger.
    ``initial_scores`` may verify a complete cached initial score matrix, but
    its BIC penalty is never used for FAB family selection.
    """
    from fmvmm.numerics.em_policy import bounded_weights
    if max_rounds < 1 or not math.isfinite(objective_tolerance) or objective_tolerance <= 0:
        raise ValueError('Positive round budget and objective tolerance required.')
    if tuple(exp.library) != tuple(backend.FAMILIES) or len(exp.library) != 13:
        raise ValueError('FAB paper comparison requires the complete 13-family library.')
    started = time.perf_counter()
    initial = copy.deepcopy(exp.student())
    initial_hash = hashlib.sha256(backend.dumps(initial, sort_keys=True).encode()).hexdigest()
    current = copy.deepcopy(initial)
    _, q = backend.mixture(exp.x, current)
    q = _memberships(q)
    initial_q_hash = _fingerprint(q)
    logs, dimensions = _state(exp, backend, current)
    objective = lower_bound(q, logs, current['pi'], dimensions)
    best = dict(value=objective, row=copy.deepcopy(current), q=q.copy(),
                logs=logs.copy(), dimensions=dimensions.copy(), round=-1)
    n, k = q.shape
    result = dict(method='constrained_fixed_k_fab', status='running', qualified=False,
                  algorithm='Fujimaki and Morinaga (2012), fixed-K FABtwo inner iteration',
                  objective_definition='G=Q+H(q)-(K-1)/2 log(n)-sum_j d_j/2 log(sum_i q_ij)',
                  selection_criterion='maximum attained FAB G; not BIC',
                  component_deletion=False, maximum_rounds=int(max_rounds),
                  objective_tolerance_per_observation=float(objective_tolerance),
                  initial_mvt_sha256=initial_hash, initial_membership_sha256=initial_q_hash,
                  objective_trace=[dict(round=-1, phase='initial_mvt', value=objective)],
                  ascent_checks=[], profile_failures=[], operation_keys=[],
                  initial_profile_keys=[], assumptions_note='Shared constrained parameter domain; no extension of unrestricted FAB asymptotic guarantees is claimed.')

    def save():
        row = copy.deepcopy(best['row'])
        ll = float(logsumexp(best['logs'] + np.log(row['pi'])[None, :], axis=1).sum())
        df = int(k - 1 + best['dimensions'].sum())
        row.update(loglik=ll, bic=float(-2 * ll + df * np.log(n)),
                   structure=tuple(sorted(row['names'])), parameter_count=df)
        # Never expose ordinary EM's diagnostic from the initial MVT as FAB's
        # convergence status, including in interrupted progress checkpoints.
        row.update(qualified=bool(result['qualified']), diagnostics=dict(
            algorithm='constrained_fixed_k_fab', algorithm_converged=result['status'] == 'converged'))
        operations = {key: value for key, value in exp.ops.items() if key.startswith('fab:profile:')}
        result.update(row=row, objective=float(best['value']), selected_round=best['round'],
                      variational_memberships=best['q'], active_operation=exp.active_operation,
                      operations=operations, incremental_fit_attempts=0,
                      incremental_profile_attempts=len(operations),
                      incremental_operation_seconds=float(sum(value.get('seconds', 0.) for value in operations.values())),
                      failed_profile_attempts=sum(value.get('status') in ('unresolved', 'interrupted') for value in operations.values()),
                      elapsed_seconds=time.perf_counter() - started)
        result['incremental_accounted_seconds'] = max(result['incremental_operation_seconds'],
            result['elapsed_seconds'] - result.get('quality_evaluation_seconds', 0.))
        result['incremental_overhead_seconds'] = result['incremental_accounted_seconds'] - result['incremental_operation_seconds']
        if checkpoint is not None:
            checkpoint(result)

    save()
    iteration = -1
    try:
        for iteration in range(max_rounds + 1):
            exp.check_stop()
            previous = objective
            if iteration:
                new_q = v_step(q, logs, current['pi'], dimensions)
                tangent = lower_bound(new_q, logs, current['pi'], dimensions, tangent_q=q)
                _check_ascent('variational_update', objective, tangent, result['ascent_checks'])
                reset = lower_bound(new_q, logs, current['pi'], dimensions)
                _check_ascent('tangent_update', tangent, reset, result['ascent_checks'])
                q, objective = new_q, reset
            profiles = []
            for j in range(k):
                component = []
                for h, name in enumerate(exp.library):
                    exp.check_stop()
                    key = f'profile:student:{j}:{name}'
                    try:
                        if iteration == 0 and key in exp.ops:
                            profile = exp.ops[key]
                            if not profile['diagnostics'].get('gradient_acceptable'):
                                raise FABUnresolved('Unresolved initial profile: ' + key)
                            checked_ll = _verify_cached_profile(
                                exp.x, q[:, j], name, profile, backend,
                                None if initial_scores is None else initial_scores[j][h])
                            result['initial_profile_keys'].append(key)
                            # Reused profile starts need not contain the exact
                            # selected MVT parameters. Retain that feasible
                            # incumbent if the cached weighted fit is inferior.
                            if current['names'][j] == name:
                                positive = q[:, j] > 0
                                incumbent_ll = float(np.dot(q[positive, j], logs[positive, j]))
                                if not math.isfinite(incumbent_ll):
                                    raise FABUnresolved('Nonfinite current-family incumbent likelihood: ' + key)
                                if checked_ll < incumbent_ll:
                                    key, profile = _weighted_profile(exp, backend, q, j, name, iteration,
                                                                     current, save)
                                    result['operation_keys'].append(key)
                        else:
                            key, profile = _weighted_profile(exp, backend, q, j, name, iteration, current, save)
                            result['operation_keys'].append(key)
                        component.append(profile)
                    except (Exception, backend.DeadlineExceeded) as error:
                        if not numerical_error(error, backend):
                            raise
                        result['profile_failures'].append(dict(round=iteration, component=j,
                                                               family=name, error=str(error)))
                        component.append(None)
                profiles.append(component)
            if any(profile is None for component in profiles for profile in component):
                raise FABUnresolved('Full 13-family M-step unresolved; no family was silently discarded.')
            ll = np.asarray([[p['weighted_loglik'] for p in component] for component in profiles])
            dims = np.asarray([[p['dimension'] for p in component] for component in profiles])
            scores = family_objectives(ll, dims, q.sum(0))
            selected = np.argmax(scores, axis=1)
            new_row = dict(names=tuple(exp.library[h] for h in selected),
                           pi=bounded_weights(q.sum(0), exp.policy.minimum_weight),
                           alpha=[copy.deepcopy(profiles[j][h]['alpha']) for j, h in enumerate(selected)])
            new_logs, new_dimensions = _state(exp, backend, new_row)
            new_objective = lower_bound(q, new_logs, new_row['pi'], new_dimensions)
            _check_ascent('family_parameter_weight_update', objective, new_objective, result['ascent_checks'])
            current, logs, dimensions, objective = new_row, new_logs, new_dimensions, new_objective
            if objective >= best['value']:
                best = dict(value=objective, row=copy.deepcopy(current), q=q.copy(),
                            logs=logs.copy(), dimensions=dimensions.copy(), round=iteration)
            change = float(objective - previous)
            result['objective_trace'].append(dict(round=iteration, phase='completed_m_step', value=objective,
                                                  change_per_observation=change / n, names=current['names'],
                                                  counts=q.sum(0), pi=current['pi']))
            save()
            if iteration > 0 and abs(change) / n <= objective_tolerance:
                result.update(status='converged', qualified=True)
                break
        else:
            result.update(status='iteration_cap', qualified=False)
    except (Exception, backend.DeadlineExceeded) as error:
        if not numerical_error(error, backend):
            result.update(status='infrastructure_error', error=str(error), error_type=type(error).__name__)
            save()
            raise
        result.update(status='unresolved', error=str(error), error_type=type(error).__name__)
    except BaseException as error:
        result.update(status='interrupted', error=str(error), error_type=type(error).__name__)
        save()
        raise
    # Select only complete states by their own FAB objective. Tiny decreases
    # admitted by the fixed rounding allowance never displace a better state.
    current, logs, dimensions = best['row'], best['logs'], best['dimensions']
    loglik = float(logsumexp(logs + np.log(current['pi'])[None, :], axis=1).sum())
    parameter_count = int(k - 1 + dimensions.sum())
    current.update(structure=tuple(sorted(current['names'])), loglik=loglik,
                   bic=float(-2 * loglik + parameter_count * np.log(n)),
                   parameter_count=parameter_count, qualified=result['qualified'],
                   diagnostics=dict(algorithm='constrained_fixed_k_fab',
                                    algorithm_converged=result['status'] == 'converged',
                                    all_family_profiles_resolved=not result['profile_failures'],
                                    objective_ascent_verified=all(c['ascent_verified'] for c in result['ascent_checks']),
                                    iteration_cap=result['status'] == 'iteration_cap',
                                    standard_em_stationarity_not_required=True),
                   interface='Frozen FMVMM logpdf and constrained fit_component; external FAB V/M loop')
    quality_started = time.perf_counter()
    quality = evaluate(current)
    result['quality_evaluation_seconds'] = time.perf_counter() - quality_started
    result.update(objective=float(best['value']), selected_round=best['round'],
                  variational_memberships=best['q'],
                  completed_rounds=max(0, len(result['objective_trace']) - 2),
                  selected=current, row=current, quality=quality, evaluation=quality,
                  convergence_status=result['status'], stop_reason=result['status'],
                  attained_row_available=True, attained_bic=current['bic'],
                  elapsed_seconds=time.perf_counter() - started)
    result['operation_keys'] = sorted(set(result['operation_keys']) | {key for key in exp.ops if key.startswith('fab:profile:')})
    result['weighted_fit_seconds'] = float(sum(exp.ops[key].get('seconds', 0.) for key in result['operation_keys']))
    result['initial_mvt_preserved'] = hashlib.sha256(backend.dumps(exp.pilots['student'], sort_keys=True).encode()).hexdigest() == initial_hash
    if not result['initial_mvt_preserved']:
        raise AssertionError('FAB mutated the selected initial MVT mixture.')
    save()
    result['row'] = current
    return result
