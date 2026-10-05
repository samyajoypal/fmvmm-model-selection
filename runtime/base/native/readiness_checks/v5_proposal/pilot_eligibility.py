"""Proposed pilot selection and accounting; no distribution or optimizer changes."""
import copy
import time
import traceback
import numpy as np

SEED_OFFSETS = (101, 211, 307)
RULE = dict(enabled=True, minimum_component_mass='strictly greater than p+1',
            seed_offsets=list(SEED_OFFSETS), require_both_pilots_qualified=True,
            selection='maximum native Student likelihood; ties choose earliest attempt')


class NoEligiblePilot(RuntimeError):
    pass


def fit_qualified(row):
    d = row.get('diagnostics', {})
    return (row.get('qualified') is True and d.get('converged') is True
            and d.get('final_stationarity_acceptable') is True and d.get('hit_cap') is False)


def checkpoint(exp):
    callback = getattr(exp, '_pilot_checkpoint', None)
    if callback is not None:
        callback()


def observed_operation(exp, key, call):
    """Record even a failed operation with no returned parameter row."""
    attempt = getattr(exp, '_pilot_capture', None)
    if attempt is None:
        return call()
    event = dict(key=key, status='running')
    attempt['events'].append(event)
    checkpoint(exp)
    started = time.perf_counter()
    try:
        result = call()
        event['status'] = 'complete'
        return result
    except BaseException as error:
        event.update(status='failed', error=str(error), exception_type=type(error).__name__,
                     traceback=traceback.format_exc())
        raise
    finally:
        event['seconds'] = time.perf_counter()-started
        checkpoint(exp)


def assess(exp, mixture):
    if any(name not in exp.pilots for name in ('gaussian', 'student')):
        return dict(eligible=False, reason='Missing completed pilot.')
    logpdf, q = mixture(exp.x, exp.pilots['student'])
    q = np.asarray(q, float)
    if (q.shape != (len(exp.x), exp.k) or not np.isfinite(q).all()
            or np.any(q < 0) or np.any(q > 1)
            or np.max(np.abs(q.sum(1)-1)) > 1e-12 or not np.isfinite(logpdf).all()):
        raise ValueError('Invalid native Student posterior or likelihood.')
    mass = q.sum(0)
    ess = np.divide(mass*mass, (q*q).sum(0), out=np.zeros_like(mass), where=(q*q).sum(0)>0)
    qualified = all(fit_qualified(exp.pilots[name]) for name in ('gaussian', 'student'))
    size_eligible = bool(np.all(mass > exp.x.shape[1]+1))
    return dict(eligible=qualified and size_eligible, both_pilots_qualified=qualified,
                size_eligible=size_eligible, mass=mass, ess=ess,
                ess_interpretation='Diagnostic only: ESS>=mass for membership weights.',
                threshold=exp.x.shape[1]+1, native_student_loglik=float(np.sum(logpdf)))


def attempt_cost(attempt):
    completed = sum(row['seconds'] for row in attempt['operations'].values())
    failed_without_row = sum(event.get('seconds', 0.) for event in attempt['events']
                             if event['status'] == 'failed' and event['key'] not in attempt['operations'])
    return completed+failed_without_row


def construct_student(exp, ordinary_student, trial_factory, mixture, recoverable=(Exception,)):
    """One original attempt, then the fixed three retries only when required.

    The caller's canonical cache contains the chosen attempt once. Rejected and
    failed attempts are retained outside that cache and are charged separately.
    """
    old = getattr(exp, 'pilot_construction', None)
    if old is not None:
        if old['status'] == 'selected':
            return exp.pilots['student']
        raise NoEligiblePilot('Previous unfinished/failed construction is preserved; no implicit retry.')
    if any(key.startswith('profile:student:') for key in exp.ops):
        raise RuntimeError('Cannot change a pilot after Student-table profiles exist.')
    ledger = dict(rule=copy.deepcopy(RULE), status='running', attempts=[],
                  selected_attempt=None, additional_incumbents={})
    exp.pilot_construction = ledger
    candidates = []
    for index, offset in enumerate((0,)+SEED_OFFSETS):
        if index and ledger['attempts'][0]['assessment'].get('eligible'):
            break
        try:
            exp.check_stop()
        except BaseException:
            ledger['status'] = 'interrupted'
            checkpoint(exp)
            raise
        started = time.perf_counter()
        task = dict(exp.task, seed=(exp.task['seed']+offset)%2**32)
        trial = None
        attempt = dict(index=index, seed_offset=offset, initialization_seed=task['seed'],
                       data_seed=task['data_seed'], status='running', operations={},
                       events=[], assessment=dict(eligible=False))
        ledger['attempts'].append(attempt)
        interrupted = None
        try:
            checkpoint(exp)
            trial = exp if index == 0 else trial_factory(task)
            trial._pilot_checkpoint = getattr(exp, '_pilot_checkpoint', None)
            attempt['operations'] = trial.ops
            trial._pilot_capture = attempt
            ordinary_student() if index == 0 else trial.student()
            attempt['assessment'] = assess(trial, mixture)
            attempt['status'] = 'complete'
            if attempt['assessment']['eligible']:
                candidates.append((attempt, trial))
        except BaseException as error:
            attempt.update(status='failed' if isinstance(error, recoverable) else 'interrupted',
                           error=str(error), exception_type=type(error).__name__, traceback=traceback.format_exc())
            if not isinstance(error, recoverable):
                interrupted = error
        finally:
            if trial is not None:
                trial._pilot_capture = None
            attempt['wall_seconds'] = time.perf_counter()-started
            attempt['operation_seconds'] = attempt_cost(attempt)
            attempt['construction_and_checkpoint_overhead_seconds'] = max(0.,attempt['wall_seconds']-attempt['operation_seconds'])
            attempt['accounted_seconds'] = attempt['operation_seconds']+attempt['construction_and_checkpoint_overhead_seconds']
            checkpoint(exp)
        if interrupted is not None:
            ledger['status'] = 'interrupted'
            checkpoint(exp)
            raise interrupted
    ledger['all_attempt_operation_seconds'] = sum(a['operation_seconds'] for a in ledger['attempts'])
    ledger['all_attempt_wall_seconds'] = sum(a['wall_seconds'] for a in ledger['attempts'])
    ledger['all_attempt_accounted_seconds'] = sum(a['accounted_seconds'] for a in ledger['attempts'])
    if not candidates:
        ledger['status'] = 'no_eligible_pilot'
        checkpoint(exp)
        raise NoEligiblePilot('No eligible pilot among the original and three registered alternatives.')
    chosen, trial = max(candidates, key=lambda pair:pair[0]['assessment']['native_student_loglik'])
    ledger['selected_attempt'] = chosen['index']
    ledger['selected_initialization_seed'] = chosen['initialization_seed']
    ledger['selected_operation_seconds'] = chosen['operation_seconds']
    ledger['additional_nonselected_operation_seconds'] = sum(a['operation_seconds'] for a in ledger['attempts'] if a is not chosen)
    ledger['additional_nonselected_accounted_seconds'] = sum(a['accounted_seconds'] for a in ledger['attempts'] if a is not chosen)
    ledger['selected_construction_and_checkpoint_overhead_seconds'] = chosen['construction_and_checkpoint_overhead_seconds']
    ledger['canonical_operation_keys'] = sorted(trial.ops)
    for attempt in ledger['attempts']:
        if attempt is chosen:
            continue
        for key, row in attempt['operations'].items():
            if key.startswith('pilot:') and fit_qualified(row):
                ledger['additional_incumbents'][f"pilot_attempt:{attempt['index']}:{key}"] = copy.deepcopy(row)
    exp.ops = copy.deepcopy(trial.ops)
    exp.pilots = {label:exp.ops['pilot:'+label] for label in ('gaussian','student')}
    exp.dependencies = {'gaussian':{'pilot:gaussian'},
                        'student':{'pilot:gaussian','pilot:student'}|{f'profile:gaussian:{j}:mvt' for j in range(exp.k)}}
    ledger['status'] = 'selected'
    checkpoint(exp)
    return exp.pilots['student']


def add_gate_d_accounting(summary, ledger, canonical_operations, fitted_operations):
    """Extend a normal Gate D summary without duplicating selected pilot costs."""
    result = copy.deepcopy(summary)
    if ledger is None:
        return result
    if ledger.get('status') != 'selected':
        raise ValueError('Gate D requires a completed eligible-pilot construction.')
    selected = ledger['attempts'][ledger['selected_attempt']]
    extra = sum(a['accounted_seconds'] for a in ledger['attempts'] if a is not selected)
    if abs(extra-ledger['additional_nonselected_accounted_seconds']) > 1e-9:
        raise ValueError('Nonselected pilot-cost ledger is inconsistent.')
    result['additional_pilot_attempt_accounted_seconds'] = extra
    overhead=ledger['selected_construction_and_checkpoint_overhead_seconds']
    result['selected_pilot_construction_overhead_seconds'] = overhead
    result['shared_accounted_seconds'] = summary['shared_accounted_seconds']+extra+overhead
    result['all_pilot_attempt_wall_seconds'] = ledger['all_attempt_wall_seconds']
    result['pilot_construction_attempts'] = len(ledger['attempts'])
    pilot_attempts=0;completed_pilots=0;qualified_pilots=0
    for attempt in ledger['attempts']:
        events=[event for event in attempt['events'] if event['key'].startswith('pilot:')]
        completed={key:row for key,row in attempt['operations'].items() if key.startswith('pilot:')}
        pilot_attempts+=len(events)+len(set(completed)-{event['key'] for event in events})
        completed_pilots+=len(completed)
        qualified_pilots+=sum(fit_qualified(row) for row in completed.values())
    result.update(attempted_pilot_mixture_fits=pilot_attempts,completed_pilot_mixture_fits=completed_pilots,
                  qualified_pilot_mixture_fits=qualified_pilots,
                  total_public_mixture_fit_attempts=pilot_attempts+summary.get('attempted_assignments',0),
                  fit_count_scope='Every pilot fit attempt including retries/failures, plus retained candidate attempts. Pilot construction count is separate.')
    candidates = {key:row for key,row in canonical_operations.items() if key.startswith('pilot:')}
    candidates.update(ledger['additional_incumbents'])
    candidates.update(fitted_operations)
    if candidates:
        key = min(candidates, key=lambda k:(candidates[k]['bic'], k))
        row = candidates[key]
        result.update(winning_key=key,bic=row['bic'],structure=row['structure'],winner_qualified=fit_qualified(row),
                      winner_scope='All qualified pilot attempts, plus all completed retained fits including unqualified completed fits.')
    return result


def validate_ledger(ledger, canonical_operations, task, settings):
    """Structural/cache validation before Gate D uses the chosen pilot."""
    if settings.get('pilot_eligibility',{'enabled':False}) == {'enabled':False}:
        if ledger is not None:
            raise ValueError('Disabled pilot rule has an unexpected retry ledger.')
        return
    if (settings['pilot_eligibility'] != RULE or ledger is None
            or ledger.get('rule') != RULE or ledger.get('status') != 'selected'):
        raise ValueError('Missing or inconsistent registered pilot construction.')
    attempts = ledger['attempts']
    expected_count = 1 if attempts[0]['assessment'].get('eligible') else 4
    if len(attempts) != expected_count:
        raise ValueError('Pilot retry count differs from the registered rule.')
    eligible = []
    expected_extra = {}
    for index,attempt in enumerate(attempts):
        offset = ((0,)+SEED_OFFSETS)[index]
        if (attempt['index'] != index or attempt['seed_offset'] != offset
                or attempt['initialization_seed'] != (task['seed']+offset)%2**32
                or attempt['data_seed'] != task['data_seed']
                or attempt['status'] not in ('complete','failed')):
            raise ValueError('Invalid pilot attempt identity/status.')
        if abs(attempt_cost(attempt)-attempt['operation_seconds']) > 1e-9:
            raise ValueError('Pilot operation cost differs from its ledger.')
        expected_overhead=max(0.,attempt['wall_seconds']-attempt['operation_seconds'])
        if (abs(expected_overhead-attempt['construction_and_checkpoint_overhead_seconds'])>1e-9
                or abs(attempt['accounted_seconds']-attempt['operation_seconds']-expected_overhead)>1e-9):
            raise ValueError('Pilot residual construction cost is inconsistent.')
        a=attempt['assessment']
        if a.get('eligible'):
            if (attempt['status']!='complete' or a.get('both_pilots_qualified') is not True
                    or not np.all(np.asarray(a['mass']) > task['p']+1)):
                raise ValueError('Eligible pilot does not satisfy the registered strict mass rule.')
            eligible.append(attempt)
        if index != ledger['selected_attempt']:
            expected_extra.update({f'pilot_attempt:{index}:{key}':row for key,row in attempt['operations'].items()
                                   if key.startswith('pilot:') and fit_qualified(row)})
    chosen = max(eligible,key=lambda a:a['assessment']['native_student_loglik'])
    if chosen['index'] != ledger['selected_attempt']:
        raise ValueError('Selected pilot differs from the registered likelihood ordering.')
    # Arrays make plain dict equality unsuitable; compare the original objects
    # recursively without lossy string/rounded-number conversions.
    def same(a,b):
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return np.array_equal(a,b,equal_nan=True)
        return a==b or (isinstance(a,float) and isinstance(b,float) and np.isnan(a) and np.isnan(b))
    if not same(expected_extra,ledger['additional_incumbents']):
        raise ValueError('Additional incumbent set differs from the qualified pilot attempts.')
    if any(key not in canonical_operations or not same(row,canonical_operations[key])
           for key,row in chosen['operations'].items()):
        raise ValueError('Chosen pilot operations differ from the canonical Gate D cache.')
    extra=sum(a['operation_seconds'] for a in attempts if a is not chosen)
    if (abs(extra-ledger['additional_nonselected_operation_seconds'])>1e-9
            or abs(sum(a['operation_seconds'] for a in attempts)-ledger['all_attempt_operation_seconds'])>1e-9):
        raise ValueError('Pilot cost totals are inconsistent.')
    if (abs(sum(a['accounted_seconds'] for a in attempts if a is not chosen)-ledger['additional_nonselected_accounted_seconds'])>1e-9
            or abs(sum(a['accounted_seconds'] for a in attempts)-ledger['all_attempt_accounted_seconds'])>1e-9
            or abs(chosen['construction_and_checkpoint_overhead_seconds']-ledger['selected_construction_and_checkpoint_overhead_seconds'])>1e-9):
        raise ValueError('Complete pilot-construction costs are inconsistent.')
