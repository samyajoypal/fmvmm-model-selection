"""Frozen native-FMVMM paper study: independent table and candidate jobs.

Every table uses all 13 families. An unresolved profile prevents screening, and
all numerical/resource failures are retained. Candidate jobs consume exact
checkpointed arrays, never regenerated data. Timings are shared-ledger operation
costs, not a direct end-to-end speed comparison.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import uuid

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
SCHEMA = 'fmvmm-paper-study-20260917-v1'
TERMINAL = {'qualified', 'unqualified', 'deadline', 'failed', 'interrupted', 'infrastructure_error'}
_CONTEXTS = {}


class IdentityError(RuntimeError):
    pass


class Shutdown(BaseException):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_identity(bundle):
    """Hash the frozen executable sources once per process, including data hashes."""
    paths = {bundle / 'manifest.json', bundle / 'runner.py', bundle / 'real_data_adapter.py'}
    for subdir in ('native', 'Study'):
        for path in (bundle / subdir).rglob('*'):
            if path.is_file() and (path.suffix in {'.py', '.json', '.npz', '.zip', '.R'}):
                paths.add(path)
    files = {str(p.relative_to(bundle)): sha256(p) for p in sorted(paths) if p.exists()}
    return {'manifest_sha256': files['manifest.json'],
            'bundle_source_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def context(bundle):
    bundle = Path(bundle).resolve()
    if bundle in _CONTEXTS:
        return _CONTEXTS[bundle]
    if _CONTEXTS:
        raise IdentityError('One process may import only one frozen bundle.')
    identity = source_identity(bundle)
    os.environ['FMVMM_REFERENCE_ROOT'] = str(bundle / 'native' / 'fmvmm')
    os.environ['FMVMM_STUDY_ROOT'] = str(bundle / 'Study')
    for path in (bundle, bundle / 'native', bundle / 'native' / 'readiness_checks' / 'v5_proposal'):
        sys.path.insert(0, str(path))
    backend = importlib.import_module('l13_readiness_v5')
    if not Path(backend.__file__).resolve().is_relative_to(bundle / 'native'):
        raise IdentityError('Native adapter was imported from a different bundle.')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if not isinstance(manifest['tasks'], list) or not manifest['tasks']:
        raise ValueError('Manifest requires a nonempty task list.')
    policy = manifest.get('challenger_policy', {})
    if policy and policy != {'frontier_nonretained': 100, 'random_remaining': 100, 'all_homogeneous': True, 'all_retained': True, 'replacement': False}:
        raise ValueError('Manifest challenger policy differs from this frozen runner.')
    ids = [t['task_id'] for t in manifest['tasks']]
    if len(ids) != len(set(ids)) or any(Path(i).name != i for i in ids):
        raise ValueError('Task identifiers must be unique file names.')
    identity.update(schema=SCHEMA, runtime=backend.runtime())
    result = dict(bundle=bundle, backend=backend, manifest=manifest, identity=identity)
    _CONTEXTS[bundle] = result
    return result


def atomic_write(path, value, backend):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()) + '.' + uuid.uuid4().hex)
    try:
        with tmp.open('w') as stream:
            stream.write(backend.dumps(value, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def signal_handler():
    def stop(signum, _frame):
        raise Shutdown('Received signal ' + str(signum))
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def qualified(row):
    d = row.get('diagnostics', {})
    return bool(row.get('qualified') is True and d.get('converged') is True
                and d.get('final_stationarity_acceptable') is True and d.get('hit_cap') is False)


def numerical_error(error, backend):
    if isinstance(error, backend.DeadlineExceeded):
        return True
    if isinstance(error, backend.np.linalg.LinAlgError):
        return True
    if type(error).__name__ == 'NoEligiblePilot':
        return True
    return isinstance(error, RuntimeError) and str(error).startswith((
        'No fitted candidate ', 'Unresolved weighted profile ',
        'All profile starts invalid ', 'No eligible pilot among '))


def pilot_code_failure(exp):
    """The inherited retry adapter preserves broad exceptions; surface coding failures."""
    for attempt in getattr(exp, 'pilot_construction', {}).get('attempts', []):
        if attempt.get('status') != 'failed':
            continue
        kind, message = attempt.get('exception_type'), attempt.get('error', '')
        if kind in {'DeadlineExceeded', 'LinAlgError', 'NoEligiblePilot'}:
            continue
        if kind == 'RuntimeError' and message.startswith((
                'No fitted candidate ', 'Unresolved weighted profile ', 'All profile starts invalid ')):
            continue
        raise RuntimeError('Pilot infrastructure/code exception: ' + str(kind) + ': ' + message)


def select_fit_indices(catalogue, reference, seed, lowest=100, random_count=100):
    """All retained models, all homogeneous models, and predeclared challengers."""
    if reference == 'exhaustive':
        return list(range(len(catalogue)))
    if reference != 'challengers':
        raise ValueError('Unknown reference design: ' + str(reference))
    retained = {i for i, row in enumerate(catalogue) if row['retained']}
    homogeneous = {i for i, row in enumerate(catalogue) if len(set(row['structure'])) == 1}
    nonretained = sorted((i for i in range(len(catalogue)) if i not in retained),
                         key=lambda i: (catalogue[i]['deficit'], tuple(catalogue[i]['structure'])))
    top = set(nonretained[:lowest])
    included = retained | homogeneous | top
    remaining = sorted(set(range(len(catalogue))) - included)
    sampled = set(random.Random(int(seed)).sample(remaining, min(random_count, len(remaining))))
    return sorted(included | sampled)


def candidate_reasons(index, catalogue, reference, seed):
    if reference == 'exhaustive':
        return ['exhaustive', 'retained'] if catalogue[index]['retained'] else ['exhaustive']
    row = catalogue[index]
    reasons = []
    if row['retained']:
        reasons.append('retained')
    if len(set(row['structure'])) == 1:
        reasons.append('homogeneous')
    ranks = sorted((i for i, r in enumerate(catalogue) if not r['retained']),
                   key=lambda i: (catalogue[i]['deficit'], tuple(catalogue[i]['structure'])))
    if index in ranks[:100]:
        reasons.append('lowest_deficit_nonretained')
    if not reasons:
        reasons.append('seeded_random_remaining')
    return reasons


def evaluate(exp, row, backend):
    """Metrics use labels/truth only after parameters have been fitted."""
    np = backend.np
    result = {'modeled_space': exp.data.get('metadata', {}).get('preprocessing', {}).get('modeled_space', 'generated feature coordinates')}
    for split in ('train', 'test'):
        raw = exp.data['X' + split]
        if raw is None or not len(raw):
            continue
        x = backend.solve_triangular(exp.cholesky, (raw - exp.mean).T, lower=True).T
        ld, q = backend.mixture(x, row)
        if not np.isfinite(ld).all() or not np.isfinite(q).all():
            raise FloatingPointError('Nonfinite native evaluation density or posterior.')
        result[split + '_log_density'] = float(ld.mean() - np.log(np.diag(exp.cholesky)).sum())
        labels = exp.data.get('y' + split)
        if labels is not None:
            result[split + '_ari'] = float(backend.adjusted_rand_score(labels, q.argmax(1)))
        truth = exp.data.get('qtrue' if split == 'train' else 'qtrue_test')
        if truth is not None and q.shape == truth.shape:
            cost = np.array([[np.abs(q[:, j] - truth[:, h]).mean() for h in range(q.shape[1])]
                             for j in range(q.shape[1])])
            a, b = backend.linear_sum_assignment(cost)
            result[split + '_posterior_tv'] = float(.5 * cost[a, b].sum())
        jac = exp.data.get('metadata', {}).get('preprocessing', {}).get('original_space_log_abs_det_jacobian')
        if jac is not None:
            result[split + '_original_space_log_density'] = result[split + '_log_density'] + jac
    if 'truth_families' in exp.task:
        result['truth_structure_match'] = tuple(sorted(row['names'])) == tuple(sorted(exp.task['truth_families']))
    return result


def incumbents(exp, backend):
    candidates = {key: copy.deepcopy(row) for key, row in exp.ops.items()
                  if key.startswith('pilot:') and qualified(row)}
    candidates.update(copy.deepcopy(getattr(exp, 'pilot_construction', {}).get('additional_incumbents', {})))
    return {key: {'row': row, 'quality': evaluate(exp, row, backend)}
            for key, row in candidates.items() if qualified(row)}


def table_path(output, task):
    return Path(output) / 'tables' / (task['task_id'] + '.json')


def fit_path(output, task, candidate):
    return Path(output) / 'fits' / task['task_id'] / (f'{candidate:05d}.json')


def task_identity(ctx, task, index):
    return dict(ctx['identity'], task_index=index, task=task, settings=ctx['manifest']['settings'])


def cached(path, expected, backend):
    if not path.exists():
        return None
    previous = backend.loads(path.read_text())
    if previous.get('identity') != expected:
        raise IdentityError('Existing checkpoint identity differs: ' + str(path))
    if previous.get('status') == 'running':
        raise IdentityError('Previous interrupted/running checkpoint preserved; no implicit retry: ' + str(path))
    return previous


def table(bundle, output, index=0, smoke=False):
    ctx = context(bundle)
    backend, manifest = ctx['backend'], ctx['manifest']
    task = manifest['smoke_task'] if smoke else manifest['tasks'][index]
    if smoke:
        index = -1
    path = table_path(output, task)
    identity = task_identity(ctx, task, index)
    with lock(path.with_suffix('.lock')):
        old = cached(path, identity, backend)
        if old is not None:
            return old
        result = {'schema': SCHEMA, 'identity': identity, 'task': task, 'status': 'running',
                  'created_at': utc_now(), 'fit_indices': [], 'incumbents': {}, 'profile_attempts': []}
        exp = None
        started = time.perf_counter()
        def checkpoint():
            result['seconds'] = time.perf_counter() - started
            result['updated_at'] = utc_now()
            if exp is not None:
                result.update(operations=exp.ops, pilot_construction=getattr(exp, 'pilot_construction', None),
                              active_operation=exp.active_operation)
            atomic_write(path, result, backend)
        checkpoint()
        try:
            with signal_handler():
                if task['kind'] == 'simulation':
                    data = backend.generate_l13(task)
                elif task['kind'] == 'real_data':
                    adapter = importlib.import_module('real_data_adapter')
                    if not Path(adapter.__file__).resolve().is_relative_to(ctx['bundle']):
                        raise IdentityError('Real-data adapter imported from outside the frozen bundle.')
                    data = adapter.load_task(task)
                else:
                    raise ValueError('Unknown task kind: ' + str(task['kind']))
                if data['Xtrain'].ndim != 2 or data['Xtrain'].shape[1] != task['p']:
                    raise ValueError('Task dimension differs from modeled data.')
                result['data'] = data
                result['training_sha256'] = hashlib.sha256(data['Xtrain'].tobytes()).hexdigest()
                exp = backend.ReadinessExperiment(task, manifest['settings'], data, None)
                result['whitening'] = {'mean': exp.mean, 'cholesky': exp.cholesky, 'x': exp.x}
                exp._pilot_checkpoint = checkpoint
                exp.student()
                pilot_code_failure(exp)
                result['incumbents'] = incumbents(exp, backend)
                scores = backend.np.full((exp.k, len(exp.library)), backend.np.nan)
                failures = []
                for j in range(exp.k):
                    for h, name in enumerate(exp.library):
                        begin = time.perf_counter()
                        attempt = {'component': j, 'family': name, 'status': 'running'}
                        result['profile_attempts'].append(attempt)
                        checkpoint()
                        try:
                            scores[j, h] = exp.profile('student', j, name)[1]['score']
                            attempt['status'] = 'qualified'
                        except BaseException as error:
                            attempt.update(status='deadline' if isinstance(error, backend.DeadlineExceeded) else 'failed',
                                           error=str(error), exception_type=type(error).__name__, traceback=traceback.format_exc())
                            failures.append(copy.deepcopy(attempt))
                            if not numerical_error(error, backend):
                                raise
                        finally:
                            attempt['seconds'] = time.perf_counter() - begin
                            checkpoint()
                result.update(scores=scores, failures=failures, families=exp.library,
                              qualified=not failures, status='qualified' if not failures else 'table_unresolved')
                if not failures:
                    catalogue = backend.ranked_catalogue(scores.max(1, keepdims=True) - scores, exp.library, len(exp.x))
                    result['catalogue'] = catalogue
                    result['fit_indices'] = select_fit_indices(catalogue, task['reference'], task['challenge_seed'])
                    result['retained_indices'] = [i for i, row in enumerate(catalogue) if row['retained']]
                    if task['kind'] == 'simulation':
                        true_structure = tuple(sorted(task['truth_families']))
                        truth_rows = [row for row in catalogue if tuple(row['structure']) == true_structure]
                        result['truth_structure_retained'] = bool(truth_rows and truth_rows[0]['retained'])
        except BaseException as error:
            result.update(status='interrupted' if isinstance(error, (Shutdown, KeyboardInterrupt, SystemExit)) else
                          'table_unresolved' if numerical_error(error, backend) else 'infrastructure_error',
                          error=str(error), exception_type=type(error).__name__, traceback=traceback.format_exc(), qualified=False)
            if exp is not None:
                try:
                    pilot_code_failure(exp)
                    result['incumbents'] = incumbents(exp, backend)
                except Exception as nested:
                    result.update(status='infrastructure_error', infrastructure_error=str(nested), infrastructure_traceback=traceback.format_exc())
            checkpoint()
            if result['status'] in {'infrastructure_error', 'interrupted'}:
                raise
        result['ended_at'] = utc_now()
        checkpoint()
        return result


def restore(ctx, checkpoint):
    """Restore exact recorded coordinates and profiles without RNG/optimizer work."""
    backend = ctx['backend']
    exp = backend.ReadinessExperiment.__new__(backend.ReadinessExperiment)
    exp.task = checkpoint['task']
    exp.settings = ctx['manifest']['settings']
    exp.data = copy.deepcopy(checkpoint['data'])
    for key in ('Xtrain', 'Xtest', 'ytrain', 'ytest', 'qtrue', 'qtrue_test'):
        if exp.data.get(key) is not None:
            exp.data[key] = backend.np.asarray(exp.data[key])
    for key in ('Xtrain', 'Xtest'):
        exp.data[key] = backend.np.asarray(exp.data[key], dtype=float).reshape(-1, exp.task['p'])
    exp.policy = backend.EMPolicy(**exp.settings['numerical_policy'])
    exp.stop_file = None
    exp.mean = backend.np.asarray(checkpoint['whitening']['mean'], dtype=float)
    exp.cholesky = backend.np.asarray(checkpoint['whitening']['cholesky'], dtype=float)
    exp.x = backend.np.asarray(checkpoint['whitening']['x'], dtype=float)
    exp.k = exp.task['fit_k']
    exp.library = list(backend.FAMILIES)
    exp.ops = copy.deepcopy(checkpoint['operations'])
    exp.pilots = {name: exp.ops['pilot:' + name] for name in ('gaussian', 'student')}
    exp.dependencies = {}
    exp.active_operation = None
    exp.pilot_construction = copy.deepcopy(checkpoint.get('pilot_construction'))
    if hashlib.sha256(exp.data['Xtrain'].tobytes()).hexdigest() != checkpoint['training_sha256']:
        raise IdentityError('Checkpoint data hash differs.')
    scores = backend.np.asarray([[exp.ops[f'profile:student:{j}:{name}']['score'] for name in exp.library]
                                 for j in range(exp.k)])
    if not backend.np.array_equal(scores, checkpoint['scores']):
        raise IdentityError('Stored scores differ from profile operations.')
    if not all(exp.ops[f'profile:student:{j}:{name}']['diagnostics'].get('gradient_acceptable')
               for j in range(exp.k) for name in exp.library):
        raise IdentityError('An unresolved profile was marked eligible for screening.')
    return exp


def fit(bundle, output, index, candidate, smoke=False):
    ctx = context(bundle)
    backend = ctx['backend']
    task = ctx['manifest']['smoke_task'] if smoke else ctx['manifest']['tasks'][index]
    if smoke:
        index = -1
    input_path = table_path(output, task)
    checkpoint_bytes = input_path.read_bytes()
    table_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = backend.loads(checkpoint_bytes.decode())
    if checkpoint.get('identity') != task_identity(ctx, task, index):
        raise IdentityError('Table identity differs from this frozen task/runtime.')
    if checkpoint.get('status') != 'qualified' or checkpoint.get('qualified') is not True:
        raise IdentityError('An unresolved table cannot be used for screening.')
    if candidate not in checkpoint['fit_indices']:
        raise IdentityError('Candidate is not in the predeclared task reference/screen union.')
    catalogue_row = checkpoint['catalogue'][candidate]
    identity = dict(task_identity(ctx, task, index), table_sha256=table_hash, candidate_index=candidate,
                    assignment=catalogue_row['assignment'])
    path = fit_path(output, task, candidate)
    with lock(path.with_suffix('.lock')):
        old = cached(path, identity, backend)
        if old is not None:
            return old
        result = {'schema': SCHEMA, 'identity': identity, 'task_id': task['task_id'],
                  'candidate_index': candidate, 'catalogue_row': catalogue_row,
                  'reasons': candidate_reasons(candidate, checkpoint['catalogue'], task['reference'], task['challenge_seed']),
                  'status': 'running', 'qualified': False, 'created_at': utc_now()}
        started = time.perf_counter()
        atomic_write(path, result, backend)
        try:
            with signal_handler():
                exp = restore(ctx, checkpoint)
                names = tuple(catalogue_row['assignment'])
                alpha = [exp.ops[f'profile:student:{j}:{name}']['alpha'] for j, name in enumerate(names)]
                pi = exp.pilots['student']['pi']
                result['initialization_sha256'] = hashlib.sha256(backend.dumps({'pi': pi, 'alpha': alpha}, sort_keys=True).encode()).hexdigest()
                atomic_write(path, result, backend)
                begin = time.perf_counter()
                row = exp.full_fit(names, pi, alpha)
                result['fit_seconds'] = time.perf_counter() - begin
                result['row'] = row
                result['qualified'] = qualified(row)
                result['status'] = 'qualified' if result['qualified'] else 'unqualified'
                result['quality'] = evaluate(exp, row, backend)
        except BaseException as error:
            result.update(status='interrupted' if isinstance(error, (Shutdown, KeyboardInterrupt, SystemExit)) else
                          'deadline' if isinstance(error, backend.DeadlineExceeded) else
                          'failed' if numerical_error(error, backend) else 'infrastructure_error',
                          qualified=False, error=str(error), exception_type=type(error).__name__, traceback=traceback.format_exc())
            if result['status'] in {'infrastructure_error', 'interrupted'}:
                raise
        finally:
            result.update(seconds=time.perf_counter() - started, ended_at=utc_now())
            if sha256(input_path) != table_hash:
                result.update(status='infrastructure_error', qualified=False, error='Table changed during fitting.')
                atomic_write(path, result, backend)
                raise IdentityError('Input table changed during fitting.')
            atomic_write(path, result, backend)
        return result


def best(rows, only_qualified=True):
    eligible = {key: value for key, value in rows.items()
                if 'row' in value and math.isfinite(value['row'].get('bic', math.nan))
                and (not only_qualified or (qualified(value['row']) and value.get('status', 'qualified') == 'qualified'
                                             and value.get('qualified', True) is True))}
    if not eligible:
        return None
    key = min(eligible, key=lambda name: (eligible[name]['row']['bic'], name))
    value = eligible[key]
    return dict(key=key, bic=value['row']['bic'], structure=value['row']['structure'],
                qualified=qualified(value['row']), quality=value.get('quality', {}))


def summarize_case(ctx, output, index):
    task = ctx['manifest']['tasks'][index]
    backend = ctx['backend']
    path = table_path(output, task)
    result = {'task': task, 'task_index': index, 'table_status': 'missing', 'reference': task['reference']}
    if not path.exists():
        return result
    checkpoint = backend.loads(path.read_text())
    if checkpoint.get('identity') != task_identity(ctx, task, index):
        raise IdentityError('Table identity mismatch while summarizing.')
    result.update(table_status=checkpoint['status'], table_seconds=checkpoint.get('seconds', 0.),
                  truth_structure_retained=checkpoint.get('truth_structure_retained'),
                  pilot_incumbents=len(checkpoint.get('incumbents', {})),
                  pilot_construction_attempts=len((checkpoint.get('pilot_construction') or {}).get('attempts', [])))
    result['best_qualified_pilot'] = best(checkpoint.get('incumbents', {}))
    if checkpoint['status'] != 'qualified':
        result['screening_available'] = False
        return result
    table_hash = sha256(path)
    fits = {}
    missing = []
    for candidate in checkpoint['fit_indices']:
        p = fit_path(output, task, candidate)
        if not p.exists():
            missing.append(candidate)
            continue
        value = backend.loads(p.read_text())
        expected = dict(task_identity(ctx, task, index), table_sha256=table_hash, candidate_index=candidate,
                        assignment=checkpoint['catalogue'][candidate]['assignment'])
        if value.get('identity') != expected:
            raise IdentityError('Fit identity mismatch while summarizing: ' + str(p))
        fits[candidate] = value
    retained = checkpoint['retained_indices']
    screen = dict(checkpoint.get('incumbents', {}))
    reference = dict(screen)
    reference.update({'candidate:' + str(i): value for i, value in fits.items() if 'row' in value})
    screen.update({'candidate:' + str(i): fits[i] for i in retained if i in fits and 'row' in fits[i]})
    counts = {name: sum(v['status'] == name for v in fits.values()) for name in sorted(TERMINAL | {'running'})}
    screening_complete = all(i in fits and fits[i]['status'] in TERMINAL for i in retained)
    reference_complete = not missing and all(v['status'] in TERMINAL for v in fits.values())
    screening_qualified = screening_complete and all(fits[i]['status'] == 'qualified' for i in retained)
    reference_qualified = reference_complete and all(v['status'] == 'qualified' for v in fits.values())
    result.update(screening_available=True, total_canonical_structures=len(checkpoint['catalogue']),
                  retained_structures=len(retained), requested_reference_structures=len(checkpoint['fit_indices']),
                  fitted_results=len(fits), missing_indices=missing, fit_status_counts=counts,
                  screening_complete=screening_complete, reference_complete=reference_complete,
                  screening_all_fits_qualified=screening_qualified, reference_all_fits_qualified=reference_qualified,
                  screening_best_qualified=best(screen), reference_best_qualified=best(reference),
                  screening_best_attained=best(screen, False), reference_best_attained=best(reference, False),
                  screen_candidate_seconds=sum(fits[i].get('seconds', 0.) for i in retained if i in fits),
                  reference_candidate_seconds=sum(v.get('seconds', 0.) for v in fits.values()),
                  candidate_count_reduction=1 - len(retained) / len(checkpoint['catalogue']))
    result['screen_shared_accounted_seconds'] = result['table_seconds'] + result['screen_candidate_seconds']
    result['reference_shared_accounted_seconds'] = result['table_seconds'] + result['reference_candidate_seconds']
    if result['screening_best_qualified'] and result['reference_best_qualified']:
        gap = result['screening_best_qualified']['bic'] - result['reference_best_qualified']['bic']
        result.update(qualified_attained_bic_gap=gap,
                      qualified_attained_structure_agreement=result['screening_best_qualified']['structure'] == result['reference_best_qualified']['structure'],
                      qualified_attained_bic_agreement=abs(gap) <= 1e-6,
                      comparison_scope='Complete numerically qualified finite-budget canonical reference' if reference_qualified and task['reference'] == 'exhaustive' else
                      'Attained qualified fits only; failed, incomplete or challenger-subset reference is not exhaustive verification')
    return result


def summarize(bundle, output):
    ctx = context(bundle)
    cases = [summarize_case(ctx, output, i) for i in range(len(ctx['manifest']['tasks']))]
    result = {'schema': SCHEMA, 'identity': ctx['identity'], 'created_at': utc_now(), 'tasks': cases,
              'planned_tasks': len(cases), 'table_status_counts': {},
              'timing_scope': 'Shared-checkpoint operation accounting, not an independent end-to-end wall-clock speedup experiment.',
              'reference_scope': 'One score-ranked label assignment per canonical family structure, with identical frozen finite fitting budgets. No global MLE claim.'}
    for row in cases:
        status = row['table_status']
        result['table_status_counts'][status] = result['table_status_counts'].get(status, 0) + 1
    result['requested_candidate_fits'] = sum(r.get('requested_reference_structures', 0) for r in cases)
    result['written_candidate_results'] = sum(r.get('fitted_results', 0) for r in cases)
    result['candidate_status_counts'] = {status: sum(r.get('fit_status_counts', {}).get(status, 0) for r in cases)
                                         for status in sorted(TERMINAL | {'running'})}
    atomic_write(Path(output) / 'summary.json', result, ctx['backend'])
    return result


table_task = table
fit_task = fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('table', 'fit', 'summarize'))
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--candidate', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage in {'table', 'fit'} and args.index is None:
        parser.error('--index is required')
    if args.stage == 'fit' and args.candidate is None:
        parser.error('--candidate is required')
    result = table(args.bundle, args.output, args.index, args.smoke) if args.stage == 'table' else fit(args.bundle, args.output, args.index, args.candidate, args.smoke) if args.stage == 'fit' else summarize(args.bundle, args.output)
    print(json.dumps({key: result[key] for key in ('task_id', 'candidate_index', 'status', 'seconds', 'planned_tasks', 'table_status_counts', 'requested_candidate_fits', 'written_candidate_results') if key in result}), flush=True)


if __name__ == '__main__':
    main()
