#!/usr/bin/env python3
"""Refit the paper experiments using unchanged numerical code and new outputs.

The public source inventory replaces cluster-specific wheel verification. No
scientific source, search rule, numerical budget or stopping rule is replaced.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
BASE = RUNTIME / 'base'
COMPARISONS = RUNTIME / 'comparisons'
_CONFIG = None
_RUNNER = None
_WORKER = None
_CONTEXT = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def verify_sources():
    inventory = read(RUNTIME / 'SOURCES.json')
    for relative, record in inventory['files'].items():
        path = RUNTIME / relative
        if not path.is_file() or digest(path) != record['sha256']:
            raise RuntimeError('Published runtime source/data differs: ' + relative)
    return dict(inventory_sha256=digest(RUNTIME / 'SOURCES.json'),
                package_commit=inventory['package_commit'],
                original_base_freeze_sha256=inventory['original_base_freeze_sha256'],
                verified_files=len(inventory['files']))


def verify_runtime():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Refitting requires Python 3.12 and runtime/base/requirements.lock')
    versions = {}
    for line in (BASE / 'requirements.lock').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'Expected {name}=={expected}; found {actual}')
        versions[name] = actual
    return dict(python=sys.version, packages=versions)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize(config):
    """One immutable source inventory and one native context per worker."""
    global _CONFIG, _RUNNER, _WORKER, _CONTEXT
    _CONFIG = config
    verify_sources()
    verify_runtime()
    sys.dont_write_bytecode = True
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['FMVMM_REFERENCE_ROOT'] = str(BASE / 'native/fmvmm')
    os.environ['FMVMM_STUDY_ROOT'] = str(BASE / 'Study')
    for path in (BASE, COMPARISONS):
        sys.path.insert(0, str(path))
    _RUNNER = load_module('public_paper_runner', BASE / 'runner.py')
    _CONTEXT = _RUNNER.context(BASE)
    _WORKER = load_module('public_comparison_worker', COMPARISONS / 'worker.py')
    # The original environment verifies an inaccessible cluster wheel archive.
    # Replace only that orchestration hook, after checking the public inventory.
    _WORKER.environment = portable_environment


def portable_environment(args):
    if args.base_bundle.resolve() != BASE or args.base_output.resolve() != Path(_CONFIG['base_output']):
        raise ValueError('Comparison inputs are not this new refitting run')
    if args.output.resolve() != Path(_CONFIG['comparison_output']):
        raise ValueError('Unexpected comparison output directory')
    if args.protocol.resolve() != Path(_CONFIG['protocol']):
        raise ValueError('Unexpected refitting protocol')
    if digest(args.protocol) != _CONFIG['protocol_sha256']:
        raise ValueError('Refitting protocol changed during the run')
    protocol = read(args.protocol)
    return protocol, _RUNNER, _CONTEXT


def comparison_args(index=None, candidate=None, method=None):
    return SimpleNamespace(base_bundle=BASE, base_output=Path(_CONFIG['base_output']),
        output=Path(_CONFIG['comparison_output']), protocol=Path(_CONFIG['protocol']),
        index=index, candidate=candidate, method=method)


def execute(item):
    """Run unchanged scientific functions; return only small scheduling records."""
    stage, index, detail = item
    if stage == 'table':
        row = _RUNNER.table(BASE, Path(_CONFIG['base_output']), index)
        return dict(index=index, status=row['status'], fit_indices=row.get('fit_indices', []))
    if stage == 'base_fit':
        row = _RUNNER.fit(BASE, Path(_CONFIG['base_output']), index, detail)
    elif stage == 'plan':
        row = _WORKER.plan(comparison_args(index))
        return dict(index=index, status=row['status'], extra_fit_indices=row.get('extra_fit_indices', []))
    elif stage == 'extra_fit':
        row = _WORKER.fit(comparison_args(index, candidate=detail))
    elif stage == 'adaptive':
        row = _WORKER.adaptive(comparison_args(index, method=detail))
    elif stage == 'summarize':
        row = _WORKER.summarize(comparison_args(index))
    else:
        raise ValueError('Unknown stage: ' + stage)
    return dict(index=index, status=row['status'])


def run_stage(pool, name, items, jobs, keep=False):
    """Bound queued work so large fit arrays do not exhaust controller memory."""
    iterator = iter(items)
    pending, completed, results = {}, 0, []
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < 2 * jobs:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[pool.submit(execute, item)] = item
        if not pending:
            break
        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            item = pending.pop(future)
            try:
                answer = future.result()
            except BaseException:
                for waiting in pending:
                    waiting.cancel()
                print('Failed operation:', item, file=sys.stderr, flush=True)
                raise
            completed += 1
            if keep:
                results.append(answer)
            if completed <= 3 or completed % 100 == 0:
                print(f'{name}: {completed} operations finished', flush=True)
    print(f'{name}: complete ({completed} operations)', flush=True)
    return results


def select_cases(scope, case=None):
    protocol = read(COMPARISONS / 'protocol.json')
    cases = [record for record in protocol['cases'] if scope == 'all' or
             record['task']['kind'] == ('simulation' if scope == 'simulation' else 'real_data')]
    if case is not None:
        cases = [record for record in cases if record['task_id'] == case]
    if not cases:
        raise ValueError('No matching paper cases; use plan --scope all to list task IDs')
    return protocol, cases


def description(cases):
    return dict(cases=len(cases), simulation_cases=sum(c['task']['kind'] == 'simulation' for c in cases),
                real_data_cases=sum(c['task']['kind'] == 'real_data' for c in cases),
                alternative_method_cases=sum(bool(c['competitors']) for c in cases),
                cutoff_multipliers=[0, 1, 2, 4, 8],
                task_ids=[c['task_id'] for c in cases])


def make_protocol(original, cases, source_identity):
    protocol = dict(original)
    protocol['cases'] = cases
    protocol['reproduction'] = dict(
        source_inventory_sha256=source_identity['inventory_sha256'],
        original_protocol_sha256=digest(COMPARISONS / 'protocol.json'),
        scope='New data regeneration and new fitting results; no published fitted outcomes reused.',
        benchmark='The base_summary.json beside this protocol is generated in this new run.')
    return protocol


@contextmanager
def run_directory(output, identity, resume):
    output = output.resolve()
    for protected in ('runtime', 'scripts', 'data', 'results', 'analysis', 'docs', 'paper'):
        path = ROOT / protected
        if output == path or output.is_relative_to(path):
            raise ValueError('Refitting outputs must be separate from published inputs: ' + str(output))
    if output == ROOT:
        raise ValueError('Use a new subdirectory such as runs/simulation')
    existed = output.exists()
    if existed and not resume:
        raise FileExistsError('Output directory exists; choose a new directory or explicitly use --resume')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'RUN.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if existed:
            if read(output / 'RUN_IDENTITY.json') != identity:
                raise ValueError('Cannot resume with different sources, runtime, command or case selection')
        else:
            write(output / 'RUN_IDENTITY.json', identity)
        yield output


def refit(args):
    sources, versions = verify_sources(), verify_runtime()
    original, cases = select_cases(args.scope, args.case)
    protocol = make_protocol(original, cases, sources)
    identity = dict(command='refit', sources=sources, runtime=versions,
                    source_indices=[c['source_index'] for c in cases],
                    task_ids=[c['task_id'] for c in cases],
                    protocol=protocol, orchestrator_sha256=digest(__file__))
    with run_directory(args.output, identity, args.resume) as output:
        config_dir = output / 'configuration'
        protocol_path = config_dir / 'protocol.json'
        if not protocol_path.exists():
            write(protocol_path, protocol)
        config = dict(base_output=str(output / 'base'), comparison_output=str(output / 'comparisons'),
                      protocol=str(protocol_path), protocol_sha256=digest(protocol_path))
        initialize(config)
        source_indices = [c['source_index'] for c in cases]
        try:
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize, initargs=(config,)) as pool:
                tables = run_stage(pool, 'Score tables', (('table', i, None) for i in source_indices), args.jobs, True)
                run_stage(pool, 'Base complete fits',
                          (('base_fit', row['index'], j) for row in tables for j in row['fit_indices']), args.jobs)
                # Preserve original source indices. Unselected manifest entries
                # are explicitly missing, never filled with published outcomes.
                base_summary = _RUNNER.summarize(BASE, Path(config['base_output']))
                _RUNNER.atomic_write(config_dir / 'base_summary.json', base_summary, _CONTEXT['backend'])
                plans = run_stage(pool, 'Comparison plans',
                                  (('plan', i, None) for i in range(len(cases))), args.jobs, True)
                run_stage(pool, 'Additional complete fits',
                          (('extra_fit', row['index'], j) for row in plans for j in row['extra_fit_indices']), args.jobs)
                run_stage(pool, 'Greedy and constrained FAB',
                          (('adaptive', i, method) for i, record in enumerate(cases)
                           if record['competitors'] for method in ('greedy', 'fab')), args.jobs)
                run_stage(pool, 'Comparison summaries',
                          (('summarize', i, None) for i in range(len(cases))), args.jobs)
            result = _WORKER.merge(comparison_args())
            verify_sources()
            write(output / 'REFIT_STATUS.json', dict(status='complete', completed_cases=len(result['cases']),
                  ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  note='Complete means every requested outcome is recorded; inspect numerical failures and caps.'))
        except BaseException as error:
            write(output / 'REFIT_STATUS.json', dict(status='stopped', error_type=type(error).__name__,
                  error=str(error), ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  note='Existing terminal outcomes are preserved. No failed numerical fit is silently retried.'))
            raise
        print('Refitting results:', output / 'comparisons/comparison_report.json', flush=True)


def smoke(args):
    sources, versions = verify_sources(), verify_runtime()
    identity = dict(command='smoke', sources=sources, runtime=versions, orchestrator_sha256=digest(__file__))
    with run_directory(args.output, identity, args.resume) as output:
        initialize(dict(base_output=str(output), comparison_output=str(output / 'unused'),
                        protocol=str(output / 'unused.json'), protocol_sha256='unused'))
        table = _RUNNER.table(BASE, output, 0, smoke=True)
        expected = read(BASE / 'SMOKE_EXPECTED.json')
        np = _CONTEXT['backend'].np
        if table['status'] != 'qualified':
            raise RuntimeError('Smoke table is not qualified; inspect recorded outcome')
        if table['training_sha256'] != expected['training_sha256']:
            raise RuntimeError('Smoke data differ from the published source check')
        if not np.allclose(table['scores'], expected['scores'], atol=1e-4, rtol=0):
            raise RuntimeError('Smoke weighted scores differ from original numerical tolerance')
        fits = []
        for index in expected['fit_indices']:
            row = _RUNNER.fit(BASE, output, 0, index, smoke=True)
            if row['status'] != 'qualified' or abs(row['row']['bic'] - expected['fit_bic'][str(index)]) > 1e-3:
                raise RuntimeError('Smoke complete-fit BIC differs from original numerical tolerance')
            fits.append(dict(candidate_index=index, bic=row['row']['bic']))
        from fixed_selection import all_selections
        # Selectors consume the checkpoint representation, as in the original worker.
        table = _CONTEXT['backend'].loads(_RUNNER.table_path(output, table['task']).read_text())
        selections = all_selections(table, len(table['data']['Xtrain']),
            read(COMPARISONS / 'protocol.json')['selections'], table['task']['seed'], True)
        if selections['cutoff_2']['indices'] != table['retained_indices']:
            raise AssertionError('Cutoff 2 does not reproduce the recorded screened set')
        write(output / 'SMOKE_PASSED.json', dict(status='passed', sources=sources, fits=fits,
              training_sha256=table['training_sha256'], families=table['families']))
        print('Native numerical smoke passed:', output / 'SMOKE_PASSED.json')


def check(run_tests=False):
    print(json.dumps(dict(sources=verify_sources(), runtime=verify_runtime()), indent=2))
    initialize(dict(base_output=str(ROOT / 'runs/check/base'), comparison_output=str(ROOT / 'runs/check/comparisons'),
                    protocol=str(ROOT / 'runs/check/protocol.json'), protocol_sha256='unused'))
    adapter = load_module('public_data_adapter_check', BASE / 'real_data_adapter.py')
    for name in ('faithful', 'seeds', 'wdbc'):
        data = adapter.load_task(dict(dataset_id=name, split='full'))
        print(f"Verified {name}: {len(data['Xtrain'])} rows, {data['Xtrain'].shape[1]} modeled dimensions")
    print('All 13 native distribution modules and comparison modules imported successfully.')
    if run_tests:
        suite = unittest.defaultTestLoader.discover(str(COMPARISONS), pattern='test_*.py')
        portable_tests = load_module('portable_refit_tests', RUNTIME / 'test_portable_refit.py')
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(portable_tests))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise RuntimeError('Comparison validation tests failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'check', 'test', 'smoke', 'run'))
    parser.add_argument('--scope', choices=('simulation', 'real_data', 'all'), default='all')
    parser.add_argument('--case', help='Exact task ID from the plan; performs all methods applicable to that case')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel single-threaded worker processes')
    parser.add_argument('--output', type=Path, help='New output directory; default runs/<scope> or runs/smoke')
    parser.add_argument('--resume', action='store_true', help='Resume only unattempted work with an identical run identity')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.command == 'plan':
        _, cases = select_cases(args.scope, args.case)
        print(json.dumps(description(cases), indent=2))
    elif args.command in ('check', 'test'):
        check(args.command == 'test')
    else:
        args.output = (args.output or ROOT / 'runs' / ('smoke' if args.command == 'smoke' else args.scope)).resolve()
        if args.command == 'smoke':
            if args.case or args.scope != 'all':
                parser.error('The fixed smoke case does not use --case or --scope')
            smoke(args)
        else:
            refit(args)


if __name__ == '__main__':
    main()
