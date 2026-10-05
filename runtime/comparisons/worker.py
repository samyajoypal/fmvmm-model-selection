"""Read-only reuse of the frozen study plus isolated comparison outputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from fixed_selection import all_selections


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def environment(args):
    base = args.base_bundle.resolve()
    output = args.output.resolve()
    for protected in (base, args.base_output.resolve()):
        if output == protected or output.is_relative_to(protected):
            raise ValueError('New outputs must be outside the original run and cache')
    protocol = json.loads(args.protocol.read_text())
    hpc = load_module('comparison_base_hpc', base / 'hpc.py')
    identity = hpc.verify(base)
    if identity != protocol['base_freeze_sha256']:
        raise ValueError('Unexpected frozen FMVMM source')
    hpc.runtime(base)
    runner = hpc.runner(base)
    ctx = runner.context(base)
    return protocol, runner, ctx


def write(path, value, ctx):
    ctx['runner'].atomic_write(path, value, ctx['backend'])


def prepare_context(args):
    protocol, runner, ctx = environment(args)
    ctx = dict(ctx, runner=runner)
    return protocol, runner, ctx


def source_table(args, protocol, runner, ctx, index, smoke=False):
    record = protocol['cases'][index] if not smoke else None
    source_index = record['source_index'] if not smoke else -1
    task = ctx['manifest']['tasks'][source_index] if not smoke else ctx['manifest']['smoke_task']
    if record and task != record['task']:
        raise ValueError('Protocol task differs from frozen manifest')
    root = args.base_output / 'smoke' if smoke else args.base_output
    path = runner.table_path(root, task)
    data = path.read_bytes()
    table = ctx['backend'].loads(data.decode())
    if table['identity'] != runner.task_identity(ctx, task, source_index):
        raise ValueError('Original table identity mismatch')
    table_hash = hashlib.sha256(data).hexdigest()
    return table, table_hash, task, source_index


def stamp(args, task, source_index, table_hash):
    return dict(protocol_sha256=digest(args.protocol), task_id=task['task_id'],
                source_index=source_index, table_sha256=table_hash)


def compact(value):
    """Keep exactly the selection/evaluation fields, not optimization histories."""
    result = {k: value[k] for k in ('status', 'qualified', 'seconds', 'fit_seconds', 'quality',
                                  'error', 'exception_type') if k in value}
    if 'row' in value:
        row = value['row']
        result['row'] = {k: row[k] for k in ('bic', 'loglik', 'names', 'structure', 'qualified') if k in row}
        result['row']['diagnostics'] = {k: row.get('diagnostics', {}).get(k) for k in
            ('converged', 'final_stationarity_acceptable', 'hit_cap')}
    return result


def plan(args):
    protocol, runner, ctx = prepare_context(args)
    table, thash, task, src = source_table(args, protocol, runner, ctx, args.index)
    path = args.output / 'plans' / f'{args.index:04d}.json'
    identity = stamp(args, task, src, thash)
    with runner.lock(path.with_suffix('.lock')):
        if path.exists():
            old = ctx['backend'].loads(path.read_text())
            if old['identity'] != identity or old['status'] != 'complete':
                raise ValueError('Existing plan needs inspection; no implicit retry')
            return old
        if table['status'] != 'qualified':
            result = dict(identity=identity, source_index=src, status='table_unresolved', task=task,
                          table_status=table['status'], extra_fit_indices=[])
            write(path, result, ctx)
            return result
        competitors = protocol['cases'][args.index]['competitors']
        n = len(table['data']['Xtrain'])
        methods = all_selections(table, n, protocol['selections'],
                                 task['seed'] + protocol['random_seed_offset'], competitors)
        required = set(i for value in methods.values() for i in value['indices'])
        if competitors:
            # Beam decisions must see only queried BICs, but all records are
            # available in the cache. The beam API records every actual query.
            required.update(range(len(table['catalogue'])))
        old_indices = set(table['fit_indices'])
        reuse = sorted(required & old_indices)

        def read_fit(i):
            source = runner.fit_path(args.base_output, task, i)
            raw = source.read_bytes()
            value = ctx['backend'].loads(raw.decode())
            expected = dict(runner.task_identity(ctx, task, src), table_sha256=thash,
                            candidate_index=i, assignment=table['catalogue'][i]['assignment'])
            if value.get('identity') != expected:
                raise ValueError('Original candidate identity mismatch: ' + str(source))
            if value['status'] not in runner.TERMINAL:
                raise ValueError('Original candidate is not terminal: ' + str(source))
            return str(i), compact(value), hashlib.sha256(raw).hexdigest()

        fits, hashes = {}, {}
        begin = time.perf_counter()
        # Bounded parallel reads; each raw record is discarded immediately after
        # reducing it. No repeated read per cutoff or random-search repetition.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for key, value, sha in pool.map(read_fit, reuse):
                fits[key], hashes[key] = value, sha
        result = dict(identity=identity, source_index=src, status='complete', task=task, n=n,
                      competitors=competitors, methods=methods, catalogue=table['catalogue'],
                      table_seconds=table['seconds'],
                      incumbents={k: compact(v) for k, v in table['incumbents'].items()},
                      reused_fits=fits, source_fit_sha256=hashes,
                      extra_fit_indices=sorted(required - old_indices),
                      reused_candidate_count=len(reuse), cache_read_seconds=time.perf_counter() - begin)
        # For partial comparisons the reference is read from the original
        # finalized summary; it is never relabeled exhaustive after expansion.
        reference_path = args.protocol.parent / 'base_summary.json'
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())['tasks'][src]
            if reference['task'] != task:
                raise ValueError('Original summary task mismatch')
            result['original_reference'] = reference
        write(path, result, ctx)
        return result


def fit(args):
    protocol, runner, ctx = prepare_context(args)
    table, thash, task, src = source_table(args, protocol, runner, ctx, args.index)
    plan_data = ctx['backend'].loads((args.output / 'plans' / f'{args.index:04d}.json').read_text())
    if plan_data['identity'] != stamp(args, task, src, thash):
        raise ValueError('Plan identity mismatch')
    if args.candidate not in plan_data['extra_fit_indices']:
        raise ValueError('Fit outside the declared cutoff/comparison union')
    path = args.output / 'fits' / task['task_id'] / f'{args.candidate:05d}.json'
    identity = dict(plan_data['identity'], candidate_index=args.candidate)
    with runner.lock(path.with_suffix('.lock')):
        if path.exists():
            old = ctx['backend'].loads(path.read_text())
            if old['identity'] != identity or old['status'] not in runner.TERMINAL:
                raise ValueError('Previous candidate outcome needs inspection')
            return old
        result = dict(identity=identity, status='running', qualified=False)
        write(path, result, ctx)
        start = time.perf_counter()
        try:
            with runner.signal_handler():
                exp = runner.restore(ctx, table)
                names = tuple(table['catalogue'][args.candidate]['assignment'])
                alpha = [exp.ops[f'profile:student:{j}:{name}']['alpha'] for j, name in enumerate(names)]
                pi = exp.pilots['student']['pi']
                result['initialization_sha256'] = hashlib.sha256(ctx['backend'].dumps(
                    dict(pi=pi, alpha=alpha), sort_keys=True).encode()).hexdigest()
                begin = time.perf_counter()
                result['row'] = exp.full_fit(names, pi, alpha)
                result['fit_seconds'] = time.perf_counter() - begin
                result['qualified'] = runner.qualified(result['row'])
                result['status'] = 'qualified' if result['qualified'] else 'unqualified'
                result['quality'] = runner.evaluate(exp, result['row'], ctx['backend'])
        except BaseException as error:
            result.update(status='interrupted' if isinstance(error, (runner.Shutdown, KeyboardInterrupt, SystemExit)) else
                'deadline' if isinstance(error, ctx['backend'].DeadlineExceeded) else
                'failed' if runner.numerical_error(error, ctx['backend']) else 'infrastructure_error',
                error=str(error), traceback=traceback.format_exc())
            if result['status'] in ('interrupted', 'infrastructure_error'):
                raise
        finally:
            result['seconds'] = time.perf_counter() - start
            write(path, result, ctx)
        return result


def adaptive(args):
    protocol, runner, ctx = prepare_context(args)
    if not protocol['cases'][args.index]['competitors']:
        raise ValueError('Adaptive methods only in complete-catalogue comparisons')
    table, thash, task, src = source_table(args, protocol, runner, ctx, args.index)
    path = args.output / 'adaptive' / args.method / f'{args.index:04d}.json'
    identity = dict(stamp(args, task, src, thash), method=args.method)
    with runner.lock(path.with_suffix('.lock')):
        if path.exists():
            old = ctx['backend'].loads(path.read_text())
            if old['identity'] != identity or old['status'] != 'complete':
                raise ValueError('Previous adaptive outcome needs inspection')
            return old
        result = dict(identity=identity, task=task, status='running', table_seconds=table.get('seconds', 0))
        def checkpoint(progress):
            result['result'] = progress
            write(path, result, ctx)
        try:
            if table['status'] != 'qualified':
                result.update(status='complete', result=dict(status='table_unresolved'))
            else:
                exp = runner.restore(ctx, table)
                with runner.signal_handler():
                    if args.method == 'greedy':
                        from adaptive import run_greedy
                        result['result'] = run_greedy(exp, ctx['backend'], evaluate=runner.evaluate,
                            qualified=runner.qualified, numerical_error=runner.numerical_error,
                            checkpoint=checkpoint, **protocol['greedy'])
                    elif args.method == 'fab':
                        from fab import run_fab
                        result['result'] = run_fab(exp, ctx['backend'],
                            evaluate=lambda row: runner.evaluate(exp, row, ctx['backend']),
                            max_rounds=protocol['fab']['max_rounds'],
                            objective_tolerance=protocol['fab']['objective_tolerance'], checkpoint=checkpoint)
                    else:
                        raise ValueError('Unknown adaptive method')
                result['status'] = 'complete'
        except BaseException as error:
            result.update(status='interrupted' if isinstance(error, (runner.Shutdown, KeyboardInterrupt, SystemExit))
                          else 'infrastructure_error', error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            write(path, result, ctx)
        return result


def selection_summary(indices, fits, incumbents, table_seconds, runner):
    pool = dict(incumbents)
    missing = [i for i in indices if str(i) not in fits]
    selected = {str(i): fits[str(i)] for i in indices if str(i) in fits}
    pool.update({'candidate:' + k: v for k, v in selected.items()})
    counts = {}
    for v in selected.values():
        counts[v['status']] = counts.get(v['status'], 0) + 1
    winner = runner.best(pool)
    return dict(best=winner, requested_fits=len(indices), missing_indices=missing,
                status_counts=counts, complete=not missing and all(v['status'] in runner.TERMINAL for v in selected.values()),
                all_fits_qualified=not missing and all(v['status'] == 'qualified' for v in selected.values()),
                accounted_seconds=table_seconds + sum(v.get('seconds', 0) for v in selected.values()),
                fitting_seconds=table_seconds + sum(v.get('fit_seconds', v.get('seconds', 0)) for v in selected.values()))


def summarize(args):
    protocol, runner, ctx = prepare_context(args)
    plan_data = ctx['backend'].loads((args.output / 'plans' / f'{args.index:04d}.json').read_text())
    if plan_data['identity']['protocol_sha256'] != digest(args.protocol):
        raise ValueError('Plan/protocol mismatch')
    task = plan_data['task']
    if plan_data['status'] != 'complete':
        result = dict(task=task, index=args.index, status=plan_data['status'], methods={})
        write(args.output / 'cases' / f'{args.index:04d}.json', result, ctx)
        return result
    fits = dict(plan_data['reused_fits'])
    for i in plan_data['extra_fit_indices']:
        p = args.output / 'fits' / task['task_id'] / f'{i:05d}.json'
        if not p.exists():
            continue
        value = ctx['backend'].loads(p.read_text())
        if value['identity'] != dict(plan_data['identity'], candidate_index=i):
            raise ValueError('New candidate identity mismatch')
        fits[str(i)] = compact(value)
    methods = {}
    for key, spec in plan_data['methods'].items():
        methods[key] = dict(selection_summary(spec['indices'], fits, plan_data['incumbents'],
                            plan_data['table_seconds'], runner), specification={k:v for k,v in spec.items() if k != 'indices'})
    if plan_data['competitors']:
        from adaptive import replay_beam
        vertex_incumbents = {}
        lookup = {tuple(row['structure']): i for i,row in enumerate(plan_data['catalogue'])}
        for value in plan_data['incumbents'].values():
            row = value['row']
            candidate = lookup[tuple(row['structure'])]
            if candidate not in vertex_incumbents or row['bic'] < vertex_incumbents[candidate]['row']['bic']:
                vertex_incumbents[candidate] = dict(value, status='qualified', qualified=True)
        for width in protocol['beam']['widths']:
            for budget_option in protocol['beam']['budgets']:
                budget = len(plan_data['methods']['cutoff_2']['indices']) if isinstance(budget_option, str) else budget_option
                budget = min(budget, len(plan_data['catalogue']))
                beam = replay_beam(plan_data['catalogue'], {int(k): v for k, v in fits.items()},
                    budget=budget, width=width, qualified=runner.qualified,
                    seed=task['seed'] + protocol['beam']['random_seed_offset'],
                    vertex_incumbents=vertex_incumbents)
                methods[f'beam_w{width}_b{budget_option}'] = dict(selection_summary(beam['evaluated_indices'],
                    fits, plan_data['incumbents'], plan_data['table_seconds'], runner),
                    specification=dict(kind='beam', width=width, budget_option=budget_option), search=beam)
        for method in protocol['enabled_adaptive_methods']:
            p = args.output / 'adaptive' / method / f'{args.index:04d}.json'
            if not p.exists():
                methods[method] = dict(complete=False, status='missing')
                continue
            obj = ctx['backend'].loads(p.read_text())
            if obj['identity'] != dict(plan_data['identity'], method=method):
                raise ValueError('Adaptive identity mismatch')
            r = obj.get('result', {})
            row = r.get('row')
            # FAB eligibility follows its own convergence and stationarity
            # checks, never the unrelated ordinary-EM stopping criterion.
            eligible = r.get('winner_qualified', False) if method == 'greedy' else bool(r.get('qualified', False))
            winner = None
            if row is not None and eligible and math.isfinite(row.get('bic', math.inf)):
                winner = dict(key=method, bic=row['bic'], structure=row['structure'],
                              quality=r.get('quality', r.get('evaluation', {})), qualified=True)
            methods[method] = dict(best=winner, complete=obj['status'] == 'complete',
                status=r.get('status'), stop_reason=r.get('stop_reason', r.get('convergence_status')),
                native_objective=r.get('objective'),
                accounted_seconds=plan_data['table_seconds'] + r.get('elapsed_seconds', r.get('incremental_accounted_seconds', 0)),
                fitting_seconds=plan_data['table_seconds'] + r.get('incremental_operation_seconds', 0),
                incremental_fit_attempts=r.get('incremental_fit_attempts', 0),
                incremental_profile_attempts=r.get('incremental_profile_attempts', 0),
                failed_profile_attempts=r.get('failed_profile_attempts', 0),
                failed_fit_attempts=r.get('failed_fit_attempts', 0),
                unqualified_fit_attempts=r.get('unqualified_fit_attempts', 0),
                all_fits_qualified=eligible and not any(r.get(k, 0) for k in
                    ('failed_profile_attempts', 'failed_fit_attempts', 'unqualified_fit_attempts')),
                attained_bic=r.get('attained_bic', row.get('bic') if row else None),
                specification=dict(kind=method))
    original = plan_data.get('original_reference', {})
    reference = original.get('reference_best_qualified')
    if task['reference'] == 'exhaustive':
        reference = runner.best(dict(plan_data['incumbents'], **{'candidate:' + k:v for k,v in fits.items()}))
    winners = [m['best'] for m in methods.values() if m.get('best') is not None]
    if reference:
        winners.append(reference)
    common_best = min((w['bic'] for w in winners), default=None)
    baseline = methods['cutoff_2']['best']
    for item in methods.values():
        if not item.get('best'):
            continue
        bic = item['best']['bic']
        item['bic_gap_original_reference'] = bic - reference['bic'] if reference else None
        item['bic_gap_common_best'] = bic - common_best if common_best is not None else None
        item['bic_change_from_cutoff_2'] = bic - baseline['bic'] if baseline else None
        q = item['best'].get('quality', {})
        item['test_log_density_change_from_cutoff_2'] = q.get('test_log_density', math.nan) - baseline.get('quality', {}).get('test_log_density', math.nan) if baseline else None
    result = dict(index=args.index, task=task, identity=plan_data['identity'],
                  status='complete' if all(m.get('complete') for m in methods.values()) else 'incomplete',
                  reference_scope=task['reference'], original_reference=reference,
                  common_best_bic=common_best, methods=methods,
                  additional_requested_fits=len(plan_data['extra_fit_indices']))
    write(args.output / 'cases' / f'{args.index:04d}.json', result, ctx)
    return result


def merge(args):
    protocol, runner, ctx = prepare_context(args)
    cases, missing = [], []
    for i in range(len(protocol['cases'])):
        p = args.output / 'cases' / f'{i:04d}.json'
        if not p.exists():
            missing.append(i)
        else:
            row = ctx['backend'].loads(p.read_text())
            if row['task'] != protocol['cases'][i]['task']:
                raise ValueError('Merged task mismatch')
            cases.append(row)
    result = dict(schema=protocol['schema'], protocol_sha256=digest(args.protocol),
                  scheduled_cases=len(protocol['cases']), written_cases=len(cases),
                  missing_indices=missing, cases=cases,
                  complete=not missing and all(c['status'] == 'complete' for c in cases))
    write(args.output / 'summary.json', result, ctx)
    from report import build_report
    write(args.output / 'comparison_report.json', build_report(cases), ctx)
    if not result['complete']:
        raise RuntimeError('Incomplete comparison outputs; summary retains all missing and failed outcomes')
    return result


def smoke(args):
    protocol, runner, ctx = prepare_context(args)
    table, thash, task, src = source_table(args, protocol, runner, ctx, 0, smoke=True)
    if table['status'] != 'qualified' or len(table['families']) != 13:
        raise ValueError('Smoke requires complete13-family table')
    selections = all_selections(table, len(table['data']['Xtrain']), protocol['selections'],
                               task['seed'], True)
    if selections['cutoff_2']['indices'] != table['retained_indices']:
        raise AssertionError('Cutoff2 fails to reproduce original screening')
    exp = runner.restore(ctx, table)
    candidate = 0
    names = tuple(table['catalogue'][candidate]['assignment'])
    alpha = [exp.ops[f'profile:student:{j}:{name}']['alpha'] for j, name in enumerate(names)]
    row = exp.full_fit(names, exp.pilots['student']['pi'], alpha)
    old = ctx['backend'].loads(runner.fit_path(args.base_output / 'smoke', task, candidate).read_text())
    if not runner.qualified(row) or abs(row['bic'] - old['row']['bic']) > 1e-3:
        raise AssertionError('Native smoke refit does not reproduce original accepted BIC')
    result = dict(status='passed', protocol_sha256=digest(args.protocol), table_sha256=thash,
                  native_fit_bic=row['bic'], cutoff2_reproduces_original=True,
                  families=table['families'])
    write(args.output / 'SMOKE_PASSED.json', result, ctx)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('plan', 'fit', 'adaptive', 'summarize', 'merge', 'smoke'))
    p.add_argument('--base-bundle', type=Path, required=True)
    p.add_argument('--base-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--index', type=int)
    p.add_argument('--candidate', type=int)
    p.add_argument('--method', choices=('greedy', 'fab'))
    args = p.parse_args()
    result = globals()[args.mode](args)
    print(json.dumps(dict(mode=args.mode, index=args.index, status=result.get('status', 'complete'))), flush=True)


if __name__ == '__main__':
    main()
