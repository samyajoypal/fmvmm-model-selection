"""Gate D: every retained ranked assignment from a frozen qualified L13 table.

Shared-accounting numerical qualification, not a direct timing or exhaustive-
agreement experiment. The existing readiness runner and FMVMM source are read
only. Case checkpoints retain each attempted fit; failures are never retried or
silently removed on resume. Only assignments never attempted are resumed.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

# Choose an explicitly frozen package before importing the unchanged runner.
if __name__=='__main__':
    early=argparse.ArgumentParser(add_help=False)
    early.add_argument('--package')
    selected,_=early.parse_known_args()
    if selected.package:os.environ['FMVMM_REFERENCE_ROOT']=str(Path(selected.package).resolve())
import l13_readiness_v5 as readiness
from pilot_eligibility import add_gate_d_accounting, validate_ledger, assess
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_COMMIT='c40dc36'
SCHEMA='l13-selected-fits-v5-PROPOSAL'
SCOPE=('Shared-accounting numerical qualification of every retained ranked assignment. '
       'No direct speedup, exhaustive-agreement, or global-optimality claim.')
TERMINAL_ATTEMPTS={'qualified','unqualified','failed','deadline','interrupted_process'}


class IdentityMismatch(RuntimeError):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write(path,value):
    """Exclusive case writers still use a unique atomic temporary checkpoint."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
    try:
        with temporary.open('w') as stream:
            stream.write(readiness.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary.exists():temporary.unlink()


def frozen_package(expected_commit,reference_repository=None):
    """Require exact equality of all package Python files to a named git commit.

    The reference repository can differ from --package, allowing a read-only
    source snapshot to be checked against its original repository.
    """
    repo=Path(reference_repository or readiness.PACKAGE).resolve()
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse',expected_commit+'^{commit}'],text=True).strip()
    archive=subprocess.check_output(['git','-C',str(repo),'archive',commit,'fmvmm'])
    expected={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as source:
        for member in source:
            if member.isfile() and member.name.endswith('.py'):
                expected[member.name]=hashlib.sha256(source.extractfile(member).read()).hexdigest()
    actual={str(path.relative_to(readiness.PACKAGE)):sha256(path)
            for path in sorted((readiness.PACKAGE/'fmvmm').rglob('*.py'))}
    if not expected or actual!=expected:
        changed=sorted(key for key in set(actual)|set(expected) if actual.get(key)!=expected.get(key))
        raise IdentityMismatch('Package Python sources differ from '+commit+': '+', '.join(changed))
    return {'commit':commit,'reference_repository':str(repo),'package':str(readiness.PACKAGE),
            'python_manifest_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()}


def validate_checkpoint(checkpoint,task,settings,source,runtime):
    phase='tables' if task.get('phase')=='development' else 'reserved_tables'
    expected={'task':task,'settings':settings,'phase':phase,'source_hashes':source,'runtime':runtime}
    if phase=='reserved_tables':
        release=checkpoint.get('run_identity',{}).get('validation_release',{})
        expected['validation_release']=readiness.release_identity(release.get('path'),phase,task)
        if expected['validation_release']!=release:raise IdentityMismatch('Validation release changed.')
    if checkpoint.get('run_identity')!=expected or checkpoint.get('task')!=task:
        raise IdentityMismatch('Table task/settings/source/runtime identity does not match the requested frozen run.')
    result=checkpoint.get('result',{})
    if checkpoint.get('status')!='complete' or result.get('qualified') is not True:
        raise IdentityMismatch('Only complete qualified table checkpoints can start selected-mixture fits.')
    if result.get('failures') or not isinstance(result.get('ranked_catalogue'),list):
        raise IdentityMismatch('Qualified table has failures or no complete ranked catalogue.')
    if task.get('phase') not in ('development','validation') or task.get('library_id')!='L13':
        raise IdentityMismatch('This runner only consumes registered L13 tables.')


def reconcile_row(exp,key,row):
    """Re-evaluate cached native quantities; this never refits a model."""
    if key.startswith('pilot:'):
        if row.get('qualified') is not True:raise IdentityMismatch('Unqualified cached pilot: '+key)
        diagnostic=row.get('diagnostics',{})
        if not (diagnostic.get('converged') is True and diagnostic.get('final_stationarity_acceptable') is True and diagnostic.get('hit_cap') is False):
            raise IdentityMismatch('Cached pilot qualification fields disagree: '+key)
        values,_=readiness.mixture(exp.x,row)
        if not np.isfinite(values).all() or abs(float(values.sum())-row['loglik'])>1e-6:
            raise IdentityMismatch('Cached pilot native likelihood disagrees: '+key)
        return
    label,j,name=key.split(':')[1:];j=int(j)
    if not row.get('diagnostics',{}).get('gradient_acceptable'):
        raise IdentityMismatch('Unresolved cached profile: '+key)
    if row.get('name')!=name:raise IdentityMismatch('Profile family mismatch: '+key)
    _,q=readiness.mixture(exp.x,exp.pilots[label]);w=q[:,j];positive=w>0
    ld=np.asarray(readiness.dist_map[name].logpdf(exp.x[positive],*row['alpha']),float)
    likelihood=float(w[positive]@ld)
    dimension=int(readiness.dist_map[name].total_params(*row['alpha']))
    score=2*likelihood-dimension*np.log(len(exp.x))
    if (not np.isfinite(ld).all() or abs(likelihood-row['weighted_loglik'])>1e-6
            or dimension!=row['dimension'] or abs(score-row['score'])>2e-6):
        raise IdentityMismatch('Cached profile native likelihood/score disagrees: '+key)


def restore_experiment(checkpoint,stop_file):
    task=checkpoint['task'];settings=checkpoint['run_identity']['settings']
    data=readiness.generate_l13(task)
    training_hash=hashlib.sha256(data['Xtrain'].tobytes()).hexdigest()
    if training_hash!=checkpoint.get('training_hash'):
        raise IdentityMismatch('Regenerated training data do not match the frozen table hash.')
    exp=readiness.ReadinessExperiment(task,settings,data,stop_file)
    exp.ops=copy.deepcopy(checkpoint['operations'])
    exp.pilot_construction=copy.deepcopy(checkpoint.get('pilot_construction'))
    validate_ledger(exp.pilot_construction,exp.ops,exp.task,exp.settings)
    required={'pilot:gaussian','pilot:student'}|{f'profile:gaussian:{j}:mvt' for j in range(exp.k)}
    required|={f'profile:student:{j}:{name}' for j in range(exp.k) for name in exp.library}
    if set(exp.ops)!=required:
        raise IdentityMismatch('Cached operation set does not equal the full pilot/profile table dependencies.')
    exp.pilots={label:exp.ops['pilot:'+label] for label in ('gaussian','student')}
    if exp.pilot_construction is not None:
        for attempt in exp.pilot_construction['attempts']:
            if attempt['status']!='complete':continue
            pilots={label:attempt['operations']['pilot:'+label] for label in ('gaussian','student')}
            actual=assess(SimpleNamespace(x=exp.x,k=exp.k,pilots=pilots),readiness.mixture)
            saved=attempt['assessment']
            if (actual['eligible']!=saved['eligible'] or actual['both_pilots_qualified']!=saved['both_pilots_qualified']
                    or not np.array_equal(actual['mass'],np.asarray(saved['mass']))
                    or not np.array_equal(actual['ess'],np.asarray(saved['ess']))
                    or actual['native_student_loglik']!=saved['native_student_loglik']):
                raise IdentityMismatch('Pilot eligibility assessment differs from its native posterior.')
    if tuple(exp.pilots['gaussian']['names'])!=('mvn',)*exp.k or tuple(exp.pilots['student']['names'])!=('mvt',)*exp.k:
        raise IdentityMismatch('Cached pilot component families disagree.')
    exp.dependencies={'gaussian':{'pilot:gaussian'},
                      'student':{'pilot:gaussian','pilot:student'}|{f'profile:gaussian:{j}:mvt' for j in range(exp.k)}}
    for key in sorted(required):reconcile_row(exp,key,exp.ops[key])
    if exp.pilot_construction is not None:
        for key,row in exp.pilot_construction['additional_incumbents'].items():
            reconcile_fit_native(exp,key,row)
    scores=np.array([[exp.ops[f'profile:student:{j}:{name}']['score'] for name in exp.library] for j in range(exp.k)])
    saved=np.asarray(checkpoint['result']['scores'],float)
    if scores.shape!=saved.shape or not np.array_equal(scores,saved):
        raise IdentityMismatch('Saved score matrix differs from its profile operations.')
    expected=readiness.ranked_catalogue(scores.max(1,keepdims=True)-scores,exp.library,len(exp.x))
    saved_rows=checkpoint['result']['ranked_catalogue']
    if len(expected)!=len(saved_rows):raise IdentityMismatch('Ranked catalogue is incomplete.')
    for current,old in zip(expected,saved_rows):
        if (tuple(current['structure'])!=tuple(old['structure']) or tuple(current['assignment'])!=tuple(old['assignment'])
                or current['retained']!=old['retained'] or current['deficit']!=old['deficit']):
            raise IdentityMismatch('Ranked catalogue differs from the frozen screening rule.')
    retained=[copy.deepcopy(row) for row in saved_rows if row['retained']]
    assignments=[tuple(row['assignment']) for row in retained]
    if len(assignments)!=len(set(assignments)):raise IdentityMismatch('Duplicate retained assignments.')
    return exp,retained


def live_identity(input_path):
    return {'source_hashes':readiness.source_hashes(),'runtime':readiness.runtime(),
            'selected_runner_sha256':sha256(__file__),'input_checkpoint_sha256':sha256(input_path)}


def check_unchanged(input_path,baseline):
    if live_identity(input_path)!=baseline:
        raise IdentityMismatch('Source, runtime, selected runner, or input checkpoint changed; no further fit may start.')


def operation_key(assignment):
    return 'fit:student:'+','.join(assignment)


def fit_qualified(row):
    diagnostic=row.get('diagnostics',{})
    return bool(row.get('qualified') is True and diagnostic.get('converged') is True
                and diagnostic.get('final_stationarity_acceptable') is True and diagnostic.get('hit_cap') is False)


def reconcile_fit_native(exp,key,row):
    """Validate completed native cache quantities without running an optimizer."""
    names=tuple(row['names']);pi=np.asarray(row['pi'],float);alpha=row['alpha']
    if (len(names)!=exp.k or len(alpha)!=exp.k or pi.shape!=(exp.k,)
            or not np.isfinite(pi).all() or np.any(pi<=0) or abs(float(pi.sum())-1)>1e-12):
        raise IdentityMismatch('Invalid cached full-fit component parameters: '+key)
    values,q=readiness.mixture(exp.x,row)
    if (not np.isfinite(values).all() or not np.isfinite(q).all()
            or np.any(q<0) or np.any(q>1) or np.max(np.abs(q.sum(1)-1))>1e-12):
        raise IdentityMismatch('Invalid cached full-fit native density/posterior: '+key)
    likelihood=float(values.sum())
    dimension=exp.k-1+sum(int(readiness.dist_map[name].total_params(*a)) for name,a in zip(names,alpha))
    bic=-2*likelihood+dimension*np.log(len(exp.x))
    trace=np.asarray(row.get('trace',[]),float)
    if (not np.isfinite(row.get('loglik',np.nan)) or not np.isfinite(row.get('bic',np.nan))
            or abs(likelihood-row['loglik'])>1e-6 or abs(bic-row['bic'])>2e-6
            or trace.ndim!=1 or not len(trace) or not np.isfinite(trace).all()
            or abs(float(trace[-1])-likelihood)>1e-6):
        raise IdentityMismatch('Cached full-fit native likelihood/BIC/trace disagrees: '+key)


def reconcile_completed_attempt(exp,attempt,row):
    """A cached terminal status must agree with the actual stored full fit."""
    key=attempt['operation_key'];assignment=tuple(attempt['assignment']);d=row.get('diagnostics',{})
    if (operation_key(assignment)!=key or tuple(row.get('names',()))!=assignment
            or tuple(row.get('structure',()))!=tuple(sorted(assignment))):
        raise IdentityMismatch('Cached fit assignment/structure differs from attempt ledger: '+key)
    if any(type(d.get(field)) is not bool for field in ('converged','final_stationarity_acceptable','hit_cap')):
        raise IdentityMismatch('Cached fit has incomplete qualification diagnostics: '+key)
    diagnostic_qualification=d['converged'] and d['final_stationarity_acceptable'] and not d['hit_cap']
    if row.get('qualified') is not diagnostic_qualification or d.get('policy')!=readiness.asdict(exp.policy):
        raise IdentityMismatch('Cached fit qualification/policy disagrees with diagnostics: '+key)
    expected=fit_qualified(row)
    if (attempt['status']!=('qualified' if expected else 'unqualified')
            or attempt.get('qualified') is not expected or attempt.get('bic')!=row.get('bic')):
        raise IdentityMismatch('Cached fit qualification/BIC disagrees with attempt ledger: '+key)
    reconcile_fit_native(exp,key,row)


def summary(result,base_operations):
    attempts=result['attempts'];operations=result['fit_operations'];retained=result['retained_catalogue']
    terminal=[a for a in attempts if a['status'] in TERMINAL_ATTEMPTS]
    qualified=[a for a in terminal if a['status']=='qualified' and fit_qualified(operations.get(a['operation_key'],{}))]
    qualified_keys={a['operation_key'] for a in qualified}
    failures=[a for a in terminal if a['operation_key'] not in qualified_keys]
    required=len(retained);complete=len(terminal)==required
    result['summary']={'required_assignments':required,'attempted_assignments':len(attempts),
        'terminal_assignments':len(terminal),'qualified_assignments':len(qualified),
        'failed_or_unqualified_assignments':len(failures),'unattempted_assignments':required-len(attempts),
        'every_retained_assignment_attempted':len(attempts)==required,
        'catalogue_qualified':bool(complete and not failures and not result.get('identity_invalidated')),
        'failure_keys':[a['operation_key'] for a in failures],
        'candidate_cap':None,'global_optimum_certified':False,
        'fitting_seconds_all_attempts':sum(a.get('seconds',0.) for a in attempts),
        'shared_table_operation_seconds':sum(r['seconds'] for r in base_operations.values()),
        'accounting_scope':SCOPE,
        'attempt_timing_complete':all(a['status'] in TERMINAL_ATTEMPTS and not a.get('seconds_is_lower_bound') for a in attempts),
        'interrupted_unknown_duration_attempts':sum(bool(a.get('seconds_is_lower_bound')) for a in attempts)}
    result['summary']['shared_accounted_seconds']=result['summary']['fitting_seconds_all_attempts']+result['summary']['shared_table_operation_seconds']
    candidates={k:r for k,r in base_operations.items() if k.startswith('pilot:')}
    candidates.update(operations)
    if candidates:
        winner=min(candidates,key=lambda k:(candidates[k]['bic'],k));row=candidates[winner]
        result['summary'].update(winning_key=winner,bic=row['bic'],structure=row['structure'],winner_qualified=fit_qualified(row),
            winner_scope='Cached qualified pilots plus all completed retained fits; unqualified completed fits are not discarded.')
    result['summary']=add_gate_d_accounting(result['summary'],result.get('pilot_construction'),base_operations,operations)
    return complete


def _run_locked(input_path,output,task,settings,expected_commit,reference_repository,inspect_only=False):
    input_path=Path(input_path).resolve();output=Path(output).resolve();path=output/(task['task_id']+'.json')
    source=readiness.source_hashes();run_runtime=readiness.runtime()
    input_bytes=input_path.read_bytes()
    checkpoint=readiness.loads(input_bytes.decode())
    validate_checkpoint(checkpoint,task,settings,source,run_runtime)
    freeze=frozen_package(expected_commit,reference_repository)
    baseline=live_identity(input_path)
    # Refuse even a same-identity payload change while the checkpoint is loaded.
    if baseline['input_checkpoint_sha256']!=hashlib.sha256(input_bytes).hexdigest():
        raise IdentityMismatch('Input checkpoint changed during initial validation.')
    exp,retained=restore_experiment(checkpoint,output/'STOP')
    check_unchanged(input_path,baseline)
    identity={'schema':SCHEMA,'table_run_identity':checkpoint['run_identity'],'frozen_package':freeze,
              'live_identity':baseline,'input_checkpoint_path':str(input_path),'scope':SCOPE}
    if inspect_only:
        return {'task':task['task_id'],'status':'validated_without_fitting','retained_assignments':len(retained),
                'training_hash':checkpoint['training_hash'],'frozen_commit':freeze['commit']}
    if path.exists():
        result=readiness.loads(path.read_text())
        if result.get('run_identity')!=identity:raise IdentityMismatch('Existing output identity differs; preserve it and choose a new output directory.')
        if result.get('identity_invalidated'):raise IdentityMismatch('This output was invalidated; preserve it and choose a new output directory.')
        if result.get('retained_catalogue')!=retained:raise IdentityMismatch('Existing retained catalogue changed.')
        exp.ops.update(copy.deepcopy(result['fit_operations']))
        # A process killed inside a fit cannot be silently retried or called successful.
        for attempt in result['attempts']:
            if attempt['status']=='running':
                attempt.update(status='interrupted_process',ended_at=utc_now(),seconds=None,
                               error='Previous writer ended without a terminal fit checkpoint; no automatic retry.',
                               duration_unknown=True)
        if any(a.get('seconds') is None for a in result['attempts']):
            result['unmeasured_interrupted_attempts']=sum(a.get('seconds') is None for a in result['attempts'])
            for a in result['attempts']:
                if a.get('seconds') is None:a['seconds']=0.;a['seconds_is_lower_bound']=True
    else:
        result={'schema':SCHEMA,'run_identity':identity,'task':task,'settings':settings,
                'training_hash':checkpoint['training_hash'],'retained_catalogue':retained,
                'fit_operations':{},'attempts':[],'status':'pending','created_at':utc_now(),
                'resume_rule':'Continue only unattempted assignments; preserve all failed and interrupted attempts.',
                'base_operation_keys':sorted(exp.ops),'scope':SCOPE,
                'pilot_construction':copy.deepcopy(checkpoint.get('pilot_construction'))}
    keys=[a['operation_key'] for a in result['attempts']]
    required={operation_key(row['assignment']) for row in retained}
    if len(keys)!=len(set(keys)) or not set(keys)<=required:raise IdentityMismatch('Invalid or duplicate attempt ledger.')
    if set(result['fit_operations'])!={a['operation_key'] for a in result['attempts'] if a['status'] in ('qualified','unqualified')}:
        raise IdentityMismatch('Completed fit operations and attempt ledger disagree.')
    for attempt in result['attempts']:
        if attempt['status'] in ('qualified','unqualified'):
            reconcile_completed_attempt(exp,attempt,result['fit_operations'][attempt['operation_key']])
    base_operations=checkpoint['operations'];session_started=time.perf_counter()
    def checkpoint_now(status):
        result['status']=status;result['updated_at']=utc_now()
        result['last_session_seconds']=time.perf_counter()-session_started
        summary(result,base_operations);atomic_write(path,result)
    checkpoint_now('running')
    for catalogue_index,row in enumerate(retained):
        assignment=tuple(row['assignment']);key=operation_key(assignment)
        if key in keys:continue
        try:
            check_unchanged(input_path,baseline);exp.check_stop()
        except readiness.StopRequested as error:
            result['stop_reason']=str(error);checkpoint_now('interrupted');return {'task':task['task_id'],'status':'interrupted',**result['summary']}
        except IdentityMismatch as error:
            result['identity_invalidated']=True;result['error']=str(error);checkpoint_now('source_changed_during_run')
            return {'task':task['task_id'],'status':result['status'],'error':str(error)}
        start=time.perf_counter()
        initialization={'pi':exp.pilots['student']['pi'],
                        'alpha':[exp.ops[f'profile:student:{j}:{name}']['alpha'] for j,name in enumerate(assignment)]}
        attempt={'operation_key':key,'assignment':assignment,'catalogue_index':catalogue_index,
                 'deficit':row['deficit'],'status':'running','started_at':utc_now(),
                 'initialization_sha256':hashlib.sha256(readiness.dumps(initialization,sort_keys=True).encode()).hexdigest(),
                 'profile_dependencies':[f'profile:student:{j}:{name}' for j,name in enumerate(assignment)],
                 'fit_deadline_seconds':settings['fit_deadline_seconds']}
        result['attempts'].append(attempt);keys.append(key);checkpoint_now('running')
        print(json.dumps({'task':task['task_id'],'event':'fit_started','assignment':assignment,
                          'attempt':len(keys),'required':len(retained)}),flush=True)
        interrupted=False
        boundary_stop_file=exp.stop_file
        # The boundary was checked above. Complete this one assigned fit even
        # if STOP appears during cached-parameter assembly or fitting; then
        # honor it before the next assignment. The existing deadline remains.
        exp.stop_file=None
        try:
            returned_key,fitted=exp.fit('student',assignment)
            if returned_key!=key:raise AssertionError('Unexpected cached operation key.')
            result['fit_operations'][key]=fitted
            attempt['status']='qualified' if fit_qualified(fitted) else 'unqualified'
            attempt['qualified']=attempt['status']=='qualified'
            attempt['bic']=fitted['bic']
            if attempt['status']=='unqualified':attempt['error']='Returned full fit did not satisfy the frozen qualification rule.'
        except (Exception,readiness.DeadlineExceeded,readiness.StopRequested,KeyboardInterrupt,SystemExit) as error:
            interrupted=isinstance(error,(readiness.StopRequested,KeyboardInterrupt,SystemExit))
            attempt.update(status='interrupted_process' if interrupted else 'deadline' if isinstance(error,readiness.DeadlineExceeded) else 'failed',
                           qualified=False,error=str(error),exception_type=type(error).__name__,traceback=traceback.format_exc())
        finally:
            exp.stop_file=boundary_stop_file
        attempt.update(seconds=time.perf_counter()-start,ended_at=utc_now())
        try:check_unchanged(input_path,baseline)
        except IdentityMismatch as error:
            result['identity_invalidated']=True;result['error']=str(error);checkpoint_now('source_changed_during_run')
            return {'task':task['task_id'],'status':result['status'],'error':str(error)}
        checkpoint_now('interrupted' if interrupted else 'running')
        print(json.dumps({'task':task['task_id'],'event':'fit_finished','assignment':assignment,
                          'status':attempt['status'],'seconds':attempt['seconds'],'error':attempt.get('error')}),flush=True)
        if interrupted:return {'task':task['task_id'],'status':'interrupted',**result['summary']}
    complete=summary(result,base_operations)
    checkpoint_now('complete' if complete and result['summary']['catalogue_qualified'] else 'unqualified' if complete else 'partial')
    return {'task':task['task_id'],'status':result['status'],**result['summary']}


def run_case(input_path,output,task,settings,expected_commit=DEFAULT_COMMIT,reference_repository=None,inspect_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/(task['task_id']+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Case already has an active writer; no duplicate fit started.') from error
        return _run_locked(input_path,output,task,settings,expected_commit,reference_repository,inspect_only)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='development_results')
    parser.add_argument('--output',default='selected_development_results')
    parser.add_argument('--package',help='Frozen package checkout/snapshot; selected before imports.')
    parser.add_argument('--expected-commit',default=DEFAULT_COMMIT)
    parser.add_argument('--reference-repository',help='Git repository containing expected commit; defaults to package path.')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--phase',choices=('tables','reserved_tables'),default='tables')
    parser.add_argument('--task',action='append',help='Registered table task; repeat to select cases. Every retained assignment is always attempted.')
    parser.add_argument('--inspect-only',action='store_true',help='Validate identity, native cached quantities and ranking without fitting.')
    parser.add_argument('--reviewed-development-run',action='store_true')
    args=parser.parse_args()
    if not args.inspect_only and not args.reviewed_development_run:parser.error('Proposal only; scientific fits require reviewed opt-in.')
    if args.workers<1:parser.error('--workers must be positive')
    protocol=readiness.loads(readiness.PROTOCOL.read_text());tasks=protocol['phases'][args.phase]
    if args.task:
        missing=set(args.task)-{t['task_id'] for t in tasks}
        if missing:parser.error('Unknown registered table tasks: '+', '.join(sorted(missing)))
        tasks=[t for t in tasks if t['task_id'] in set(args.task)]
    source_dir=(HERE/args.input).resolve();output=(HERE/args.output).resolve()
    if source_dir==output:parser.error('Input checkpoint directory must remain read only; choose a separate output.')
    eligible=[];ineligible=[]
    for task in tasks:
        path=source_dir/(task['task_id']+'.json')
        if not path.exists():ineligible.append({'task':task['task_id'],'reason':'checkpoint_missing'});continue
        saved=readiness.loads(path.read_text())
        if saved.get('status')!='complete' or saved.get('result',{}).get('qualified') is not True:
            ineligible.append({'task':task['task_id'],'reason':'checkpoint_not_qualified','status':saved.get('status')});continue
        eligible.append((task,path))
    if args.task and ineligible:raise IdentityMismatch('Explicitly requested table is not eligible: '+json.dumps(ineligible))
    if not eligible:raise IdentityMismatch('No qualified table checkpoint is available; no fit started.')
    output.mkdir(parents=True,exist_ok=True)
    run_id=utc_now().replace(':','-')+'_'+str(os.getpid())
    manifest={'schema':SCHEMA,'created_at':utc_now(),'scope':SCOPE,'inspect_only':args.inspect_only,
              'selected_runner_sha256':sha256(__file__),'expected_commit':args.expected_commit,
              'package':str(readiness.PACKAGE),'workers':args.workers,'eligible':[t['task_id'] for t,_ in eligible],
              'ineligible_not_fitted':ineligible,'results':[]}
    manifest_path=output/'sessions'/(run_id+'.json');atomic_write(manifest_path,manifest)
    print(json.dumps({'event':'eligibility','eligible':manifest['eligible'],'ineligible_not_fitted':ineligible,'inspect_only':args.inspect_only}),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,path,output,task,protocol['settings'],args.expected_commit,args.reference_repository,args.inspect_only):task['task_id'] for task,path in eligible}
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as error:row={'task':futures[future],'status':'refused_or_failed_before_fit','error':str(error),'traceback':traceback.format_exc()}
            manifest['results'].append(row);manifest['updated_at']=utc_now();atomic_write(manifest_path,manifest)
            print(json.dumps(row),flush=True)
    if any(r['status'] not in ('complete','validated_without_fitting') for r in manifest['results']):raise SystemExit(1)


if __name__=='__main__':main()
