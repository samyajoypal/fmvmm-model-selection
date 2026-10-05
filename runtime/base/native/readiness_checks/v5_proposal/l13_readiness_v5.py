"""PROPOSAL ONLY: fixed pilot retries, full cost ledger and fresh table seeds.

The frozen root runner and package are imported without modification. Scientific
runs require an explicit reviewed-implementation opt-in and remain disabled by
default. Merely registering this protocol generates no data and fits no models.
"""
from pathlib import Path
import sys
PROPOSAL = Path(__file__).resolve().parent
ROOT = PROPOSAL.parent.parent
sys.path.insert(0,str(ROOT))
import l13_readiness as base
from l13_readiness import *
from pilot_eligibility import RULE, construct_student, observed_operation

PROTOCOL = PROPOSAL/'l13_readiness_protocol_v5.json'


class CapturedExperiment(base.ReadinessExperiment):
    def full_fit(self,names,pi,alpha):
        label='gaussian' if set(names)=={'mvn'} else 'student' if set(names)=={'mvt'} else None
        key='pilot:'+label if label else 'public_fit:'+','.join(names)
        return observed_operation(self,key,lambda:super(CapturedExperiment,self).full_fit(names,pi,alpha))

    def profile(self,label,j,name):
        key=f'profile:{label}:{j}:{name}'
        call=lambda:super(CapturedExperiment,self).profile(label,j,name)
        return call() if key in self.ops else observed_operation(self,key,call)


class ReadinessExperiment(CapturedExperiment):
    def student(self):
        configured=self.settings.get('pilot_eligibility',{'enabled':False})
        if configured=={'enabled':False}:
            return super().student()
        if configured!=RULE:
            raise ValueError('Pilot retry rule differs from the registered v5 proposal.')
        return construct_student(self,lambda:super(ReadinessExperiment,self).student(),
            lambda task:CapturedExperiment(task,self.settings,self.data,self.stop_file),
            base.mixture,recoverable=(Exception,DeadlineExceeded))


def make_protocol():
    prior_path=ROOT/'readiness_checks/v3_proposal/l13_readiness_protocol_v3.json'
    prior=loads(prior_path.read_text());protocol=copy.deepcopy(prior)
    protocol.update(version='full13-readiness-v5-PROPOSAL',
                    proposal_status='numerical hybrid review and source freeze pending',
                    validation_locked=True,
                    prior_protocol_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                    development_change='Reviewed two-stage smooth optimizer schedule within unchanged budgets; statistical settings, family domains, starts and tolerances unchanged from v3.')
    for original in prior['phases']['reserved_tables']:
        if original['task_id'] not in ('reserved_table_p4_k5','reserved_table_p5_k3'):continue
        task=copy.deepcopy(original)
        task.update(task_id='regression_'+original['task_id'],phase='development',
                    prior_failed_validation_task=original['task_id'])
        protocol['phases']['tables'].append(task)
    for phase,offset,prefix in [('reserved_validation',9000000,'v5_'),('reserved_tables',9900000,'v5_')]:
        for task in protocol['phases'][phase]:
            task.update(task_id=prefix+task['task_id'],seed=task['data_seed']+offset,
                        data_seed=task['data_seed']+offset)
    protocol['validation_rule']='One new48-capability/12-table batch, locked until reviewed source freeze and complete development regression. Earlier failed v3 validation remains reported. Any change after release ends independent validation; no seed substitution.'
    protocol['development_rule']='Original48 capability and12 table cases, plus both observed failed v3 validation tables as explicitly labelled development regressions. Replays are not fresh validation.'
    return protocol


def source_hashes():
    source=base.source_hashes()
    for path in (Path(__file__),PROPOSAL/'pilot_eligibility.py',PROPOSAL/'l13_selected_v5.py',PROTOCOL):
        source['v5_proposal/'+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return source


def release_identity(release_file,phase,task):
    """An external reviewed release never changes hashed protocol/source files."""
    protocol=loads(PROTOCOL.read_text())
    if phase not in protocol['phases'] or task not in protocol['phases'][phase]:
        raise ValueError('Task is not registered in this phase.')
    if phase in ('capability','tables'):
        if task.get('phase')!='development':raise ValueError('Development phase has validation task.')
        return None
    if phase not in ('reserved_validation','reserved_tables') or task.get('phase')!='validation':
        raise ValueError('Unknown registered phase.')
    if release_file is None:raise RuntimeError('Reserved validation is locked; separate reviewed release required.')
    path=Path(release_file).resolve();release=loads(path.read_text())
    if (release.get('status')!='released' or release.get('protocol_sha256')!=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            or release.get('source_hashes')!=source_hashes() or release.get('runtime')!=runtime()
            or phase not in release.get('permitted_phases',[]) or task['task_id'] not in release.get('permitted_task_ids',[])):
        raise RuntimeError('Validation release does not match this frozen source/protocol/runtime/task.')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_task(task,settings,phase,output,validation_release=None):
    if settings!=loads(PROTOCOL.read_text())['settings']:
        raise ValueError('Settings differ from the registered protocol.')
    released=release_identity(validation_release,phase,task)
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    with (folder/(task['task_id']+'.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=folder/(task['task_id']+'.json')
        if path.exists():raise FileExistsError('Preserve previous attempts; choose a new output directory.')
        source=source_hashes();identity=dict(task=task,settings=settings,phase=phase,source_hashes=source,runtime=runtime())
        if released is not None:identity['validation_release']=released
        result=dict(run_identity=identity,task=task,status='running');exp=None;start=time.perf_counter()
        def save():
            if exp is not None:
                result.update(operations=exp.ops,active_operation=exp.active_operation)
                if hasattr(exp,'pilot_construction'):result['pilot_construction']=exp.pilot_construction
            result['elapsed_seconds']=time.perf_counter()-start
            if source!=source_hashes():raise RuntimeError('Proposed source changed during run.')
            if released is not None and release_identity(validation_release,phase,task)!=released:
                raise RuntimeError('Validation release changed during run.')
            write(path,result)
        try:
            data=generate_l13(task);result.update(training_hash=hashlib.sha256(data['Xtrain'].tobytes()).hexdigest(),metadata=data['metadata'])
            exp=ReadinessExperiment(task,settings,data,folder/'STOP');exp._pilot_checkpoint=save
            save();result['result']=exp.capability() if phase in ('capability','reserved_validation') else exp.score_table()
            result['status']='complete' if result['result']['qualified'] else 'unqualified'
        except (Exception,DeadlineExceeded,StopRequested,KeyboardInterrupt,SystemExit) as error:
            result.update(status='interrupted' if isinstance(error,(StopRequested,KeyboardInterrupt,SystemExit)) else 'failed',error=str(error),traceback=traceback.format_exc())
        save()
        return dict(task=task['task_id'],status=result['status'],seconds=result['elapsed_seconds'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register-protocol',action='store_true')
    parser.add_argument('--reviewed-development-run',action='store_true');parser.add_argument('--phase',choices=('capability','tables','reserved_validation','reserved_tables'),default='tables')
    parser.add_argument('--validation-release',type=Path)
    parser.add_argument('--task');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,default=PROPOSAL/'development_results')
    args=parser.parse_args()
    if args.register_protocol:
        if PROTOCOL.exists():raise FileExistsError('Registered proposal protocol already exists.')
        write(PROTOCOL,make_protocol());return
    if not args.reviewed_development_run:parser.error('Proposal only; no scientific run without explicit review opt-in.')
    protocol=loads(PROTOCOL.read_text());tasks=protocol['phases'][args.phase]
    if args.task:tasks=[task for task in tasks if task['task_id']==args.task]
    if not tasks:raise ValueError('No matching development tasks.')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_task,task,protocol['settings'],args.phase,args.output,args.validation_release) for task in tasks]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)


if __name__=='__main__':main()
