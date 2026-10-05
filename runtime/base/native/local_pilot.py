"""Isolated selection pilot using the actual FMVMM public soft-EM workflow.

Development uses a shared operation ledger. A separate direct mode repeats
pipelines with fresh caches and a common final checkpoint. No HPC commands.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fmvmm-em-repair-mpl')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
PACKAGE = Path(os.environ.get('FMVMM_REFERENCE_ROOT', HERE / 'fmvmm')).resolve()
STUDY = Path(os.environ.get('FMVMM_STUDY_ROOT', HERE.parent.parent)).resolve()
os.environ['FMVMM_REFERENCE_ROOT'] = str(PACKAGE)
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(STUDY))
import argparse
import copy
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
import signal
import time
import traceback
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from fmvmm.mixtures.FMVMM import fmvmm, dist_map, all_dist
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import fit_component, prepare_component
from fmvmm.numerics.lossless_json import dumps, loads
from data_generation import generate
from fmvmm_engine import initial_parameters
if not Path(sys.modules['fmvmm.mixtures.FMVMM'].__file__).resolve().is_relative_to(PACKAGE):
    raise RuntimeError('Imported FMVMM is outside the requested package snapshot')

LIBRARIES = {'L3': ['mvn','mvt','mvsn'], 'L5': ['mvn','mvt','mvsn','mvst','msnc'], 'L13': list(all_dist)}
MODES = ('gaussian_screen', 'student_screen', 'student_ranked_screen', 'student_local', 'fast_student', 'reference_student')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fingerprint():
    paths = [HERE / 'local_pilot.py', HERE / 'pilot_protocol.json']
    result = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result.update({'fmvmm/' + str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((PACKAGE / 'fmvmm').rglob('*.py'))})
    for name in ('data_generation.py', 'fmvmm_engine.py'):
        result['Study/' + name] = hashlib.sha256((STUDY / name).read_bytes()).hexdigest()
    return result


def runtime():
    return {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'pandas')},
            'numerical_threads': {name: os.environ.get(name) for name in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}


def compact_diagnostics(diagnostic):
    """Retain final component records, all aggregate counts and the EM trace."""
    result = copy.deepcopy(diagnostic)
    history = result.get('component_iterations', [])
    result['component_history_iterations'] = len(history)
    result['component_history_retention'] = 'final iteration; aggregate counts retain all updates'
    result['component_iterations'] = history[-1:]
    return result


class DeadlineExceeded(BaseException):
    """Bypass numerical objectives' ordinary Exception-to-penalty handlers."""


class StopRequested(BaseException):
    pass


class Deadline:
    def __init__(self, seconds): self.seconds = seconds
    def __enter__(self):
        self.old = signal.signal(signal.SIGALRM, self.expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
    def expired(self, *_): raise DeadlineExceeded(f'Operation exceeded {self.seconds} seconds')
    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old)


def canonical(a): return tuple(sorted(a))


def mixture(x, row):
    a = np.column_stack([np.log(w) + dist_map[name].logpdf(x, *alpha)
                         for w,name,alpha in zip(row['pi'],row['names'],row['alpha'])])
    log_density = logsumexp(a, axis=1)
    return log_density, np.exp(a-log_density[:,None])


def moments(x, weights):
    w = weights / weights.sum()
    mean = w @ x
    centered = x - mean
    covariance = centered.T @ (w[:,None] * centered)
    return mean, (covariance + covariance.T) / 2


class Experiment:
    def __init__(self, task, settings, data, stop_file=None):
        self.task, self.settings, self.data = task, settings, data
        self.stop_file = stop_file
        self.policy = EMPolicy(**settings['numerical_policy'])
        raw = data['Xtrain']
        self.mean = raw.mean(0)
        self.cholesky = np.linalg.cholesky(np.cov(raw,rowvar=False,bias=True))
        self.x = solve_triangular(self.cholesky,(raw-self.mean).T,lower=True).T
        self.k, self.library = task['fit_k'], LIBRARIES[task['library_id']]
        self.orders = list(itertools.product(self.library,repeat=self.k))
        self.structures = sorted({canonical(a) for a in self.orders})
        self.ops, self.pilots, self.dependencies = {}, {}, {}

    def full_fit(self, names, pi, alpha):
        self.check_stop()
        started = time.perf_counter()
        names = tuple(names)
        model = fmvmm(n_clusters=self.k, list_of_dist=self.library,
            candidate_combinations=[names], candidate_initializations={names:(pi,alpha)},
            numerical_policy=asdict(self.policy), strict_density=True, em_type='soft',
            max_iter=self.settings['em_iterations'], tol=self.settings['em_tolerance'],
            verbose=False, print_log_likelihood=False, n_jobs=1)
        with Deadline(self.settings['fit_deadline_seconds']): model.fit(self.x)
        if not model.list_bic:
            raise RuntimeError(f'No fitted candidate {names}: {model.candidate_failures}')
        pi, alpha = model.best_params()
        row = {'names': names, 'structure': canonical(names), 'pi':pi, 'alpha':alpha,
               'bic':float(model.best_bic()), 'loglik':float(model.list_log_likelihood[0]),
               'trace':model.list_all_log_likelihood[0],
               'diagnostics':compact_diagnostics(model.em_diagnostics[0]),
               'interface':'fmvmm(..., em_type="soft", numerical_policy=...).fit(X)',
               'fitting_backend':'existing BaseMixture._fit outer EM with constrained weighted M-steps'}
        ld, q = mixture(self.x,row)
        row['native_loglik_error'] = abs(float(ld.sum())-row['loglik'])
        if row['native_loglik_error'] > 1e-6: raise AssertionError('Stored/public-density likelihood mismatch')
        diagnostic = row['diagnostics']
        row['qualified'] = bool(diagnostic.get('converged') and
            diagnostic.get('final_stationarity_acceptable') and not diagnostic.get('hit_cap'))
        row['seconds'] = time.perf_counter()-started
        return row

    def check_stop(self):
        if self.stop_file is not None and self.stop_file.exists():
            raise StopRequested('Local run stopped at operation boundary: ' + self.stop_file.read_text().strip())

    def gaussian(self):
        if 'gaussian' not in self.pilots:
            started = time.perf_counter()
            labels = KMeans(n_clusters=self.k,n_init=10,random_state=self.task['seed']).fit_predict(self.x)
            weights = np.eye(self.k)[labels]
            alpha = []
            for j in range(self.k):
                mu, sigma = moments(self.x, weights[:,j])
                sigma += self.policy.covariance_floor * np.eye(self.x.shape[1])
                alpha.append(prepare_component('mvn',(mu,sigma),self.policy)[0])
            row = self.full_fit(('mvn',)*self.k,weights.mean(0),alpha)
            row['seconds'] = time.perf_counter()-started
            self.ops['pilot:gaussian'] = row
            self.pilots['gaussian'] = row
            self.dependencies['gaussian'] = {'pilot:gaussian'}
        return self.pilots['gaussian']

    def profile(self,label,j,name):
        self.check_stop()
        key = f'profile:{label}:{j}:{name}'
        if key in self.ops: return key,self.ops[key]
        started = time.perf_counter()
        _, q = mixture(self.x,self.pilots[label]);w = q[:,j]
        mu, covariance = moments(self.x,w)
        # This is a declared feasible starting scale, not a density-time repair.
        covariance += self.policy.covariance_floor * np.eye(self.x.shape[1])
        standardized = (self.x-mu)/np.sqrt(np.diag(covariance))
        direction = (w[:,None]*standardized**3).sum(0)/w.sum()
        direction /= max(np.linalg.norm(direction),1e-12)
        if np.linalg.norm(direction)<.5: direction=np.eye(self.x.shape[1])[0]
        if name=='mvt':
            starts = [initial_parameters(name,mu,covariance,tail=df) for df in (4.,20.)]
        else:
            starts = [initial_parameters(name,mu,covariance)]
            if name in {'mvsn','mvst','msnc','mssl'}:
                starts += [initial_parameters(name,mu,covariance,sign*2*direction) for sign in (-1.,1.)]
        records=[];best=None;best_ll=-np.inf
        with Deadline(self.settings['profile_deadline_seconds']):
            for initial in starts:
                initial, preparation=prepare_component(name,initial,self.policy)
                alpha, diagnostic = fit_component(name,self.x,w,initial,self.policy,
                                                   maxiter=self.settings['profile_iterations'])
                ll=float(np.dot(w[w>0],dist_map[name].logpdf(self.x[w>0],*alpha)))
                records.append({'diagnostics':diagnostic,'weighted_loglik':ll,'initialization':preparation})
                if np.isfinite(ll) and ll>best_ll: best,best_ll=alpha,ll
        if best is None: raise RuntimeError(f'All profile starts invalid for {key}')
        dimension=int(dist_map[name].total_params(*best))
        selected=max(range(len(records)),key=lambda i:records[i]['weighted_loglik'])
        row={'name':name,'alpha':best,'weighted_loglik':best_ll,'score':2*best_ll-dimension*np.log(len(self.x)),
             'dimension':dimension,'diagnostics':records[selected]['diagnostics'],
             'starts':records,'seconds':time.perf_counter()-started,'global_optimum_certified':False}
        self.ops[key]=row
        if not row['diagnostics'].get('gradient_acceptable'):
            raise RuntimeError(f'Unresolved weighted profile {key}; score retained for diagnosis but not used to exclude families')
        return key,row

    def student(self):
        if 'student' not in self.pilots:
            self.gaussian();keys=set(self.dependencies['gaussian']);alpha=[]
            for j in range(self.k):
                key,row=self.profile('gaussian',j,'mvt');keys.add(key);alpha.append(row['alpha'])
            key='pilot:student'
            row=self.full_fit(('mvt',)*self.k,self.pilots['gaussian']['pi'],alpha)
            self.ops[key]=row;keys.add(key)
            self.pilots['student']=row;self.dependencies['student']=keys
        return self.pilots['student']

    def table(self,label):
        self.gaussian() if label=='gaussian' else self.student()
        keys=set(self.dependencies[label]);scores=np.empty((self.k,len(self.library)))
        for j in range(self.k):
            for h,name in enumerate(self.library):
                key,row=self.profile(label,j,name);keys.add(key);scores[j,h]=row['score']
        d=scores.max(1,keepdims=True)-scores
        deficits={a:float(sum(d[j,self.library.index(name)] for j,name in enumerate(a))) for a in self.orders}
        return keys,deficits

    def fit(self,label,assignment):
        assignment=tuple(assignment)
        if label=='gaussian' and assignment==('mvn',)*self.k:
            return 'pilot:gaussian',self.gaussian()
        key=f'fit:{label}:'+','.join(assignment)
        if key not in self.ops:
            params=[self.profile(label,j,name)[1]['alpha'] for j,name in enumerate(assignment)]
            self.ops[key]=self.full_fit(assignment,self.pilots[label]['pi'],params)
        return key,self.ops[key]

    def method(self,mode):
        if mode not in MODES: raise ValueError(mode)
        label='gaussian' if mode=='gaussian_screen' else 'student'
        keys,deficits=self.table(label)
        screened={canonical(a) for a in self.orders if deficits[a]<=2*np.log(len(self.x))+1e-10}|{('mvn',)*self.k}
        retained=set(self.structures) if mode in {'reference_student','fast_student'} else screened
        for structure in sorted(retained):
            assignments=sorted(set(itertools.permutations(structure)))
            if mode in {'fast_student','student_ranked_screen'}:
                assignments=[min(assignments,key=lambda a:(deficits[a],a))]
            for assignment in assignments: keys.add(self.fit(label,assignment)[0])
        def fits(): return sorted(k for k in keys if k.startswith(('fit:','pilot:')))
        def winner(): return min(fits(),key=lambda k:(self.ops[k]['bic'],k))
        incumbent=winner();history=[]
        if mode=='student_local':
            for _ in range(len(self.orders)+2):
                names=tuple(self.ops[incumbent]['names']);neighbours={names}
                for j in range(self.k):
                    for name in self.library:
                        a=list(names);a[j]=name;neighbours.add(tuple(a))
                before=len(fits())
                for a in sorted(neighbours): keys.add(self.fit(label,a)[0])
                challenger=winner();improves=self.ops[challenger]['bic']<self.ops[incumbent]['bic']-1e-7
                history.append({'centre':incumbent,'challenger':challenger,'improves':improves,
                                'new_full_fits':len(fits())-before})
                if not improves: break
                incumbent=challenger
            else: raise AssertionError('Fixed-catalogue search did not terminate')
        full=fits();profiles=[k for k in keys if k.startswith('profile:')];row=self.ops[incumbent]
        return {'mode':mode,'status':'complete','winning_key':incumbent,'bic':row['bic'],'structure':row['structure'],
                'winner_qualified':row['qualified'],
                'catalogue_qualified':all(self.ops[key]['qualified'] for key in full),
                'unqualified_full_fits':[key for key in full if not self.ops[key]['qualified']],
                'winner_diagnostics':row['diagnostics'],'operation_keys':sorted(keys),'history':history,
                'retained_structures':sorted(retained),'full_fits':len(full),'profiles':len(profiles),
                'distinct_models':len({canonical(self.ops[k]['names']) for k in full}),
                'accounted_seconds':sum(self.ops[k]['seconds'] for k in keys)}

    def quality(self,key):
        test=solve_triangular(self.cholesky,(self.data['Xtest']-self.mean).T,lower=True).T
        ld,q=mixture(test,self.ops[key]);target=self.data['qtrue_test']
        cost=np.array([[np.abs(q[:,j]-target[:,h]).mean() for h in range(self.k)] for j in range(self.k)])
        a,b=linear_sum_assignment(cost)
        return {'test_log_density':float(ld.mean()-np.log(np.diag(self.cholesky)).sum()),
                'posterior_tv':float(.5*cost[a,b].sum()),
                'ari':float(adjusted_rand_score(self.data['ytest'],q.argmax(1)))}


def run_case(task,settings,output,modes,direct=False,index=0):
    """One writer per task, including before the first checkpoint exists."""
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    with (folder/(task['task_id']+'.lock')).open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This task already has an active writer; no duplicate fit started') from error
        return _run_case(task,settings,output,modes,direct,index)


def _run_case(task,settings,output,modes,direct=False,index=0):
    path=Path(output)/(task['task_id']+'.json');source=fingerprint()
    run_identity={'task':task, 'settings':settings, 'methods':list(modes), 'direct':direct,
                  'order_index':index, 'source_hashes':source, 'runtime':runtime()}
    if path.exists():
        old=loads(path.read_text())
        if old.get('run_identity') != run_identity:
            raise RuntimeError('Source, runtime or requested experiment changed; choose a new output directory')
        if old.get('status') != 'complete' or set(old.get('methods',{})) != set(modes):
            raise RuntimeError('Incomplete result preserved; choose a new output directory for an explicit rerun')
        return {'task':task['task_id'],'status':'cached_complete',
                'all_winners_qualified':all(row.get('winner_qualified') for row in old['methods'].values())}
    exp=None;methods={};started=time.perf_counter()
    result={'task':task,'settings':settings,'source_hashes':source,'run_identity':run_identity,
            'training_hash':None,'methods':methods,
            'timing_scope':'fresh pipelines with common compact checkpoint; test metrics outside' if direct else 'development shared operation ledger'}
    ordered=list(modes[index%len(modes):]+modes[:index%len(modes)]) if direct else modes
    result['execution_order']=ordered
    try:
        data=generate(task)
        result['training_hash']=hashlib.sha256(data['Xtrain'].tobytes()).hexdigest()
        for mode in ordered:
            method_started=time.perf_counter()
            if exp is None or direct: exp=Experiment(task,settings,data,Path(output)/'STOP')
            row=exp.method(mode)
            if direct:
                write(Path(output)/'timed_checkpoints'/(task['task_id']+'_'+mode+'.json'),
                      {'method':row, 'operations':exp.ops, 'run_identity':run_identity,
                       'training_hash':result['training_hash']})
                row['direct_seconds']=time.perf_counter()-method_started
            row['quality']=exp.quality(row['winning_key'])
            if direct: row['operations']=exp.ops
            methods[mode]=row
            print(json.dumps({'task':task['task_id'],'method':mode,'bic':row['bic'],
                              'fits':row['full_fits'],'seconds':row['accounted_seconds']}),flush=True)
            # One checkpoint per completed method, outside any direct timer.
            result['status']='partial'
            if not direct: result['operations']=exp.ops
            write(path,result)
        result['status']='complete'
    except (Exception, DeadlineExceeded, StopRequested) as error:
        result.update(status='interrupted' if isinstance(error,StopRequested) else 'failed',
                      error=str(error),traceback=traceback.format_exc())
        if exp is not None: result['operations']=exp.ops
    result['elapsed_seconds']=time.perf_counter()-started
    if source!=fingerprint(): result['status']='source_changed_during_run'
    write(path,result)
    return {'task':task['task_id'],'status':result['status'],'error':result.get('error'),'seconds':result['elapsed_seconds']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='development_l5')
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--output',default='development_results_v1')
    parser.add_argument('--limit',type=int);parser.add_argument('--direct',action='store_true')
    parser.add_argument('--methods',nargs='+',choices=MODES)
    args=parser.parse_args();protocol=json.loads((HERE/'pilot_protocol.json').read_text())
    tasks=protocol['phases'][args.phase]
    if args.limit:tasks=tasks[:args.limit]
    modes=args.methods or list(MODES)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,t,protocol['settings'],HERE/args.output,modes,args.direct,i) for i,t in enumerate(tasks)]
        for f in as_completed(futures): print(json.dumps(f.result()),flush=True)


if __name__=='__main__': main()
