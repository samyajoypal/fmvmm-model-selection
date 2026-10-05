"""Bounded full-library development checks using the public FMVMM soft-EM API.

Capability cases fit a prescribed mixed-family model from data-only starts.
They do not estimate selection accuracy. Score-table cases expose every family
to the same pilot weights. Development and reserved validation seeds differ.
No HPC operations are performed by this script.
"""
from local_pilot import *
from fmvmm.numerics.gh_policy import GH_FAMILIES, unpack_gh_tail

PROTOCOL = HERE / 'l13_readiness_protocol.json'
FAMILIES = tuple(LIBRARIES['L13'])


def make_protocol():
    settings = json.loads((HERE/'pilot_protocol.json').read_text())['settings']
    # Operational deadline only; all candidates have the same stopping rules.
    settings['fit_deadline_seconds'] = 600
    settings['profile_deadline_seconds'] = 240
    settings['profile_iterations'] = 1000
    groups = {'capability': [], 'tables': [], 'reserved_validation': []}
    for p in range(2,6):
        for k in range(3,6):
            for block in range((13+k-1)//k):
                names = [FAMILIES[(block*k+j)%13] for j in range(k)]
                seed = 710000 + 10000*p + 100*k + block
                task = dict(task_id=f'cap_p{p}_k{k}_b{block}', kind='simulation',
                            library_id='L13', n=80*k+20*p, p=p, true_k=k, fit_k=k,
                            truth_families=names, separation=4.5,
                            weights='balanced' if block%2==0 else 'unequal',
                            vg_shape_gap=.25 if block%2==0 else 1.5,
                            seed=seed, data_seed=seed, test_n=500, phase='development')
                groups['capability'].append(task)
                validation = copy.deepcopy(task)
                validation.update(task_id=task['task_id'].replace('cap_', 'reserved_'),
                                  seed=seed+900000, data_seed=seed+900000, phase='validation')
                groups['reserved_validation'].append(validation)
            # Rotate the truth blocks so every p,K cell has a predeclared example.
            task = copy.deepcopy(groups['capability'][-((13+k-1)//k)+(p+k)%((13+k-1)//k)])
            task.update(task_id=f'table_p{p}_k{k}', seed=seed+100000,
                        data_seed=seed+100000, separation=3.)
            groups['tables'].append(task)
    return dict(version='full13-readiness-v2', families=FAMILIES, settings=settings,
                phases=groups, validation_locked=True,
                scope='Development capability and score-table checks, not an exhaustive selection study.',
                gate='Every attempted fit and selected profile must qualify; failures remain in the denominator.',
                starts='K-means and weighted moments use X only; truth labels/parameters are never supplied to fitting.',
                validation_rule='Release only after numerical source and methodology are frozen; never tune to reserved results.')


def truth_parameters(task):
    p,k = task['p'],task['true_k']
    angle = np.arange(k)*2*np.pi/k
    centers = np.zeros((k,p))
    centers[:,:2] = np.column_stack((np.cos(angle),np.sin(angle)))*task['separation']/(2*np.sin(np.pi/k))
    weights = np.ones(k) if task['weights']=='balanced' else np.geomspace(1.,.3,k)
    weights /= weights.sum()
    alpha=[]
    for j,name in enumerate(task['truth_families']):
        rho = .25 if j%2==0 else -.2
        sigma = (.8+.2*(j%3))*rho**np.abs(np.arange(p)[:,None]-np.arange(p)[None,:])
        mu=centers[j].copy()
        shape=np.zeros(p);shape[:2]=[3.,-1.25]
        shape *= (-1)**j
        gamma=.35*shape/np.linalg.norm(shape)
        if name=='mvn': a=(mu,sigma)
        elif name=='mvt': a=(mu,sigma*.6,np.array([5.]))
        elif name=='mvsn': a=(mu,sigma,shape)
        elif name=='mvst': a=(mu,sigma*.6,shape,np.array([5.]))
        elif name=='msnc': a=(mu,sigma,shape,np.array([.12,.25]))
        elif name=='msl': a=(mu,sigma*.6,np.array([2.5]))
        elif name=='mssl': a=(mu,sigma*.6,shape,np.array([2.5]))
        elif name=='mvsl': a=(mu-gamma,sigma/(p+1),gamma/(p+1))
        elif name in GH_FAMILIES:
            tail = ([.7,np.log(1.5)] if name=='mghp' else [np.log(2.)] if name=='mgst'
                    else [np.log(p/2+task['vg_shape_gap'])] if name=='mvvg' else [np.log(1.5)])
            a=tuple(unpack_gh_tail(name,tail,p))+ (mu-gamma,sigma,gamma)
        else: raise ValueError(name)
        alpha.append(a)
    return dict(pi=weights, families=task['truth_families'], alpha=alpha, centers=centers,
                contamination=0., center_interpretation='Location geometry; not equal component means for SMSN.')


def generate_l13(task):
    from data_generation import true_posterior
    truth=truth_parameters(task);seed=int(task['data_seed']);test_seed=(seed+50000017)%2**32
    x,y=sample_l13(task['n'],truth,seed);xt,yt=sample_l13(task['test_n'],truth,test_seed)
    return dict(Xtrain=x,Xtest=xt,ytrain=y,ytest=yt,qtrue=true_posterior(x,truth),
                qtrue_test=true_posterior(xt,truth),
                metadata=dict(truth=truth,training_seed=seed,test_seed=test_seed,
                              sampling_streams='Labels: default_rng(seed); component j: SeedSequence([seed,j,13]); MVSL explicit random_state.',
                              test_observations_independent=True))


def sample_l13(n,truth,seed):
    labels=np.random.default_rng(seed).choice(len(truth['pi']),size=n,p=truth['pi'])
    p=truth['centers'].shape[1];x=np.empty((n,p))
    for j,(name,alpha) in enumerate(zip(truth['families'],truth['alpha'])):
        indices=np.flatnonzero(labels==j)
        if not len(indices): continue
        component_seed=int(np.random.SeedSequence([seed,j,13]).generate_state(1)[0])
        np.random.seed(component_seed)
        kwargs={'random_state':component_seed} if name=='mvsl' else {}
        x[indices]=np.asarray(dist_map[name].rvs(*alpha,size=len(indices),**kwargs)).reshape(len(indices),p)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite native random variates.')
    return x,labels


class ReadinessExperiment(Experiment):
    def __init__(self,task,settings,data,stop_file=None):
        # Avoid materializing 13**K assignments just to fit one capability model.
        self.task,self.settings,self.data,self.stop_file=task,settings,data,stop_file
        self.policy=EMPolicy(**settings['numerical_policy'])
        raw=data['Xtrain'];self.mean=raw.mean(0)
        self.cholesky=np.linalg.cholesky(np.cov(raw,rowvar=False,bias=True))
        self.x=solve_triangular(self.cholesky,(raw-self.mean).T,lower=True).T
        self.k,self.library=task['fit_k'],list(FAMILIES)
        self.ops,self.pilots,self.dependencies={},{},{}
        self.active_operation=None

    def full_fit(self,names,pi,alpha):
        self.active_operation='public_fit:'+','.join(names)
        return super().full_fit(names,pi,alpha)

    def profile(self,label,j,name):
        self.active_operation=f'profile:{label}:{j}:{name}'
        return super().profile(label,j,name)

    def capability(self):
        # Seeded K-means labels define the prescribed candidate ordering;
        # they are not matched to the known truth labels.
        labels=KMeans(n_clusters=self.k,n_init=10,random_state=self.task['seed']).fit_predict(self.x)
        weights=np.eye(self.k)[labels];alpha=[];preparations=[]
        for j,name in enumerate(self.task['truth_families']):
            mu,sigma=moments(self.x,weights[:,j]);sigma+=self.policy.covariance_floor*np.eye(self.x.shape[1])
            initial,prep=prepare_component(name,initial_parameters(name,mu,sigma),self.policy)
            alpha.append(initial);preparations.append(prep)
        row=self.full_fit(tuple(self.task['truth_families']),weights.mean(0),alpha)
        row['initialization_projection']=preparations
        self.ops['capability_fit']=row
        return dict(qualified=row['qualified'],scope='One prescribed mixed-family fit; no family selection claim.',
                    quality=self.quality('capability_fit'))

    def score_table(self):
        self.student();scores=np.full((self.k,len(self.library)),np.nan);failures=[];attempts=[]
        # Continue independent cells after a failure, but never use that table to screen.
        for j in range(self.k):
            for h,name in enumerate(self.library):
                start=time.perf_counter();status='complete'
                try: scores[j,h]=self.profile('student',j,name)[1]['score']
                except (Exception,DeadlineExceeded) as error:
                    status='failed'
                    failures.append(dict(component=j,family=name,error=str(error),traceback=traceback.format_exc()))
                key=f'profile:student:{j}:{name}'
                attempts.append(dict(component=j,family=name,status=status,seconds=time.perf_counter()-start,
                                     completed_start_count=len(self.ops[key]['starts']) if key in self.ops else None,
                                     partial_start_count_available=key in self.ops))
                self.check_stop()
        qualified=(not failures and all(v['qualified'] for key,v in self.ops.items() if key.startswith('pilot:')))
        result=dict(qualified=qualified,scores=scores,failures=failures,profile_attempts=attempts,
                    scope='Full 13-family score table; no selected-mixture fits or speedup claim.')
        if qualified:
            gaps=scores.max(1,keepdims=True)-scores
            result['ranked_catalogue']=ranked_catalogue(gaps,self.library,len(self.x))
        return result


def ranked_catalogue(gaps,library,n):
    """Stream unique assignments; identical rule to the frozen ranked screen."""
    k=len(gaps);index={name:i for i,name in enumerate(library)};rows=[]
    for structure in itertools.combinations_with_replacement(sorted(library),k):
        deficit,assignment=min((float(sum(gaps[j,index[name]] for j,name in enumerate(a))),a)
                               for a in set(itertools.permutations(structure)))
        rows.append(dict(structure=structure,assignment=assignment,deficit=deficit,
                         retained=deficit<=2*np.log(n)+1e-10 or structure==('mvn',)*k))
    return rows


def source_hashes():
    result=fingerprint()
    for path in (Path(__file__),PROTOCOL):
        result[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_task(task,settings,phase,output):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    with (folder/(task['task_id']+'.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=folder/(task['task_id']+'.json');source=source_hashes()
        identity=dict(task=task,settings=settings,phase=phase,source_hashes=source,runtime=runtime())
        if path.exists():
            old=loads(path.read_text())
            if old.get('run_identity')!=identity or old.get('status')!='complete':
                raise RuntimeError('Existing incomplete or different-source result preserved; use a new output directory.')
            return dict(task=task['task_id'],status='cached_complete')
        exp=None;start=time.perf_counter();result=dict(run_identity=identity,task=task)
        try:
            data=generate_l13(task)
            result.update(training_hash=hashlib.sha256(data['Xtrain'].tobytes()).hexdigest(),metadata=data['metadata'])
            exp=ReadinessExperiment(task,settings,data,folder/'STOP')
            result['result']=exp.capability() if phase=='capability' else exp.score_table()
            result['status']='complete' if result['result']['qualified'] else 'unqualified'
        except (Exception,DeadlineExceeded,StopRequested) as error:
            result.update(status='interrupted' if isinstance(error,StopRequested) else 'failed',
                          error=str(error),traceback=traceback.format_exc())
        if exp is not None: result.update(operations=exp.ops,active_operation=exp.active_operation)
        result['elapsed_seconds']=time.perf_counter()-start
        if source!=source_hashes(): result['status']='source_changed_during_run'
        write(path,result)
        return dict(task=task['task_id'],status=result['status'],seconds=result['elapsed_seconds'],error=result.get('error'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-protocol',action='store_true')
    parser.add_argument('--phase',choices=('capability','tables'),default='capability')
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--limit',type=int)
    parser.add_argument('--task');parser.add_argument('--output',default='l13_readiness_development_v2')
    args=parser.parse_args()
    if args.write_protocol:
        if PROTOCOL.exists(): raise RuntimeError('Refusing to overwrite a registered protocol.')
        write(PROTOCOL,make_protocol());return
    protocol=loads(PROTOCOL.read_text());tasks=protocol['phases'][args.phase]
    if args.task: tasks=[t for t in tasks if t['task_id']==args.task]
    if args.limit: tasks=tasks[:args.limit]
    if not tasks: raise ValueError('No matching tasks.')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_task,t,protocol['settings'],args.phase,HERE/args.output) for t in tasks]
        for future in as_completed(futures): print(json.dumps(future.result()),flush=True)


if __name__=='__main__': main()
