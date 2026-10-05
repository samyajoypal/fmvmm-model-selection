"""Bounded component M-steps for the existing FMVMM soft-EM loop.

No full-mixture optimizer is used here. The likelihood of one component is
maximized with the current EM membership weights held fixed. The covariance
parameter is L L' + covariance_floor I; bounds on L, location and tail/shape
parameters are part of the declared working model, not density repairs.
"""
from dataclasses import asdict
from types import SimpleNamespace
import time
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import log_ndtr, erfcx, ndtr
from .em_policy import make_policy

from .gh_policy import GH_FAMILIES, canonicalize_gh, pack_gh_tail, unpack_gh_tail, gh_tail_bounds
SUPPORTED=('mvn','mvt','mvsn','mvst','msnc','msl','mssl','mvsl',*sorted(GH_FAMILIES))
LOG2PI=np.log(2*np.pi)


def _module(name):
    from fmvmm.mixtures.FMVMM import dist_map
    if name not in SUPPORTED:
        raise NotImplementedError(f'Constrained EM does not yet support {name}; no unconstrained fallback is used.')
    return dist_map[name]


def _bounds(name,p,policy):
    if name in ('msl','mssl') and policy.slash_df_max + p/2 > 500:
        raise ValueError(
            'The configured slash domain requires slash_df_max + p/2 <= 500; '
            'choose and report an explicit smaller slash_df_max before fitting.'
        )
    base=([(-policy.location_bound,policy.location_bound)]*p+
          [(np.log(policy.cholesky_min),np.log(policy.cholesky_max))]*p+
          [(-policy.cholesky_max,policy.cholesky_max)]*(p*(p-1)//2))
    if name in ('mvsn','mvst','msnc','mssl'):
        base += [(-policy.skew_coefficient_bound,policy.skew_coefficient_bound)]*p
    if name in ('mvt','mvst'):
        base += [(0.,policy.inverse_df_max)]
    if name=='msnc':
        base += [(policy.contamination_min,1-policy.contamination_min),
                 (policy.inverse_inflation_min,1.)]
    if name in ('msl','mssl'):
        base += [(np.log(policy.slash_df_min),np.log(policy.slash_df_max))]
    if name in GH_FAMILIES or name=='mvsl':
        base += [(-policy.skew_coefficient_bound,policy.skew_coefficient_bound)]*p
        if name in GH_FAMILIES: base += gh_tail_bounds(name,p,policy)
    return base


def _unpack(u,name,p,policy):
    mu=np.asarray(u[:p]); L=np.diag(np.exp(u[p:2*p]))
    m=p*(p-1)//2; L[np.tril_indices(p,-1)]=u[2*p:2*p+m]
    sigma=L@L.T+policy.covariance_floor*np.eye(p)
    extra=np.asarray(u[2*p+m:]); alpha=(mu,sigma)
    if name in ('mvsn','mvst','msnc','mssl'):
        vals,vecs=np.linalg.eigh(sigma)
        alpha += (((vecs*np.sqrt(vals))@vecs.T)@extra[:p],)
        extra=extra[p:]
    if name in ('mvt','mvst'):
        alpha += (np.array([np.inf if extra[0]==0 else 1./extra[0]]),)
    if name=='msnc': alpha += (extra.copy(),)
    if name in ('msl','mssl'): alpha += (np.array([np.exp(extra[0])]),)
    if name in GH_FAMILIES:
        alpha=tuple(unpack_gh_tail(name,extra[p:],p,policy))+(mu,sigma,extra[:p].copy())
    if name=='mvsl': alpha=(mu,sigma/(p+1),extra[:p].copy()/(p+1))
    return alpha,L


def _pack(name,alpha,policy,project):
    _module(name)
    if name in GH_FAMILIES:
        if project: alpha=canonicalize_gh(name,alpha)
        mu=np.asarray(alpha[-3],float).reshape(-1); sigma=np.asarray(alpha[-2],float)
    else:
        mu=np.asarray(alpha[0],float).reshape(-1); sigma=np.asarray(alpha[1],float)
    p=len(mu)
    if name=='mvsl': sigma=sigma*(p+1)
    if sigma.shape!=(p,p) or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError('Invalid location or scale.')
    if not np.allclose(sigma,sigma.T,rtol=1e-10,atol=1e-12):
        raise ValueError('Scale is not symmetric.')
    vals,vecs=np.linalg.eigh(sigma)
    if np.min(vals)<=0:
        raise ValueError('Scale must be strictly positive definite even at initialization.')
    base=(sigma+sigma.T)*.5-policy.covariance_floor*np.eye(p)
    projected_scale=False
    if project:
        # Explicit initialization projection only; no objective evaluation repairs.
        adjusted=np.maximum(vals-policy.covariance_floor,policy.cholesky_min**2)
        base=(vecs*adjusted)@vecs.T
        projected_scale=bool(np.min(vals-policy.covariance_floor)<policy.cholesky_min**2)
    encoding={}
    if project:
        L=np.linalg.cholesky(base)
        spatial=np.r_[np.log(np.diag(L)),L[np.tril_indices(p,-1)]]
    else:
        from .verified_codec import encode_native_scale
        spatial,L,_,encoding['scale']=encode_native_scale(sigma,policy)
    u=np.r_[mu,spatial]
    idx=2
    if name in ('mvsn','mvst','msnc','mssl'):
        lam=np.asarray(alpha[idx],float).reshape(-1);idx+=1
        if lam.shape!=(p,) or not np.isfinite(lam).all(): raise ValueError('Invalid shape parameter.')
        b=((vecs/np.sqrt(vals))@vecs.T)@lam
        u=np.r_[u,b]
    if name in ('mvt','mvst'):
        df=np.asarray(alpha[idx],float).reshape(-1)
        if df.size!=1 or np.isnan(df[0]) or df[0]<=0: raise ValueError('Invalid degrees of freedom.')
        u=np.r_[u,0. if np.isposinf(df[0]) else 1./df[0]]
    if name=='msnc':
        tail=np.asarray(alpha[idx],float).reshape(-1)
        if tail.shape!=(2,) or not np.isfinite(tail).all(): raise ValueError('Invalid contamination parameters.')
        u=np.r_[u,tail]
    if name in ('msl','mssl'):
        df=np.asarray(alpha[idx],float).reshape(-1)
        if df.size!=1 or not np.isfinite(df[0]) or df[0]<=0: raise ValueError('Invalid slash degrees of freedom.')
        u=np.r_[u,np.log(df[0])]
    if name in GH_FAMILIES:
        if project:tail=pack_gh_tail(name,alpha,p,policy,check_bounds=False)
        else:
            from .verified_codec import encode_gh_tail
            tail,encoding['tail']=encode_gh_tail(name,alpha,p,policy)
        u=np.r_[u,np.asarray(alpha[-1],float),tail]
    if name=='mvsl': u=np.r_[u,np.asarray(alpha[2],float)*(p+1)]
    limits=np.asarray(_bounds(name,p,policy))
    projected_coordinates=np.flatnonzero((u<limits[:,0])|(u>limits[:,1])).tolist()
    if project: u=np.clip(u,limits[:,0],limits[:,1])
    elif np.any(u<limits[:,0]-1e-8) or np.any(u>limits[:,1]+1e-8):
        raise ValueError('Previous component is outside the declared parameter domain.')
    else: u=np.clip(u,limits[:,0],limits[:,1])
    return u,{'scale_projected':projected_scale,'bounded_coordinates_projected':projected_coordinates,'native_encoding':encoding}


def prepare_component(dist_name,alpha,policy):
    """Explicitly put an initialization into the declared domain; report changes."""
    policy=make_policy(policy)
    u,info=_pack(dist_name,alpha,policy,True)
    mu=alpha[-3] if dist_name in GH_FAMILIES else alpha[0]
    native,_=_unpack(u,dist_name,len(np.asarray(mu).reshape(-1)),policy)
    info['optimizer_coordinates_vector']=u.tolist()
    if dist_name in GH_FAMILIES: info['latent_scale_normalization']='E[W]=1 before domain projection'
    if dist_name=='mvsl': info['latent_scale_normalization']='fixed E[W]=p+1; bounds on (p+1)*Sigma and (p+1)*gamma'
    return native,info


def _mills(z):
    out=np.empty_like(z);neg=z<0
    out[neg]=np.sqrt(2/np.pi)/erfcx(-z[neg]/np.sqrt(2))
    out[~neg]=np.exp(-.5*z[~neg]**2-.5*LOG2PI)/ndtr(z[~neg])
    return out


def _tail_eta_derivative(distance,eta,p):
    t=eta*distance;h=np.empty_like(t);small=np.abs(t)<1e-3;v=t[small]
    h[small]=.5+v*(-2/3+v*(3/4+v*(-4/5+v*(5/6-v*6/7))))
    v=t[~small];h[~small]=(np.log1p(v)-v/(1+v))/(v*v)
    return .5*distance**2*h-.5*p*distance/(1+t)


def _analytic(x,u,name,policy,w):
    if name in ('msl','mssl'):
        from .weighted_slash_score import values_gradient
        return values_gradient(x,u,name,policy,w)
    if name in GH_FAMILIES or name=='mvsl':
        from .weighted_gh_score import values_gradient
        return values_gradient(x,u,name,policy,w)
    if name == 'msnc':
        from .weighted_msnc_score import values_gradient
        return values_gradient(x,u,policy,w)
    if name == 'mvst':
        from .weighted_skew_t_score import values_gradient
        return values_gradient(x,u,policy,w)
    from .robust_mvt import log_normalizer_eta,derivative_log_normalizer_eta
    n,p=x.shape; alpha,L=_unpack(u,name,p,policy);mu,sigma=alpha[:2]
    C=np.linalg.cholesky(sigma);inv=cho_solve((C,True),np.eye(p))
    d=x-mu;v=d@inv;r2=np.einsum('ij,ij->i',d,v);r=np.ones(n)
    logdet=2*np.log(np.diag(C)).sum();extra=u[p+p*(p+1)//2:]
    if name=='mvt':
        eta=float(extra[0]);cn=log_normalizer_eta(eta,p)
        values=cn-.5*logdet-.5*(1+p*eta)*(r2 if eta==0 else np.log1p(eta*r2)/eta)
        r=(1+p*eta)/(1+eta*r2)
    else:
        values=-.5*(p*LOG2PI+logdet+r2)
        if name=='mvsn': z=d@extra;values+=np.log(2)+log_ndtr(z)
    mu_grad=((w*r)[:,None]*v).sum(0)
    G=.5*(v.T@((w*r)[:,None]*v)-w.sum()*inv);dL=2*G@L
    gradient=np.r_[mu_grad,np.diag(dL)*np.diag(L),dL[np.tril_indices(p,-1)]]
    if name=='mvt': gradient=np.r_[gradient,w.sum()*derivative_log_normalizer_eta(eta,p)+w@_tail_eta_derivative(r2,eta,p)]
    if name=='mvsn':
        wm=w*_mills(z);gradient[:p]-=wm.sum()*extra;gradient=np.r_[gradient,d.T@wm]
    return values,gradient


def _projected_gradient(u,g,limits):
    a=np.asarray(limits);g=np.array(g,copy=True)
    g[(u<=a[:,0]+1e-9)&(g>0)]=0;g[(u>=a[:,1]-1e-9)&(g<0)]=0
    return float(np.max(np.abs(g)))


def _component_coordinates(name,alpha,policy,p,x,w,coordinates=None):
    from .verified_codec import verify_native_likelihood
    if coordinates is None:
        u,encoding=_pack(name,alpha,policy,False)
        source='native_tuple_with_verified_encoding'
    else:
        u=np.asarray(coordinates,float).copy();limits=np.asarray(_bounds(name,p,policy))
        if u.shape!=(len(limits),) or not np.isfinite(u).all() or np.any(u<limits[:,0]) or np.any(u>limits[:,1]):
            raise ValueError('Retained optimizer coordinates are outside the declared domain.')
        encoding={};source='retained_optimizer_coordinates'
    decoded,_=_unpack(u,name,p,policy)
    agreement=verify_native_likelihood(_module(name),x,w,alpha,decoded)
    return u,{'source':source,'encoding':encoding,'native_agreement':agreement}


def fit_component(dist_name,X,weights,alpha_prev,policy,*,maxiter=None,coordinates_prev=None):
    """Return native parameters and diagnostics for one constrained GEM update.

    The incoming alpha must already be in the working domain (use
    prepare_component once for initialization). Zero-weight observations are
    removed before density evaluation. Density errors propagate as failures.
    """
    started=time.perf_counter(); policy=make_policy(policy);module=_module(dist_name)
    X=np.asarray(X,float);weights=np.asarray(weights,float).reshape(-1)
    if X.ndim!=2 or weights.shape!=(len(X),) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Invalid observations or component weights.')
    mask=weights>0; x=X[mask];w=weights[mask];mass=w.sum()
    if mass<=0: raise ValueError('Component has zero effective membership mass.')
    if not np.isfinite(x).all(): raise ValueError('Positive-weight observations must be finite.')
    p=X.shape[1];u0,coordinate_recovery=_component_coordinates(dist_name,alpha_prev,policy,p,x,w,coordinates_prev)
    limits=_bounds(dist_name,p,policy)
    scored=dist_name in ('mvn','mvt','mvsn','msnc','mvst','msl','mssl','mvsl',*GH_FAMILIES)
    cusp_family=dist_name in ('mvvg','mvsl')
    calls=0
    def native_value(alpha):
        ld=np.asarray(module.logpdf(x,*alpha),float).reshape(-1)
        if ld.shape!=(len(x),) or not np.isfinite(ld).all():
            raise FloatingPointError(f'Nonfinite {dist_name} log density in positive-weight objective.')
        return -float(w@ld)/mass
    def native_at(u):
        nonlocal calls
        calls+=1
        alpha,_=_unpack(u,dist_name,p,policy)
        return native_value(alpha)
    best_residual=[None]
    def selection_residual(z,gradient=None):
        if gradient is not None and not cusp_family:
            return _projected_gradient(z,gradient,limits)
        budget.evaluate()  # An uncached tie-selection score is optimization work.
        native,_=_unpack(z,dist_name,p,policy)
        return component_stationarity(dist_name,x,w,native,policy,coordinates=z)['projected_gradient']
    def consider(value,z,gradient=None):
        if value<best[0]:
            best[:]=[value,np.array(z,copy=True)]
            best_residual[0]=selection_residual(z,gradient) if gradient is not None and not cusp_family else None
        elif value==best[0]:
            residual=selection_residual(z,gradient)
            if best_residual[0] is None:
                best_residual[0]=residual if np.array_equal(z,best[1]) else selection_residual(best[1])
            if residual<best_residual[0]:
                best[:]=[value,np.array(z,copy=True)];best_residual[0]=residual
    def objective(u):
        nonlocal calls
        calls+=1
        if scored:
            from .weighted_gh_score import NonsmoothLocationError
            try:
                ld,g=_analytic(x,u,dist_name,policy,w)
            except NonsmoothLocationError:
                # Preserve an exactly encountered cusp for its explicit
                # location-fixed continuation; never invent its gradient.
                value=native_at(u)
                consider(value,u)
                raise
            value=-float(w@ld)/mass;g=-g/mass
            if not np.isfinite(value) or not np.isfinite(g).all(): raise FloatingPointError('Nonfinite weighted objective/gradient.')
            consider(value,u,g)
            return value,g
        value=native_at(u)
        consider(value,u)
        return value
    initial=native_value(alpha_prev);best=[initial,u0.copy()]
    maxiter=policy.component_maxiter if maxiter is None else int(maxiter)
    if maxiter<=0:raise ValueError('Positive component iteration budget required.')
    from .component_budget import Budget,BudgetExhausted,solve
    budget=Budget(maxiter,int(policy.component_maxfun))
    optimizer_records=[];anchor_records=[];escape_records=[]
    rounds=int(policy.cusp_max_rounds) if cusp_family else 1
    for round_index in range(rounds):
        u=best[1];fixed=False
        if cusp_family:
            from .cusp_stationarity import location_classification, generalized_stationarity, anchor_continuation, leave_failed_norm_anchor
            fixed=location_classification(x,w,u,dist_name,policy)['location_nonsmooth']
        if not fixed and budget.remaining_iterations and budget.remaining_evaluations:
            from .weighted_gh_score import NonsmoothLocationError
            iterations_before,evaluations_before=budget.iterations,budget.evaluations
            try:
                solved=solve(objective,u,limits,policy,budget,restart_ftol=policy.restart_ftol,
                             initial_ftol=policy.ftol if round_index==0 else policy.restart_ftol,
                             max_restarts=0 if cusp_family else None)
                for row in solved['runs']:
                    optimizer_records.append(dict(row,hit_cap=not row['remaining_iterations'] or not row['remaining_evaluations']))
            except NonsmoothLocationError as error:
                optimizer_records.append({'success':False,'message':'Interrupted at exact cusp: '+str(error),
                    'iterations':budget.iterations-iterations_before,'evaluations':budget.evaluations-evaluations_before,
                    'hit_cap':not budget.remaining_iterations or not budget.remaining_evaluations})
        if not cusp_family:break
        if budget.remaining_iterations and budget.remaining_evaluations:
            try:
                u,attempts=anchor_continuation(x,w,best[1],dist_name,policy,maxiter,native_at,budget=budget)
                anchor_records.extend(attempts);value=native_at(u)
                consider(value,u)
            except BudgetExhausted:
                break
        state=generalized_stationarity(x,w,best[1],dist_name,policy)
        if state['gradient_acceptable']:break
        if not budget.remaining_iterations or not budget.remaining_evaluations:break
        try:
            escaped,escape=leave_failed_norm_anchor(x,w,best[1],dist_name,policy,native_at,budget=budget)
        except BudgetExhausted:break
        escape_records.append(escape)
        if escape['accepted']:consider(native_at(escaped),escaped)
        elif state['stationarity_kind']=='norm_subgradient_and_smooth_score' and not state['location_condition_satisfied']:break
    u=best[1];candidate,_=_unpack(u,dist_name,p,policy);final=native_value(candidate)
    accepted=final<=initial+policy.accept_tolerance/mass
    if not accepted:candidate=alpha_prev;final=initial;u=u0.copy()
    stationarity=component_stationarity(dist_name,x,w,candidate,policy,coordinates=u)
    pg=stationarity['projected_gradient']
    if scored:
        if cusp_family:
            from .weighted_gh_score import values_gradient
            ld,_=values_gradient(x,u,dist_name,policy,w,fixed_location=stationarity['location_nonsmooth'])
        else:ld,_=_analytic(x,u,dist_name,policy,w)
        if not np.isclose(-float(w@ld)/mass,final,rtol=2e-10,atol=2e-9):
            raise FloatingPointError('Scored component objective disagrees with native density.')
    cusp_round_cap=bool(cusp_family and round_index+1>=rounds and not stationarity['gradient_acceptable'])
    any_cap=not budget.remaining_iterations or not budget.remaining_evaluations or cusp_round_cap
    total_iterations=budget.iterations
    optimizer_success=bool(optimizer_records[-1]['success']) if optimizer_records else False
    message=optimizer_records[-1]['message'] if optimizer_records else 'Explicit location-fixed cusp continuation'
    labels=([f'location[{j}]' for j in range(p)]+[f'log_cholesky_diag[{j}]' for j in range(p)]+
            [f'cholesky[{i},{j}]' for i,j in zip(*np.tril_indices(p,-1))])
    if dist_name in ('mvsn','mvst','msnc','mssl'): labels += [f'skew_coefficient[{j}]' for j in range(p)]
    if dist_name in ('mvt','mvst'): labels += ['inverse_df']
    if dist_name=='msnc': labels += ['contamination_probability','inverse_inflation']
    if dist_name in ('msl','mssl'): labels += ['log_slash_df']
    if dist_name in GH_FAMILIES or dist_name=='mvsl': labels += [f'canonical_gamma[{j}]' for j in range(p)]
    if dist_name in GH_FAMILIES: labels += [f'gh_tail[{j}]' for j in range(len(gh_tail_bounds(dist_name,p,policy)))]
    boundary=[{'coordinate':labels[j],'index':j,'value':float(v),'side':'lower' if v-lo <= hi-v else 'upper',
               'lower':float(lo),'upper':float(hi)} for j,(v,(lo,hi)) in enumerate(zip(u,limits))
              if min(v-lo,hi-v)<=1e-7]
    scale=candidate[-2] if dist_name in GH_FAMILIES else candidate[1]*(p+1) if dist_name=='mvsl' else candidate[1]
    minimum_scale=float(np.linalg.eigvalsh(scale).min())
    diag={'family':dist_name,'weighted_loglik_before':-initial*mass,'weighted_loglik':-final*mass,
          'accepted':bool(accepted),'projected_gradient':pg,'gradient_acceptable':stationarity['gradient_acceptable'],
          'optimizer_success':optimizer_success,'message':message,'iterations':total_iterations,
          'evaluations':budget.evaluations,'native_validation_and_proposal_calls':calls,'hit_cap':any_cap,
          'iteration_budget':budget.max_iterations,'evaluation_budget':budget.max_evaluations,
          'remaining_iterations':budget.remaining_iterations,'remaining_evaluations':budget.remaining_evaluations,
          'budget_scope':'One total optimization budget across ordinary solves, restarts, anchor continuations and escape proposals; final native validity/stationarity checks are separate.',
          'coordinate_recovery':coordinate_recovery,'optimizer_coordinates_vector':u.tolist(),
          'finite':bool(np.isfinite(final) and np.isfinite(pg)),'mass':float(mass),
          'minimum_scale_eigenvalue':minimum_scale, 'scale_floor_active':bool(minimum_scale<=policy.covariance_floor*(1+1e-3)),
          'active_parameter_bounds':boundary, 'optimizer_coordinates':dict(zip(labels,map(float,u))),
          'seconds':time.perf_counter()-started,'analytic_gradient':dist_name in ('mvn','mvt','mvsn','msnc','mvsl'),
          'gradient_method': 'semi_analytic_eta_difference' if dist_name=='mvst' else 'analytic_spatial_native_tail_difference' if dist_name in (*GH_FAMILIES,'msl','mssl') else 'analytic' if scored else 'finite_difference',
          'stationarity':stationarity,'optimizer_runs':optimizer_records,
          'anchor_attempts':anchor_records,'norm_cusp_escape_attempts':escape_records,
          'cusp_round_limit':policy.cusp_max_rounds if cusp_family else None,'cusp_round_cap':cusp_round_cap,
          'global_optimum_certified':False}
    if 'gradient_method' in stationarity.get('score_diagnostics',{}):
        diag['gradient_method']=stationarity['score_diagnostics']['gradient_method']
    return candidate,diag


def component_stationarity(dist_name,X,weights,alpha,policy,*,coordinates=None):
    """Recomputed ordinary/generalized stationarity for the supplied memberships.

    The historical ``gradient_acceptable`` key is retained for API compatibility;
    ``stationarity_kind`` states whether an ordinary score or a derived cusp
    condition was actually used. No full location derivative exists at a cusp.
    """
    policy=make_policy(policy);module=_module(dist_name)
    X=np.asarray(X,float);weights=np.asarray(weights,float).reshape(-1)
    if X.ndim!=2 or weights.shape!=(len(X),) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Invalid weighted observations.')
    mask=weights>0;x=X[mask];w=weights[mask];mass=w.sum();p=X.shape[1]
    if mass<=0 or not np.isfinite(x).all():raise ValueError('Need positive finite effective component data.')
    u,recovery=_component_coordinates(dist_name,alpha,policy,p,x,w,coordinates);limits=_bounds(dist_name,p,policy)
    if dist_name in ('mvvg','mvsl'):
        from .cusp_stationarity import generalized_stationarity
        result=generalized_stationarity(x,w,u,dist_name,policy)
        result['coordinate_recovery']=recovery
        return result
    score_metadata={}
    if dist_name in ('mvn','mvt','mvsn','msnc','mvst','msl','mssl',*GH_FAMILIES):
        if dist_name in ('msl','mssl'):
            from .weighted_slash_score import values_gradient
            ld,g,score_metadata=values_gradient(x,u,dist_name,policy,w,return_diagnostics=True)
        else:ld,g=_analytic(x,u,dist_name,policy,w)
        g=-g/mass
        native=np.asarray(module.logpdf(x,*alpha),float).reshape(-1)
        if not np.isfinite(native).all() or not np.allclose(ld,native,rtol=2e-10,atol=2e-9):
            raise FloatingPointError('Stationarity objective disagrees with native density.')
    else:
        def objective(v):
            native,_=_unpack(v,dist_name,p,policy)
            ld=np.asarray(module.logpdf(x,*native),float).reshape(-1)
            if not np.isfinite(ld).all(): raise FloatingPointError('Nonfinite density while evaluating stationarity.')
            return -float(w@ld)/mass
        g=approx_derivative(objective,u,method='3-point',bounds=np.asarray(limits).T).reshape(-1)
    if not np.isfinite(g).all():raise FloatingPointError('Nonfinite component stationarity score.')
    pg=_projected_gradient(u,g,limits)
    return {'projected_gradient':pg,'gradient_acceptable':bool(pg<=policy.gradient_tolerance),
            'stationarity_acceptable':bool(pg<=policy.gradient_tolerance),
            'stationarity_kind':'ordinary_projected_score','ordinary_location_gradient_exists':True,
            'score_diagnostics':score_metadata,'coordinate_recovery':recovery,'global_optimum_certified':False}
