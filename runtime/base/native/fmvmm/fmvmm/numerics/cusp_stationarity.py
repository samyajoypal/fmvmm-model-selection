"""VG and skew-Laplace conditional location criteria and monotone anchor proposals.

A data anchor is an explicitly evaluated candidate. Proximity only generates
candidates; exact equality and the derived cusp/subgradient conditions classify
stationarity. No ordinary location derivative is assigned at a cusp.
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from fmvmm.numerics.constrained_mstep import _pack, _unpack, _bounds, _projected_gradient
from .weighted_gh_score import values_gradient, _full


def _data(x, weights):
    x=np.asarray(x,float);w=np.asarray(weights,float)
    if x.ndim!=2 or w.shape!=(len(x),) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Invalid observation/weight dimensions or weights')
    positive=w>0
    if not positive.any() or not np.isfinite(x[positive]).all():
        raise ValueError('Need positive finite effective membership mass and data')
    return x[positive],w[positive],np.flatnonzero(positive)


def _cone_distance(g,sigma,mu,location_bound):
    """Distance of g to the box normal cone in the Sigma metric.

    Enumerating active subsets gives the exact convex quadratic minimum up to
    floating-point linear algebra. Intended for the small p<=5 pilot setting.
    """
    if location_bound is None:return float(np.sqrt(g@sigma@g)),np.zeros_like(g)
    if np.any(np.abs(mu)>location_bound):raise ValueError('Location outside box')
    active=np.flatnonzero(np.abs(mu)==location_bound)
    if len(active)>12:raise ValueError('The normal-cone enumeration supports at most 12 active location coordinates')
    signs=np.sign(mu[active]);best=float(g@sigma@g);normal=np.zeros_like(g)
    for size in range(1,len(active)+1):
        for indices in itertools.combinations(range(len(active)),size):
            selected=active[list(indices)];sign=signs[list(indices)]
            z=np.linalg.solve(sigma[np.ix_(selected,selected)],(sigma@g)[selected])
            if np.any(z*sign<0):continue
            candidate=np.zeros_like(g);candidate[selected]=z;d=g-candidate
            value=float(d@sigma@d)
            if value<best:best=value;normal=candidate
    return float(np.sqrt(max(0.,best))),normal


def norm_cusp_location_diagnostic(x,weights,mu,sigma,gamma,*,psi=1.,location_bound=None):
    """Exact conditional location criterion for VG a=1/2 (including MVSL).

    Requires the density log f=constant+gamma' inv(Sigma)(x-mu)
    -sqrt(psi+gamma'inv(Sigma)gamma)*||inv(Sigma)^(1/2)(x-mu)||.
    The caller, not this function, must establish that the family has a=1/2.
    No proximity tolerance turns near observations into exact anchors.
    """
    x,w,_=_data(x,weights);mu=np.asarray(mu,float);sigma=np.asarray(sigma,float);gamma=np.asarray(gamma,float)
    if sigma.shape!=(len(mu),len(mu)) or gamma.shape!=mu.shape or not np.isfinite(psi) or psi<=0:
        raise ValueError('Invalid norm-cusp parameters')
    chol=np.linalg.cholesky(sigma);inv=np.linalg.solve(chol.T,np.linalg.solve(chol,np.eye(len(mu))))
    d=x-mu;center=np.all(d==0,axis=1);v=d@inv
    r=np.sqrt(np.maximum(0.,np.einsum('ij,ij->i',d,v)))
    if np.any((~center)&(r==0)):raise FloatingPointError('Nonzero residual has zero computed metric norm')
    rootB=np.sqrt(psi+gamma@inv@gamma)
    g=rootB*np.sum(w[~center,None]*v[~center]/r[~center,None],axis=0)-w.sum()*(inv@gamma)
    center_mass=float(w[center].sum());threshold=center_mass*rootB
    distance,normal=_cone_distance(g,sigma,mu,location_bound)
    return {'exact_anchor':bool(center.any()),'center_mass':center_mass,
            'dual_distance_to_box_normal_cone':distance,'norm_threshold':float(threshold),
            'condition_satisfied':bool(distance<=threshold),
            'signed_condition_margin':float(threshold-distance),
            'smooth_location_score':g.tolist(),'closest_box_normal':normal.tolist(),
            'criterion':'distance_Sigma(g, box_normal_cone) <= center_mass*sqrt(B)',
            'conditional_location_optimum':bool(distance<=threshold),
            'conditional_location_optimum_global':bool(distance<=threshold),
            'whole_parameter_optimum_certified':False}


def mvsl_location_diagnostic(x,weights,alpha,*,location_bound=None):
    """Native MVSL parameter tuple (mu,Sigma,gamma); psi=1 exactly."""
    mu,sigma,gamma=alpha
    return norm_cusp_location_diagnostic(x,weights,mu,sigma,gamma,psi=1.,location_bound=location_bound)



def location_classification(x, weights, u, name, policy):
    """Classify exact centers; recompute the condition for the supplied weights."""
    x,w,_=_data(x,weights);p=x.shape[1]
    alpha,_=_unpack(u,name,p,policy)
    lam,chi,psi,mu,sigma,gamma=_full(name,alpha,p)
    center=np.all(x==mu,axis=1)
    order=float(lam-p/2)
    base={'exact_anchor':bool(center.any()),'center_mass':float(w[center].sum()),
          'lambda_minus_half_dimension':order,'global_optimum_certified':False}
    if chi!=0 or not center.any() or order>.5:
        return dict(base,stationarity_kind='ordinary_projected_score',location_nonsmooth=False)
    if 0<order<.5:
        return dict(base,stationarity_kind='strict_conditional_location_cusp_and_smooth_score',
                    location_nonsmooth=True,location_condition_satisfied=True,
                    strict_conditional_location_local_maximum=True,
                    conditional_location_optimum_global=False,location_residual=0.,
                    radial_power=2*order)
    if order==.5:
        norm=norm_cusp_location_diagnostic(x,w,mu,sigma,gamma,psi=psi,location_bound=policy.location_bound)
        return dict(base,stationarity_kind='norm_subgradient_and_smooth_score',location_nonsmooth=True,
                    location_condition_satisfied=norm['condition_satisfied'],
                    location_residual=max(0.,norm['dual_distance_to_box_normal_cone']-norm['norm_threshold'])/w.sum(),
                    norm_subgradient=norm)
    raise FloatingPointError('Nonfinite center is outside the declared VG domain.')


def generalized_stationarity(x, weights, u, name, policy):
    """Ordinary score or a named nonsmooth criterion, never a fictitious gradient."""
    x,w,_=_data(x,weights);p=x.shape[1];limits=np.asarray(_bounds(name,p,policy))
    state=location_classification(x,w,u,name,policy)
    fixed=state['location_nonsmooth']
    ld,g=values_gradient(x,u,name,policy,w,fixed_location=fixed)
    from fmvmm.mixtures.FMVMM import dist_map
    alpha,_=_unpack(u,name,p,policy)
    native=np.asarray(dist_map[name].logpdf(x,*alpha),float).reshape(-1)
    if not np.isfinite(native).all() or not np.allclose(ld,native,rtol=2e-10,atol=2e-9):
        raise FloatingPointError('Cusp score objective disagrees with native density.')
    first=p if fixed else 0
    smooth=_projected_gradient(u[first:],-g[first:]/w.sum(),limits[first:])
    residual=max(smooth,state.get('location_residual',0.))
    state.update({'projected_gradient':residual,'smooth_projected_gradient':smooth,
                  'gradient_acceptable':bool(residual<=policy.gradient_tolerance),
                  'stationarity_acceptable':bool(residual<=policy.gradient_tolerance),
                  'ordinary_location_gradient_exists':not fixed,
                  'residual_meaning':'maximum of smooth projected score and normalized norm-subgradient violation' if fixed else 'ordinary projected score'})
    return state


def anchor_continuation(x, weights, u, name, policy, maxiter, native_value, *, budget=None):
    """Try nearby feasible anchors and continue only nonlocation coordinates.

    Returns the best monotone candidate, with all caps and rejected attempts.
    ``native_value`` is the caller's negative average native log likelihood.
    """
    from .component_budget import Budget,solve
    if budget is None:budget=Budget(int(maxiter),int(policy.component_maxfun))
    x,w,original=_data(x,weights);p=x.shape[1];mass=w.sum()
    limits=np.asarray(_bounds(name,p,policy));selected=np.array(u,copy=True)
    initial=native_value(selected);selected_value=initial
    radius=policy.cusp_proposal_radius*max(1.,np.max(np.abs(u[:p])))
    distances=np.linalg.norm(x-u[:p],axis=1);anchors=[];seen=[]
    for index in np.argsort(distances,kind='stable'):
        if distances[index]>radius:break
        if any(np.array_equal(x[index],v) for v in seen):continue
        seen.append(x[index]);anchors.append(int(index))
        if len(anchors)>=policy.cusp_max_anchors:break
    records=[]
    for index in anchors:
        if not budget.remaining_iterations or not budget.remaining_evaluations:break
        anchor=x[index].copy();start=selected.copy();start[:p]=anchor
        record={'observation_index':int(original[index]),'proposal_distance':float(distances[index]),
                'proposal_radius':float(radius),'radius_role':'candidate generation only',
                'anchor_mass':float(w[np.all(x==anchor,axis=1)].sum()),'hit_cap':False}
        records.append(record)
        if np.any(anchor<limits[:p,0]) or np.any(anchor>limits[:p,1]):
            record['status']='outside_location_domain';continue
        budget.evaluate()
        value=native_value(start);record['anchor_loglik_gain']=(selected_value-value)*mass
        if value>selected_value:
            record['status']='rejected_lower_native_likelihood';continue
        best=[value,start.copy()];calls=[0];best_residual=[None]
        def fixed_residual(z,gradient=None):
            if gradient is None:
                budget.evaluate()  # Charge uncached conditional tie-selection score.
                _,gradient=values_gradient(x,z,name,policy,w,fixed_location=True)
            return _projected_gradient(z[p:],-gradient[p:]/mass,limits[p:])
        def objective(v):
            z=np.r_[anchor,v];calls[0]+=1
            ld,g=values_gradient(x,z,name,policy,w,fixed_location=True)
            val=-float(w@ld)/mass
            if not np.isfinite(val) or not np.isfinite(g[p:]).all():
                raise FloatingPointError('Invalid fixed-location component score.')
            residual=fixed_residual(z,g)
            if val<best[0]:best[:]=[val,z.copy()];best_residual[0]=residual
            elif val==best[0]:
                if best_residual[0] is None:
                    best_residual[0]=residual if np.array_equal(z,best[1]) else fixed_residual(best[1])
                if residual<best_residual[0]:best[:]=[val,z.copy()];best_residual[0]=residual
            return val,-g[p:]/mass
        iterations_before,evaluations_before=budget.iterations,budget.evaluations
        result=solve(objective,start[p:],limits[p:],policy,budget,restart_ftol=policy.restart_ftol)
        candidate=best[1];final=native_value(candidate)
        if not np.isclose(final,best[0],rtol=2e-10,atol=2e-9):
            raise FloatingPointError('Fixed-location score objective disagrees with native density.')
        state=generalized_stationarity(x,w,candidate,name,policy)
        record.update({'iterations':budget.iterations-iterations_before,'evaluations':budget.evaluations-evaluations_before,
                       'optimizer_success':result['qualified'],'message':result['stop_reason'],'optimizer_runs':result['runs'],
                       'hit_cap':not budget.remaining_iterations or not budget.remaining_evaluations,
                       'stationarity':state,'continuation_loglik_gain':float((selected_value-final)*mass)})
        better_tie=False
        if final==selected_value and budget.remaining_evaluations:
            budget.evaluate()  # Charge the additional full generalized tie check.
            better_tie=state['projected_gradient']<generalized_stationarity(x,w,selected,name,policy)['projected_gradient']
            record['evaluations']=budget.evaluations-evaluations_before
            record['hit_cap']=not budget.remaining_iterations or not budget.remaining_evaluations
        if final<selected_value or better_tie:
            selected=candidate;selected_value=final;record['status']='accepted_native_nondecrease'
        elif final==selected_value:record['status']='equal_native_likelihood_without_better_stationarity'
        else:record['status']='rejected_lower_native_likelihood'
    return selected,records


def leave_failed_norm_anchor(x, weights, u, name, policy, native_value, *, budget=None):
    """Use the derived feasible ascent direction when a norm cusp is not optimal."""
    x,w,_=_data(x,weights);p=x.shape[1]
    state=location_classification(x,w,u,name,policy)
    record={'attempted':False,'accepted':False}
    if state['stationarity_kind']!='norm_subgradient_and_smooth_score':return u,record
    norm=state['norm_subgradient']
    if norm['condition_satisfied']:return u,record
    alpha,_=_unpack(u,name,p,policy);sigma=_full(name,alpha,p)[4]
    g=np.asarray(norm['smooth_location_score']);normal=np.asarray(norm['closest_box_normal'])
    direction=sigma@(g-normal)/norm['dual_distance_to_box_normal_cone']
    # Exact box feasibility; normal-cone projection makes active directions
    # inward/tangent up to linear-algebra roundoff, which is removed explicitly.
    at_upper=u[:p]>=policy.location_bound;at_lower=u[:p]<=-policy.location_bound
    direction[at_upper & (direction>0)]=0;direction[at_lower & (direction<0)]=0
    if not np.any(direction):return u,dict(record,attempted=True,status='no_feasible_ascent_direction')
    bounds=np.asarray(_bounds(name,p,policy));positive=direction>0;negative=direction<0
    feasible=np.r_[(bounds[:p,1][positive]-u[:p][positive])/direction[positive],
                   (bounds[:p,0][negative]-u[:p][negative])/direction[negative]]
    step=min(1.,float(np.min(feasible)))
    initial=native_value(u);record.update({'attempted':True,'direction':direction.tolist(),
                                        'theoretical_directional_gain':norm['dual_distance_to_box_normal_cone']-norm['norm_threshold']})
    for iteration in range(int(policy.cusp_escape_maxsteps)):
        candidate=u.copy();candidate[:p]=u[:p]+step*direction
        if np.any(candidate[:p]<bounds[:p,0]) or np.any(candidate[:p]>bounds[:p,1]):
            step*=.5;continue
        if budget is not None:budget.evaluate()
        value=native_value(candidate)
        if value<initial:
            record.update({'accepted':True,'step':float(step),'backtracks':iteration,
                           'native_loglik_gain':float((initial-value)*w.sum())})
            return candidate,record
        step*=.5
    return u,dict(record,status='line_search_unresolved',backtracks=policy.cusp_escape_maxsteps)
