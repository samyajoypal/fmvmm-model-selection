"""Budgeted local curvature rescue after an unresolved smooth optimizer stall.

This changes search directions, never the model, box or stationarity tolerance.
It requires an ordinary smooth score in every supplied coordinate; callers of
location-fixed problems may pass only those smooth free coordinates. Neither
positive-definite regularization nor a small residual certifies an optimum.
"""
import numpy as np


def curvature_rescue(evaluate, u, value, gradient, bounds, budget, projected):
    """Try one feasible damped-Newton step with every score request budgeted.

    ``evaluate`` is the solve wrapper that charges this shared Budget and tracks
    its best point. A rescue attempt consumes one iteration, even if its line
    search fails. Central/one-sided score differences use 8 sqrt(machine eps)
    times coordinate scale; the Hessian spectrum is floored at
    max(1e-8, 1e-12 max(abs(eigenvalues))). These are search-direction choices.
    Native objective/score acceptance is still the caller's responsibility.
    """
    u=np.asarray(u,float);g=np.asarray(gradient,float);limits=np.asarray(bounds,float)
    before_i,before_e=budget.iterations,budget.evaluations
    record={'method':'feasible_regularized_score_hessian','accepted':False,
            'base_objective':float(value),'base_projected_gradient':projected(u,g,limits),
            'relative_difference_step':float(8*np.sqrt(np.finfo(float).eps)),
            'line_search':[],'global_optimum_certified':False}
    def finish(reason):
        record.update(stop_reason=reason,iterations=budget.iterations-before_i,
                      evaluations=budget.evaluations-before_e,
                      remaining_iterations=budget.remaining_iterations,
                      remaining_evaluations=budget.remaining_evaluations)
        return record
    from .component_budget import BudgetExhausted
    try:
        budget.iterate()
        # The same active-set convention as the ordinary projected score.
        active=((u<=limits[:,0]+1e-9)&(g>0))|((u>=limits[:,1]-1e-9)&(g<0))
        free=np.flatnonzero(~active);record['free_coordinates']=free.tolist()
        if not len(free):return finish('no_free_coordinate')
        H=np.empty((len(free),len(free)));schemes=[]
        record['difference_schemes']=schemes
        record['completed_hessian_columns']=0
        for k,i in enumerate(free):
            h=record['relative_difference_step']*max(1.,abs(u[i]))
            left,right=u[i]-limits[i,0],limits[i,1]-u[i]
            d=np.zeros(len(u))
            if left>=h and right>=h:
                d[i]=h;gp=evaluate(u+d,phase="curvature_probe")[1];gm=evaluate(u-d,phase="curvature_probe")[1]
                H[:,k]=(gp[free]-gm[free])/(2*h);scheme='central'
            else:
                sign=1. if right>=left else -1.
                h=min(h,max(left,right)/4)
                if h<=0 or u[i]+sign*h==u[i]:return finish('no_representable_feasible_difference')
                d[i]=sign*h;g1=evaluate(u+d,phase="curvature_probe")[1];g2=evaluate(u+2*d,phase="curvature_probe")[1]
                H[:,k]=sign*(-3*g[free]+4*g1[free]-g2[free])/(2*h);scheme='forward' if sign>0 else 'backward'
            schemes.append({'index':int(i),'step':float(h),'scheme':scheme})
            record['completed_hessian_columns']=k+1
        record['difference_schemes']=schemes
        if not np.isfinite(H).all():raise FloatingPointError('Nonfinite score Hessian in smooth rescue')
        record['relative_hessian_asymmetry']=float(np.linalg.norm(H-H.T)/max(np.linalg.norm(H),np.finfo(float).tiny))
        H=(H+H.T)/2;eigenvalues,V=np.linalg.eigh(H)
        floor=max(1e-8,1e-12*float(np.max(np.abs(eigenvalues))))
        record.update(hessian_eigenvalues=eigenvalues.tolist(),eigenvalue_floor=floor)
        # If coupling would point out of the box at a bound, remove that coordinate
        # and solve the reduced system again; do not clip a proposed parameter.
        retained=np.ones(len(free),bool)
        for _ in range(len(free)+1):
            indices=np.flatnonzero(retained)
            if not len(indices):return finish('no_feasible_newton_direction')
            A=H[np.ix_(indices,indices)];ev,vec=np.linalg.eigh(A)
            s=-vec@((vec.T@g[free[indices]])/np.maximum(ev,floor))
            step=np.zeros(len(u));step[free[indices]]=s
            blocked=((u<=limits[:,0]+1e-9)&(step<0))|((u>=limits[:,1]-1e-9)&(step>0))
            remove=np.flatnonzero(blocked[free]&retained)
            if not len(remove):break
            retained[remove]=False
        slope=float(g@step);record['directional_slope']=slope
        if not np.isfinite(slope) or slope>=0:return finish('no_descent_direction')
        scale=1.
        for i in np.flatnonzero(step):
            distance=(limits[i,1]-u[i])/step[i] if step[i]>0 else (limits[i,0]-u[i])/step[i]
            if distance<scale:scale=max(0.,float(distance))*(1-4*np.finfo(float).eps)
        record['max_feasible_scale']=scale
        if scale<=0:return finish('no_positive_feasible_step')
        for half in range(40):
            t=scale*2.**(-half);z=u+t*step
            if np.array_equal(z,u):return finish('no_representable_descent_step')
            if np.any(z<limits[:,0]) or np.any(z>limits[:,1]):
                record['line_search'].append({'step':t,'feasible':False});continue
            f,gg=evaluate(z,phase="curvature_line_search");residual=projected(z,gg,limits)
            # An exact objective tie may retain a demonstrably smaller score.
            # No positive objective loss, including one ulp, is ever allowed.
            accepted=(f<value and f<=value+1e-4*t*slope) or (f==value and residual<record['base_projected_gradient'])
            record['line_search'].append({'step':t,'feasible':True,'objective':float(f),
                                         'projected_gradient':residual,'accepted':bool(accepted)})
            if accepted:
                record.update(accepted=True,objective=float(f),projected_gradient=residual,
                              max_coordinate_change=float(np.max(np.abs(z-u))))
                return finish('feasible_native_objective_descent' if f<value else 'exact_objective_tie_with_smaller_score')
        return finish('line_search_found_no_acceptable_step')
    except BudgetExhausted as error:
        return finish(str(error))
