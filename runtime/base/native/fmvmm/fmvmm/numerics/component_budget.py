"""Restart smooth L-BFGS solves within one shared component budget.

The same Budget can be passed to full-coordinate and location-fixed solves, so
an anchor continuation does not obtain a fresh iteration/evaluation allowance.
Native reconciliation and final-new-q checks belong to the calling EM routine.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


class BudgetExhausted(BaseException):
    pass


@dataclass
class Budget:
    max_iterations:int
    max_evaluations:int
    iterations:int=0
    evaluations:int=0

    @property
    def remaining_iterations(self):return max(0,self.max_iterations-self.iterations)
    @property
    def remaining_evaluations(self):return max(0,self.max_evaluations-self.evaluations)
    def evaluate(self):
        if self.remaining_evaluations<=0:raise BudgetExhausted('shared_evaluation_budget')
        self.evaluations+=1
    def iterate(self):
        if self.remaining_iterations<=0:raise BudgetExhausted('shared_iteration_budget')
        self.iterations+=1


def projected(u,g,bounds):
    limits=np.asarray(bounds);g=np.array(g,copy=True)
    g[(u<=limits[:,0]+1e-9)&(g>0)]=0
    g[(u>=limits[:,1]-1e-9)&(g<0)]=0
    return float(np.max(np.abs(g)))


def solve(objective,u,bounds,policy,budget,*,restart_ftol=None,initial_ftol=None,max_restarts=None):
    """Return best scored point; only inadequate-stationarity stops restart.

    restart_ftol=None preserves the original objective tolerance on every solve.
    An explicit restart_ftol=0 experiment tests stricter objective stopping
    without changing the domain, score tolerance or shared resource budgets.
    Large smooth budgets first allow at most 1000 ordinary iterations. If the
    score remains unresolved, the rest alternates at most 100 ordinary iterations
    with one existing curvature attempt. The same total Budget is shared.
    Budgets of at most 1000 iterations and explicit max_restarts control
    (including cusp handoff) keep their ordinary schedule and stall rescue.
    Every ordinary stop, scheduling boundary and total-budget cap is recorded.
    """
    u=np.asarray(u,float).copy();records=[];rescues=[];best=[np.inf,u.copy(),None];best_source=[None]
    def evaluate(z,*,phase="optimizer"):
        budget.evaluate();value,gradient=objective(z)
        gradient=np.asarray(gradient,float)
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite objective/score')
        if (value<best[0] or best[2] is None or
                (value==best[0] and projected(z,gradient,bounds)<projected(best[1],best[2],bounds))):
            best[:]=[float(value),np.array(z,copy=True),gradient.copy()]
            best_source[0]=phase
        return float(value),gradient
    reason='shared_budget_exhausted';round_number=0
    scheduled_curvature=bool(budget.max_iterations>1000 and max_restarts is None)
    while budget.remaining_iterations and budget.remaining_evaluations:
        before_u=best[1].copy();before_value=best[0]
        iterations_before,evaluations_before=budget.iterations,budget.evaluations
        first_ftol=policy.ftol if initial_ftol is None else initial_ftol
        active_ftol=first_ftol if round_number==0 or restart_ftol is None else restart_ftol
        result=None;interrupted=None
        ordinary_limit=min(1000 if round_number==0 else 100,budget.remaining_iterations) if scheduled_curvature else budget.remaining_iterations
        try:
            result=minimize(evaluate,best[1],jac=True,method='L-BFGS-B',bounds=bounds,
                callback=lambda _:budget.iterate(),
                options={'maxiter':ordinary_limit,'maxfun':budget.remaining_evaluations,
                         'ftol':active_ftol,'gtol':policy.gtol,'maxls':40,'maxcor':20})
        except BudgetExhausted as error:interrupted=str(error)
        residual=np.inf if best[2] is None else projected(best[1],best[2],bounds)
        record={'round':round_number,'ftol':active_ftol,'iterations':budget.iterations-iterations_before,
                'evaluations':budget.evaluations-evaluations_before,'success':bool(result.success) if result is not None else False,
                'message':str(result.message) if result is not None else interrupted,
                'best_objective':best[0],'projected_gradient':residual,
                'remaining_iterations':budget.remaining_iterations,'remaining_evaluations':budget.remaining_evaluations}
        record['ordinary_phase']=('initial_ordinary' if round_number==0 else 'hybrid_ordinary') if scheduled_curvature else 'ordinary'
        record['ordinary_iteration_limit']=ordinary_limit
        record['total_iteration_limit']=budget.max_iterations
        record['stop_scope']=('shared_budget_cap' if not budget.remaining_iterations or not budget.remaining_evaluations else
                              'ordinary_chunk_iteration_limit' if scheduled_curvature and record['iterations']>=ordinary_limit else
                              'ordinary_optimizer_stop')
        records.append(record)
        if residual<=policy.gradient_tolerance:reason='stationarity_tolerance';break
        if interrupted or not budget.remaining_iterations or not budget.remaining_evaluations:break
        if max_restarts is not None and round_number>=max_restarts:
            reason='return_control_for_conditional_update';break
        unchanged=np.array_equal(before_u,best[1]) and best[0]>=before_value
        if scheduled_curvature or (unchanged and (restart_ftol is None or active_ftol==restart_ftol)):
            from .smooth_rescue import curvature_rescue
            before_i,before_e=budget.iterations,budget.evaluations
            rescue_before_u=best[1].copy()
            try:
                rescue=curvature_rescue(evaluate,best[1].copy(),best[0],best[2].copy(),bounds,budget,projected)
            except BudgetExhausted as error:
                rescue={'method':'feasible_regularized_score_hessian','accepted':False,
                        'stop_reason':str(error),'iterations':budget.iterations-before_i,
                        'evaluations':budget.evaluations-before_e,
                        'remaining_iterations':budget.remaining_iterations,
                        'remaining_evaluations':budget.remaining_evaluations,'global_optimum_certified':False}
            rescue['trigger']=('initial_ordinary_stop' if round_number==0 else 'scheduled_ordinary_stop') if scheduled_curvature else 'unchanged_unresolved_stop'
            rescue['returned_best_point_source']=best_source[0]
            rescue['best_point_changed']=not np.array_equal(rescue_before_u,best[1])
            rescue['best_point_objective']=float(best[0])
            rescue['best_point_projected_gradient']=projected(best[1],best[2],bounds)
            rescues.append(rescue)
            records.append({'round':round_number,'method':'curvature_rescue','ftol':None,
                'iterations':budget.iterations-before_i,'evaluations':budget.evaluations-before_e,
                'success':bool(rescue['accepted']),'message':rescue['stop_reason'],
                'best_objective':best[0],'projected_gradient':projected(best[1],best[2],bounds),
                'remaining_iterations':budget.remaining_iterations,'remaining_evaluations':budget.remaining_evaluations,
                'curvature_rescue':rescue})
            if projected(best[1],best[2],bounds)<=policy.gradient_tolerance:
                reason='stationarity_tolerance';break
            progressed=not np.array_equal(before_u,best[1]) and best[0]<=before_value
            if not progressed or not budget.remaining_iterations or not budget.remaining_evaluations:
                reason='stalled_with_unresolved_stationarity' if budget.remaining_iterations and budget.remaining_evaluations else 'shared_budget_exhausted'
                break
        round_number+=1
    residual=np.inf if best[2] is None else projected(best[1],best[2],bounds)
    return {'u':best[1],'value':best[0],'gradient':best[2],'projected_gradient':residual,
            'qualified':bool(residual<=policy.gradient_tolerance),'stop_reason':reason,
            'total_iterations':budget.iterations,'total_evaluations':budget.evaluations,
            'iteration_limit':budget.max_iterations,'evaluation_limit':budget.max_evaluations,
            'restarts':round_number,'initial_ordinary_iteration_limit':1000 if scheduled_curvature else None,'ordinary_chunk_iterations':100 if scheduled_curvature else None,'runs':records,'curvature_rescues':rescues,'best_point_source':best_source[0],'global_optimum_certified':False,
            'budget_scope':'All solves sharing this Budget, including location-fixed continuations; each value/gradient request counts once.'}
