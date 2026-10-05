"""Declared parameter domain for opt-in, constrained soft EM.

Coordinates are supplied by the caller. In the model-selection study these are
training-whitened coordinates; this module never rescales the observations.
"""
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np

@dataclass(frozen=True)
class EMPolicy:
    covariance_floor: float = 1e-4
    cholesky_min: float = 1e-5
    cholesky_max: float = 20.
    location_bound: float = 30.
    skew_coefficient_bound: float = 50.
    inverse_df_max: float = 5.
    minimum_weight: float = 1e-4
    contamination_min: float = 1e-6
    inverse_inflation_min: float = 1e-4
    slash_df_min: float = .001
    slash_df_max: float = 450.
    gh_lambda_bound: float = 10.
    gh_alpha_bar_min: float = .001
    gh_alpha_bar_max: float = 200.
    gh_gst_tail_min: float = .05
    gh_gst_tail_max: float = 50.
    gh_vg_center_margin: float = .05
    gh_vg_lambda_max: float = 50.
    cusp_proposal_radius: float = 1e-5
    cusp_max_anchors: int = 2
    cusp_max_rounds: int = 3
    cusp_escape_maxsteps: int = 40
    component_maxiter: int = 100
    component_maxfun: int = 10000
    restart_ftol: float = 0.
    ftol: float = 1e-11
    gtol: float = 1e-6
    gradient_tolerance: float = 1e-4
    accept_tolerance: float = 1e-8


def make_policy(value):
    if isinstance(value, EMPolicy):
        result = value
    elif is_dataclass(value):
        result = EMPolicy(**asdict(value))
    elif isinstance(value, dict):
        result = EMPolicy(**value)
    else:
        raise TypeError('numerical_policy must be an EMPolicy or dictionary.')
    for key, val in asdict(result).items():
        if not np.isfinite(val) or (val < 0 if key=='restart_ftol' else val <= 0):
            raise ValueError(f'numerical_policy.{key} must be finite and '+('nonnegative.' if key=='restart_ftol' else 'positive.'))
    for key in ('component_maxiter', 'component_maxfun', 'cusp_max_anchors', 'cusp_max_rounds', 'cusp_escape_maxsteps'):
        if int(getattr(result,key)) != getattr(result,key):
            raise ValueError(f'numerical_policy.{key} must be an integer.')
    if result.restart_ftol > result.ftol:
        raise ValueError('restart_ftol cannot loosen the original objective tolerance.')
    if result.cholesky_min >= result.cholesky_max:
        raise ValueError('cholesky_min must be less than cholesky_max.')
    if result.contamination_min >= .5 or result.inverse_inflation_min >= 1:
        raise ValueError('Invalid contaminated-normal domain.')
    if result.slash_df_min >= result.slash_df_max:
        raise ValueError('Invalid slash degrees-of-freedom interval.')
    return result


def bounded_weights(counts, lower):
    """Exact maximizer of sum counts[j]*log(pi[j]), pi[j]>=lower."""
    counts=np.asarray(counts,float)
    if counts.ndim!=1 or not np.isfinite(counts).all() or np.any(counts<0) or counts.sum()<=0:
        raise ValueError('Invalid component counts.')
    if len(counts)*lower>=1:
        raise ValueError('Infeasible minimum mixture weight.')
    free=np.ones(len(counts),bool)
    answer=np.full(len(counts),lower)
    while free.any():
        remaining=1-lower*np.sum(~free)
        proposed=counts[free]/counts[free].sum()*remaining
        small=proposed<lower
        if not small.any():
            answer[free]=proposed
            break
        idx=np.flatnonzero(free)
        free[idx[small]]=False
    return answer
