"""Bounded coordinate recovery with explicit error checks.

These helpers never relax objective-domain guards. They only encode supplied
native tuples and return a feasible coordinate vector plus diagnostics. A
reconstructed tuple is still subject to the caller's native likelihood checks.
"""
import numpy as np
from fmvmm.numerics.gh_policy import pack_gh_tail,unpack_gh_tail,gh_tail_bounds

EPS=np.finfo(float).eps


def encode_gh_tail(name,alpha,p,policy):
    """Reconcile only machine-roundoff boundary error in native-tail encoding.

    Both the coordinate excess and the decoded native scalar parameters are
    checked. ``unpack_gh_tail`` retains its original strict domain validation.
    No scalar density parameter is modified in place.
    """
    raw=pack_gh_tail(name,alpha,p,policy,check_bounds=False)
    limits=np.asarray(gh_tail_bounds(name,p,policy),float)
    excess=np.maximum(limits[:,0]-raw,raw-limits[:,1])
    budget=16*EPS*np.maximum(1.,np.maximum(np.abs(limits[:,0]),np.abs(limits[:,1])))
    if np.any(excess>budget):raise ValueError('Native GH tail is substantively outside the declared coordinate domain.')
    candidate=np.clip(raw,limits[:,0],limits[:,1])
    decoded=np.asarray(unpack_gh_tail(name,candidate,p,policy),float)
    original=np.asarray(alpha[:len(decoded)],float)
    # Positive GIG scale coordinates are compared in logs; lambda is not.
    scale_start=1 if name in ('mghp','mgst','mvvg') else 0
    before=original.copy();after=decoded.copy()
    if np.any(before[scale_start:]<=0) or np.any(after[scale_start:]<=0):
        raise ValueError('Invalid positive GH scalar scale during encoding.')
    before[scale_start:]=np.log(before[scale_start:]);after[scale_start:]=np.log(after[scale_start:])
    native_budget=64*EPS*np.maximum(1.,np.maximum(abs(before),abs(after)))
    changed=np.flatnonzero(candidate!=raw)
    if len(changed) and np.any(abs(after-before)>native_budget):
        raise FloatingPointError('GH boundary encoding would change the native tail beyond its floating-point budget.')
    diagnostic=dict(encoding_reconciled=bool(len(changed)),coordinates=changed.tolist(),
        raw_coordinates=raw.tolist(),encoded_coordinates=candidate.tolist(),
        coordinate_error_budget=budget.tolist(),native_transformed_error=(after-before).tolist(),
        native_error_budget=native_budget.tolist(),scientific_bounds_changed=False)
    return candidate,diagnostic


def encode_native_scale(sigma,policy):
    """Recover a feasible factor only if it reproduces Sigma to roundoff.

    This is not an initialization projection onto a different covariance.
    A failed Cholesky or an excessive reconstruction discrepancy raises.
    The error envelope is relative to the supplied/reconstructed matrix scale,
    never a loose tolerance on poorly conditioned log-Cholesky coordinates.
    """
    sigma=np.asarray(sigma,float);p=len(sigma)
    if sigma.shape!=(p,p) or not p or not np.isfinite(sigma).all():raise ValueError('Invalid native scale.')
    matrix_scale=float(np.linalg.norm(sigma,np.inf))
    if matrix_scale<=0:raise ValueError('Native scale must be positive definite.')
    envelope=64*p*EPS*matrix_scale
    if np.linalg.norm(sigma-sigma.T,np.inf)>envelope:raise ValueError('Native scale is not symmetric to floating-point precision.')
    symmetric=(sigma+sigma.T)/2
    raw=np.linalg.cholesky(symmetric-policy.covariance_floor*np.eye(p))
    factor=raw.copy();diagonal=np.diag_indices(p);lower=np.tril_indices(p,-1)
    factor[diagonal]=np.clip(factor[diagonal],policy.cholesky_min,policy.cholesky_max)
    factor[lower]=np.clip(factor[lower],-policy.cholesky_max,policy.cholesky_max)
    reconstructed=factor@factor.T+policy.covariance_floor*np.eye(p)
    # Include the magnitude of the actual factor product in the error scale.
    product_scale=float(np.linalg.norm(abs(factor)@abs(factor).T+policy.covariance_floor*np.eye(p),np.inf))
    envelope=64*p*EPS*max(matrix_scale,product_scale)
    backward_error=float(np.linalg.norm(reconstructed-sigma,np.inf))
    if backward_error>envelope:
        raise ValueError('No backward-error-verified feasible Cholesky representation was recovered.')
    spatial=np.r_[np.log(np.diag(factor)),factor[lower]]
    changed=np.argwhere(factor!=raw)
    diagnostic=dict(encoding_reconciled=bool(len(changed)),changed_factor_entries=changed.tolist(),
        raw_diagonal=np.diag(raw).tolist(),encoded_diagonal=np.diag(factor).tolist(),
        backward_error_inf=backward_error,backward_error_budget=envelope,
        backward_error_relative=backward_error/matrix_scale,
        diagnostic_meaning='A feasible factor whose decoded covariance differs only within the explicit matrix-roundoff envelope.',
        scientific_bounds_changed=False)
    return spatial,factor,reconstructed,diagnostic


def verify_native_likelihood(module,x,weights,before,after,*,atol=2e-9,rtol=2e-10):
    """Use the existing density-agreement tolerance, without changing fit acceptance.

    The caller must still enforce native monotonicity and final stationarity
    on whichever tuple it actually returns. No fit is qualified here.
    """
    x=np.asarray(x,float);w=np.asarray(weights,float)
    if w.shape!=(len(x),) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid membership weights.')
    positive=w>0;x=x[positive];w=w[positive]
    old=np.asarray(module.logpdf(x,*before),float).reshape(-1)
    new=np.asarray(module.logpdf(x,*after),float).reshape(-1)
    if old.shape!=(len(x),) or new.shape!=old.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise FloatingPointError('Nonfinite native likelihood during coordinate verification.')
    if not np.allclose(new,old,atol=atol,rtol=rtol):
        raise FloatingPointError('Native density changed beyond the existing arithmetic agreement tolerance.')
    return dict(max_absolute_log_density_change=float(abs(new-old).max()),
        weighted_loglik_change=float(w@(new-old)),mean_loglik_change=float(w@(new-old)/w.sum()),
        atol=atol,rtol=rtol,fit_acceptance_unchanged=True,stationarity_verified=False)
