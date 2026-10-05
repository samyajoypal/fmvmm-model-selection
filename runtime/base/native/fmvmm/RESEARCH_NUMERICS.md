# Experimental numerical policy for soft EM

This development branch adds an opt-in constrained fitting path to the existing
FMVMM soft-EM implementation. It is intended for controlled research comparisons.
The outer algorithm remains `BaseMixture._fit`; each M-step optimizes a weighted
component likelihood with the current posterior weights held fixed. It does not
replace EM with a whole-mixture optimizer.

The branch also corrects log-density arithmetic for Student-t, skew-t, skew-normal,
contaminated skew-normal, slash, GH and skew-Laplace families. These density changes
apply independently of the optional fitting policy. No finite Student-t degrees of
freedom are silently replaced by infinity; infinity is an explicit normal boundary.
The Student-CDF tail calculation handles both zero and subnormal probabilities.

## Public API

The constructor accepts `numerical_policy`, `candidate_initializations`, and
`strict_density`. An explicit numerical policy enables strict density evaluation.
Provide controlled initializations for research comparisons; the legacy automatic
initialization workflow has not received the same validation as this path.

```python
from dataclasses import asdict
import numpy as np
from fmvmm.mixtures.FMVMM import fmvmm
from fmvmm.numerics.em_policy import EMPolicy
from fmvmm.numerics.constrained_mstep import prepare_component

policy = EMPolicy()
names = ("mvn", "mvt")
# X has two columns and has been centered/whitened using training data only.
starts = [
    (np.array([-1., 0.]), np.eye(2)),
    (np.array([ 1., 0.]), np.eye(2), np.array([5.])),
]
alpha = [prepare_component(name, start, policy)[0]
         for name, start in zip(names, starts)]
model = fmvmm(
    n_clusters=2, list_of_dist=list(names),
    candidate_combinations=[names],
    candidate_initializations={names: (np.array([.5, .5]), alpha)},
    numerical_policy=asdict(policy), strict_density=True,
    em_type="soft", max_iter=1200, tol=1e-8, n_jobs=1,
    verbose=False, print_log_likelihood=False,
)
model.fit(X)
diagnostics = model.em_diagnostics[0]
qualified = (diagnostics["converged"]
             and diagnostics["final_stationarity_acceptable"]
             and not diagnostics["hit_cap"])
```

Qualification is a local numerical check. It is not a certificate of the global
likelihood maximum. Inspect `candidate_failures` as well as successful candidates.
Returned native parameters can contain infinite degrees of freedom; use
`fmvmm.numerics.lossless_json` rather than replacing infinity with JSON null.

## Working domain

The policy declares fixed componentwise bounds, including a positive covariance
floor, bounded locations and shapes, and a positive mixture-weight floor. Data
whitening is the caller's responsibility. Initial values are explicitly projected
by `prepare_component`; density evaluation does not silently repair an invalid
scale. Rejected EM proposals preserve the previous feasible parameter vector.
Exact optimizer coordinates are retained across accepted EM updates, final
posterior checks and the next M-step. This avoids repeatedly subtracting the
covariance floor from a nearly singular native scale. When only native
parameters are supplied, feasible coordinate reconstruction requires an explicit
matrix backward-error check; GH tail roundoff is checked both in coordinates and
in decoded native parameters. Substantive domain violations still fail. Native
log densities are reconciled after encoding, so coordinate recovery cannot
silently change the fitted objective.

GH components use a canonical latent scale with E[W]=1 before applying bounds.
Skew-Laplace uses its fixed latent mean. Variance-gamma shapes exclude the region
with an infinite density at the component center. Finite slash degrees of freedom
are bounded explicitly, and the policy rejects configurations with
`slash_df_max + dimension/2 > 500`. These restrictions are part of the fitted
statistical model and must be reported, together with sensitivity analyses.
They do not establish equivalence to unrestricted FMVMM fitting.

## Optimization and diagnostics

Smooth solves with total budgets greater than1,000 iterations and unrestricted
restarts allow up to1,000 ordinary L-BFGS iterations initially. If stationarity
remains unresolved, they use a curvature attempt followed by ordinary segments
of at most100 iterations and further curvature attempts as needed. Every segment,
probe and rescue attempt shares the original total budget. The initial phase
avoids unnecessary curvature work in ordinary profiles; the shorter later
segments preserve several rescue opportunities for difficult smooth fits.
Budgets of at most1,000 iterations (including the study's60-step component updates)
and explicit cusp handoffs retain their existing schedule. Diagnostics distinguish
initial/later segments from a total resource cap. The model, box, likelihood
acceptance and stationarity threshold are unchanged.

Normal, Student-t, skew-normal and contaminated skew-normal weighted component
scores are analytic. The skew-t score is analytic except for its positive inverse
degrees-of-freedom coordinate, which uses a bounded, refined numerical difference;
the exact zero boundary has an analytic derivative. GH-family spatial scores use
posterior GIG moments evaluated with stable logarithmic Bessel/integral ratios;
canonical tail coordinates use native-density differences. Skew-slash scores normally evaluate the density and all latent moments together
with a refinement-checked quadrature, including the exact log-tail-score
identity. The previous latent-ratio spatial score and refined log-tail
difference remain an explicitly reported fallback when that joint rule is
outside its supported numerical range. Symmetric slash uses its latent-moment
spatial score and refined log-tail difference. For a custom
skew-slash domain where its shifted moment would leave the native quadrature
domain, an explicitly named native finite-difference score is used without
changing the fitted domain. Every returned objective is reconciled with the
native density evaluator.

Each component update has one total `component_maxiter`/`component_maxfun`
optimization budget. Ordinary solves, restarts, location-fixed continuations,
anchor proposals and escape line searches share it; an anchor does not receive
a fresh budget. Optimizer value/gradient requests and native candidate proposals
are counted, while final native-validity and stationarity verification is
reported separately. An ordinary likelihood-tolerance stop with an unresolved
projected score restarts from the incumbent with explicit `restart_ftol=0` by
default. This tightens objective stopping; it does not loosen the gradient
criterion or change the domain. Every solve's tolerance, consumed budget and
stop reason remain visible. If smooth restarts make no progress while the score
remains unresolved, a bounded curvature fallback forms a score-difference
Hessian, regularizes its spectrum and tries a feasible damped Newton direction.
Every probe and line-search request shares the same evaluation budget; each
fallback attempt consumes one iteration. Exact floating-point objective ties
prefer the smaller appropriate stationarity residual. Cusp locations retain
their explicit treatment; the fallback does not invent an ordinary derivative
there. Parameter bounds, native acceptance and convergence tolerances stay
unchanged. This is a search-direction repair, not an optimality certificate.
The legacy `native_validation_and_proposal_calls` field is a mixed request
counter and must not be interpreted as the total native density evaluations.

A component may make a nondecreasing GEM update
without resolving its score in that update, but final EM qualification still
requires the final-membership stationarity checks.

Variance-gamma and skew-Laplace may have a location cusp at a positive-weight
observation. Nearby observations are proposed as anchors, but accepted only if
the native likelihood does not decrease. Location-fixed continuation optimizes
all remaining coordinates. For variance-gamma `0 < lambda-dimension/2 < 1/2`,
the exact anchor is a strict conditional location local maximum. At exactly
`lambda-dimension/2 = 1/2`, including skew-Laplace, the location is checked by
the norm-subgradient condition with the outward normal cone of the location
box. An unsatisfied norm condition gives a feasible ascent direction, checked
by native-likelihood backtracking. Above `1/2`, the ordinary location score is
used. No numerical equality band replaces these distinct cases.

The policy exposes `cusp_proposal_radius` (candidate generation only),
`cusp_max_anchors`, `cusp_max_rounds`, and `cusp_escape_maxsteps`. Anchor
rejections, continuation caps and unresolved conditions remain in diagnostics.
These numerical budgets do not alter the statistical family or parameter box.
The conditional cusp check is not a joint or global maximum certificate.

Small likelihood change alone is insufficient for convergence. The constrained
path also checks component stationarity using the final posterior and the
mixture-weight update gap. At a cusp, the condition is recomputed with these
new weights, including the combined mass of duplicate observations. The
backwards-compatible `gradient_acceptable` field refers to the explicit
`stationarity_kind`; `ordinary_location_gradient_exists` is false at a cusp.
There, the recorded residual combines the smooth-coordinate projected score
and the normalized norm-subgradient violation, rather than inventing a full
location gradient. Diagnostics retain iteration caps, rejected updates, active
bounds, final ordinary or generalized stationarity, and the EM likelihood trace. An
unqualified fit must not be silently treated as a resolved exhaustive comparator.

The initial defaults without `numerical_policy` retain the legacy fitting path.
The new compact domains, controlled starts and strict stopping checks are
experimental behavior and should be selected consistently for every method in a
comparison. Conditional screening identities remain applicable to the same
separable working domains. They do not make a Gaussian or Student pilot an oracle,
certify local profile maxima, or turn a heuristic screening cutoff into a proven
posterior-error bound.

## Tests

From this checkout, with the required scientific Python dependencies installed:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python -m unittest discover -s Tests/numerics -p 'test_*.py'
```

The suite includes independent density references, difficult tail and limiting
cases, constrained EM behavior, component-score comparisons, and transport of the
exact normal boundary. Passing finite tests is not a uniform numerical error
bound for every possible dataset and parameter value.
