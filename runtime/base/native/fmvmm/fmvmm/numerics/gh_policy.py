"""Identifiable GH-family coordinates for the declared constrained EM model.

This is a component parameter conversion, not a mixture-fitting backend.  The
latent GIG variable is normalized to mean one before any scale restriction is
applied.  Bounds below define a working model and are never silently substituted
into a density evaluation.  Initial projection, if requested by a caller, is a
separate optimization decision and must be recorded by that caller.
"""
import numpy as np
from scipy.special import kve

GH_FAMILIES = frozenset({'mghp', 'mgst', 'mvhb', 'mnig', 'mvvg'})
DEFAULTS = {
    'gh_lambda_bound': 10.,
    'gh_alpha_bar_min': .001,
    'gh_alpha_bar_max': 200.,
    'gh_gst_tail_min': .05,
    'gh_gst_tail_max': 50.,
    'gh_vg_center_margin': .05,
    'gh_vg_lambda_max': 50.,
}


def _value(policy, key):
    value = policy.get(key, DEFAULTS[key]) if isinstance(policy, dict) else getattr(policy, key, DEFAULTS[key])
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be positive and finite.')
    return value


def _scalar(value, name):
    value = np.asarray(value, float)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be one finite scalar.')
    return float(value.reshape(-1)[0])


def mean_gig(lmbda, chi, psi):
    """E[W] for admissible GIG parameters, requiring a finite mean.

    Scaled Bessel ratios cancel exp(-sqrt(chi*psi)).  No Bessel failure is
    replaced by a different parameter or moment.
    """
    lmbda, chi, psi = (_scalar(v, n) for v, n in zip((lmbda, chi, psi), ('lambda', 'chi', 'psi')))
    if chi < 0 or psi < 0 or chi + psi <= 0:
        raise ValueError('GIG chi and psi must be nonnegative and not both zero.')
    if chi == 0:
        if lmbda <= 0 or psi <= 0:
            raise ValueError('Gamma boundary requires lambda > 0 and psi > 0.')
        result = 2. * lmbda / psi
    elif psi == 0:
        if lmbda >= -1:
            raise ValueError('Mean-one inverse-gamma coordinates require lambda < -1.')
        result = -chi / (2. * (lmbda + 1.))
    else:
        alpha_bar = np.sqrt(chi) * np.sqrt(psi)
        denominator = kve(lmbda, alpha_bar)
        numerator = kve(lmbda + 1., alpha_bar)
        if not (np.isfinite(denominator) and denominator > 0 and np.isfinite(numerator) and numerator > 0):
            raise ValueError('GIG Bessel ratio is not numerically supported at these parameters.')
        log_mean = .5 * (np.log(chi) - np.log(psi)) + np.log(numerator) - np.log(denominator)
        result = np.exp(log_mean)
    if not np.isfinite(result) or result <= 0:
        raise ValueError('GIG mean is not finite and strictly positive.')
    return float(result)


def _to_full(name, alpha):
    if name == 'mghp':
        lmbda, chi, psi, mu, sigma, gamma = alpha
    elif name == 'mgst':
        lmbda, chi, mu, sigma, gamma = alpha
        psi = 0.
    elif name == 'mvvg':
        lmbda, psi, mu, sigma, gamma = alpha
        chi = 0.
    elif name in {'mvhb', 'mnig'}:
        chi, psi, mu, sigma, gamma = alpha
        lmbda = (len(mu) + 1.) / 2. if name == 'mvhb' else -.5
    else:
        raise ValueError(f'Unsupported GH family: {name}.')
    mu, sigma, gamma = np.asarray(mu, float), np.asarray(sigma, float), np.asarray(gamma, float)
    if mu.ndim != 1 or len(mu) == 0 or gamma.shape != mu.shape or sigma.shape != (len(mu), len(mu)):
        raise ValueError('Incompatible GH location, scale, or skewness shapes.')
    if not all(np.isfinite(v).all() for v in (mu, sigma, gamma)):
        raise ValueError('GH location, scale, and skewness must be finite.')
    if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-14) or np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError('GH scale must be symmetric positive definite.')
    return (_scalar(lmbda, 'lambda'), _scalar(chi, 'chi'), _scalar(psi, 'psi'), mu, sigma, gamma)


def _from_full(name, lmbda, chi, psi, mu, sigma, gamma):
    if name == 'mghp':
        return lmbda, chi, psi, mu, sigma, gamma
    if name == 'mgst':
        return lmbda, chi, mu, sigma, gamma
    if name == 'mvvg':
        return lmbda, psi, mu, sigma, gamma
    return chi, psi, mu, sigma, gamma


def canonicalize_gh(name, alpha):
    """Density-preserving conversion to E[W]=1, in the native family tuple.

    If c=E[W], W'=W/c has parameters (lambda, chi/c, c*psi), while
    Sigma'=c*Sigma and gamma'=c*gamma preserve X=mu+W*gamma+sqrt(W)*Z.
    """
    lmbda, chi, psi, mu, sigma, gamma = _to_full(name, alpha)
    c = mean_gig(lmbda, chi, psi)
    result = _from_full(name, lmbda, chi / c, psi * c, mu.copy(), sigma * c, gamma * c)
    # Explicitly reject representational overflow; never project the density.
    _to_full(name, result)
    return result


def gh_tail_bounds(name, p, policy=None):
    """Bounds in optimizer coordinates, excluding mu, Sigma and gamma."""
    if int(p) != p or p < 1:
        raise ValueError('p must be a positive integer.')
    if name == 'mghp':
        bound = _value(policy, 'gh_lambda_bound')
        result = [(-bound, bound), (np.log(_value(policy, 'gh_alpha_bar_min')), np.log(_value(policy, 'gh_alpha_bar_max')))]
    elif name in {'mvhb', 'mnig'}:
        result = [(np.log(_value(policy, 'gh_alpha_bar_min')), np.log(_value(policy, 'gh_alpha_bar_max')))]
    elif name == 'mgst':
        result = [(np.log(_value(policy, 'gh_gst_tail_min')), np.log(_value(policy, 'gh_gst_tail_max')))]
    elif name == 'mvvg':
        result = [(np.log(p / 2. + _value(policy, 'gh_vg_center_margin')), np.log(_value(policy, 'gh_vg_lambda_max')))]
    else:
        raise ValueError(f'Unsupported GH family: {name}.')
    if any(lower >= upper for lower, upper in result):
        raise ValueError(f'Invalid GH working-domain interval for {name}.')
    return result


def _check_tail(u, bounds):
    u = np.asarray(u, float).reshape(-1)
    if len(u) != len(bounds) or not np.isfinite(u).all():
        raise ValueError('Invalid GH optimizer tail vector.')
    if any(value < lower or value > upper for value, (lower, upper) in zip(u, bounds)):
        raise ValueError('GH optimizer tail parameters are outside the declared working domain.')
    return u


def pack_gh_tail(name, alpha, p=None, policy=None, *, check_bounds=True):
    """Encode canonical native parameters; use check_bounds=False only for starts.

    The return value is not clipped.  A caller projecting a starting vector must
    record that projection and evaluate its actual projected density.
    """
    lmbda, chi, psi, mu, _, _ = _to_full(name, alpha)
    if p is None:
        p = len(mu)
    if p != len(mu):
        raise ValueError('GH dimension does not match the supplied parameters.')
    if not np.isclose(mean_gig(lmbda, chi, psi), 1., rtol=1e-10, atol=1e-12):
        raise ValueError('Canonicalize GH parameters to E[W]=1 before encoding.')
    if name == 'mgst':
        u = np.array([np.log(-lmbda - 1.)])
    elif name == 'mvvg':
        u = np.array([np.log(lmbda)])
    else:
        if chi <= 0 or psi <= 0:
            raise ValueError('Interior GH/HB/NIG coordinates require chi > 0 and psi > 0.')
        log_alpha_bar = .5 * (np.log(chi) + np.log(psi))
        u = np.array([lmbda, log_alpha_bar]) if name == 'mghp' else np.array([log_alpha_bar])
    return _check_tail(u, gh_tail_bounds(name, p, policy)) if check_bounds else u


def unpack_gh_tail(name, u, p, policy=None):
    """Decode to the native scalar prefix before (mu, Sigma, gamma).

    GH returns (lambda, chi, psi); GST (lambda, chi); VG (lambda, psi);
    hyperbolic and NIG (chi, psi).
    """
    u = _check_tail(u, gh_tail_bounds(name, p, policy))
    if name == 'mgst':
        tail = float(np.exp(u[0]))
        return -1. - tail, 2. * tail
    if name == 'mvvg':
        lmbda = float(np.exp(u[0]))
        return lmbda, 2. * lmbda
    lmbda = float(u[0]) if name == 'mghp' else ((p + 1.) / 2. if name == 'mvhb' else -.5)
    alpha_bar = float(np.exp(u[-1]))
    ratio = mean_gig(lmbda, alpha_bar, alpha_bar)
    chi, psi = alpha_bar / ratio, alpha_bar * ratio
    return (lmbda, chi, psi) if name == 'mghp' else (chi, psi)


def gh_dimension(name, p):
    """Nominal identifiable component dimension used by the native BIC."""
    if name not in GH_FAMILIES or int(p) != p or p < 1:
        raise ValueError('Invalid GH family or dimension.')
    return (2 if name == 'mghp' else 1) + 2 * p + p * (p + 1) // 2
