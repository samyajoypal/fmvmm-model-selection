from __future__ import annotations

import numpy as np

from fmvmm.inference.lrt import lrt
from fmvmm.inference.results import BootstrapResult


def parametric_bootstrap_lrt(null_model, full_model, *, simulate_null,
                             fit_null, fit_full, B=199, random_state=None,
                             df=None) -> BootstrapResult:
    """
    Parametric bootstrap reference for an LRT.

    Parameters
    ----------
    null_model, full_model : fitted models on observed data
    simulate_null : callable(null_model, rng) -> data
        Generates one bootstrap dataset under H0.
    fit_null, fit_full : callable(data) -> fitted model
        Refit functions for each bootstrap sample.
    B : int
        Number of bootstrap samples.

    Notes
    -----
    This generic function avoids imposing one simulation API on all mixture
    classes. Paper-specific simulations can pass closures here.
    """
    rng = np.random.default_rng(random_state)
    observed = lrt(full_model, null_model, df=df)
    boot_stats = np.zeros(B, dtype=float)

    for b in range(B):
        data_b = simulate_null(null_model, rng)
        null_b = fit_null(data_b)
        full_b = fit_full(data_b)
        boot_stats[b] = lrt(full_b, null_b, df=df).statistic

    pvalue = (1.0 + np.sum(boot_stats >= observed.statistic)) / (B + 1.0)
    return BootstrapResult(
        statistic=observed.statistic,
        pvalue=float(pvalue),
        bootstrap_statistics=boot_stats,
        B=int(B),
    )
