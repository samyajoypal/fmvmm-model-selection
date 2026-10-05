"""Generic inference tools for fmvmm.

The public layer is model-agnostic and works through adapters. It provides
general Wald, score, LRT, and bootstrap helpers without hard-coding
paper-specific hypotheses.
"""

from .adapters import (
    DMMAdapter,
    FMVMMAdapter,
    GenericMixtureAdapter,
    as_adapter,
    component_theta_and_names,
)
from .bootstrap import parametric_bootstrap_lrt
from .constraints import (
    LinearConstraint,
    NonlinearConstraint,
    equal,
    fixed_value,
    linear_contrast,
)
from .lrt import lrt
from .parameters import Parameter, ParameterVector
from .results import BootstrapResult, LRTResult, ScoreTestResult, WaldTestResult
from .score import score_test
from .wald import wald_test

from .inference_dmm import (
    score_test_fixed,
    wald_test_alpha_equal,
    wald_test_eta_equal,
    dims_from_model,
    idx_eta,
    idx_alpha,
    build_test_indices_alpha,
    build_test_indices_eta,
)

__all__ = [
    "WaldTestResult",
    "ScoreTestResult",
    "LRTResult",
    "BootstrapResult",
    "Parameter",
    "ParameterVector",
    "LinearConstraint",
    "NonlinearConstraint",
    "FMVMMAdapter",
    "DMMAdapter",
    "GenericMixtureAdapter",
    "as_adapter",
    "component_theta_and_names",
    "wald_test",
    "score_test",
    "lrt",
    "parametric_bootstrap_lrt",
    "fixed_value",
    "equal",
    "linear_contrast",
    "score_test_fixed",
    "wald_test_alpha_equal",
    "wald_test_eta_equal",
    "dims_from_model",
    "idx_eta",
    "idx_alpha",
    "build_test_indices_alpha",
    "build_test_indices_eta",
]
