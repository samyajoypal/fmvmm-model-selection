# FMVMM: Flexible Multivariate Mixture Model

FMVMM is a Python package providing a comprehensive collection of finite mixture models for various multivariate distributions. It is designed to offer flexibility in modeling both identical and non-identical mixture distributions, allowing users to apply advanced clustering techniques efficiently.

## Features
- Implements finite mixtures of various multivariate distributions.
- Supports clustering using Dirichlet Mixture Models and mixtures of other non-identical distributions.
- Provides multiple model selection criteria such as AIC, BIC, and ICL.
- Uses soft EM by default for FMVMM, with hard EM still available through `em_type="hard"`.
- Provides information matrices and standard errors for soft EM and hard/classification EM fits.
- Provides a generic inference layer for Wald, score, likelihood-ratio, and bootstrap-based tests.
- Example Jupyter notebooks are available in `fmvmm/Examples/`; executable examples are available in `Tests/`.

## Mixture Models Included
- **Dirichlet Mixture Model** (`DMM_Soft`, `DMM_Hard`)
- **Mixtures of Multivariate Generalized Hyperbolic Distributions** (`MixMGH`)
- **Mixtures of Skew Normal Distributions** (`SkewNormalMix`)
- **Mixtures of Skew T Distributions** (`SkewTMix`)
- **Mixtures of T Distributions** (`TMix`)
- **Mixtures of Skew Normal Contaminated Distributions** (`SkewContMix`)
- **Mixtures of Skew Slash Distributions** (`SkewSlashMix`)
- **Mixtures of Slash Distributions** (`SlashMix`)
- **Flexible Multivariate Mixture Model** (`FMVMM`)
  - This model allows fitting all possible combinations of mixtures of different identical and non-identical distributions.

## Distributions Included
- Multivariate Skew Normal
- Multivariate Normal
- Multivariate T
- Multivariate Generalized Skew T
- Multivariate Skew T
- Multivariate Hyperbolic
- Multivariate Normal Inverse Gaussian
- Multivariate Variance Gamma
- Multivariate Skew Slash
- Multivariate Slash
- Multivariate Skew Normal Contaminated
- Multivariate Skew Laplace

## Installation
You can install FMVMM using:
```bash
pip install fmvmm
```

## Usage
Each mixture class in FMVMM follows the same basic workflow:

```python
from fmvmm.mixtures.DMM_Soft import DMM_Soft

model = DMM_Soft(n_clusters=3)
model.fit(data)  # Fit a mixture model
clusters = model.predict()  # Get cluster assignments
bic_value = model.bic()  # Compute Bayesian Information Criterion
aic_value = model.aic()  # Compute Akaike Information Criterion
icl_value = model.icl()  # Compute Integrated Complete Likelihood Criterion
pi, alpha = model.get_params()  # Get MLE of parameters
info_matrix, se = model.get_info_mat()  # Get information matrix and standard errors
```

### Flexible Multivariate Mixtures

`fmvmm` fits mixtures whose components may come from different distribution
families. Soft EM is the default; hard EM can be requested explicitly.

```python
import numpy as np

from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.mixtures.FMVMM import fmvmm

rng = np.random.default_rng(123)
x1 = mvn.rvs(
    np.array([0.0, 0.0]),
    np.array([[0.45, 0.08], [0.08, 0.35]]),
    size=40,
    random_state=rng,
)
x2 = mvsl.rvs(
    np.array([2.5, 2.0]),
    np.array([[0.55, -0.05], [-0.05, 0.45]]),
    np.array([0.7, -0.3]),
    size=40,
    random_state=rng,
)
data = np.vstack([x1, x2])

model = fmvmm(
    n_clusters=2,
    list_of_dist=["mvn", "mvsl"],
    specific_comb=True,
    assignment_permutations=True,
)
model.fit(data)

print(model.best_mixture())
print(model.best_bic())

# Default: user-facing scale, all pi values followed by component parameters.
info_mats, ses = model.get_info_mat()

# Explicit alternatives useful for inference and method comparison.
soft_info, soft_se, soft_details = model.get_info_mat_soft(return_details=True)
hard_model = fmvmm(
    n_clusters=2,
    list_of_dist=["mvn", "mvsl"],
    specific_comb=True,
    assignment_permutations=True,
    em_type="hard",
)
hard_model.fit(data)
hard_info, hard_se, hard_details = hard_model.get_info_mat_hard(return_details=True)
```

### Likelihood-Based Inference

Generic Wald, score, likelihood-ratio, and bootstrap tests are available through
`fmvmm.inference`. Constraints are expressed using parameter names exposed by
the fitted model adapter.

```python
from fmvmm.inference import as_adapter, fixed_value, wald_test, score_test, lrt

adapter = as_adapter(model)
params = adapter.parameter_vector(parameterization="user")

# Example: H0: pi_1 = 0.5
H = fixed_value(params, "pi[0]", 0.5)
wald_result = wald_test(model, H, parameterization="user")

# Score and likelihood-ratio tests require a fitted null model.
null_model = fmvmm(
    n_clusters=2,
    list_of_dist=["mvn", "mvsl"],
    specific_comb=True,
    assignment_permutations=True,
    fixed_pi=[0.5, 0.5],
)
null_model.fit(data)
score_result = score_test(null_model, ["eta[0]"], parameterization="internal")

# LRT is generic; use bootstrap references for non-regular mixture hypotheses.
lrt_result = lrt(model, null_model, df=1)
```

For runnable examples and smoke tests:

```bash
PYTHONPATH=. python Tests/fmvmm_info_tests.py
PYTHONPATH=. python Tests/inference_tests.py
PYTHONPATH=. python Tests/dmm_inference_example.py
```

For more detailed examples, see the Jupyter notebooks in `fmvmm/Examples/`.

## Citation
If you use FMVMM in your research, please cite the relevant papers:

1. Pal, Samyajoy, and Christian Heumann. "Clustering compositional data using Dirichlet mixture model." *PLoS ONE* 17, no. 5 (2022): e0268438. [https://doi.org/10.1371/journal.pone.0268438](https://doi.org/10.1371/journal.pone.0268438).
2. Pal, Samyajoy, and Christian Heumann. "Gene coexpression analysis with Dirichlet mixture model: accelerating model evaluation through closed-form KL divergence approximation using variational techniques." *International Workshop on Statistical Modelling* (2024). [https://doi.org/10.1007/978-3-031-65723-8_21](https://doi.org/10.1007/978-3-031-65723-8_21).
3. Pal, Samyajoy, and Christian Heumann. "Revisiting Dirichlet Mixture Model: Unraveling Deeper Insights and Practical Applications." *Statistical Papers* 66, no. 1 (2025): 1-38. [https://doi.org/10.1007/s00362-024-01627-0](https://doi.org/10.1007/s00362-024-01627-0).
4. Pal, Samyajoy, and Christian Heumann. "Flexible Multivariate Mixture Models: A Comprehensive Approach for Modeling Mixtures of Non-Identical Distributions." *International Statistical Review* (2024). [https://doi.org/10.1111/insr.12593](https://doi.org/10.1111/insr.12593).

## License
FMVMM is open-source and distributed under the MIT License.

## Contact
For any questions or contributions, feel free to reach out or open an issue on GitHub.
