# FMVMM Test and Example Scripts

This folder contains lightweight executable examples as well as older notebook-style
test scripts. Run them from the repository root with `PYTHONPATH=.`.

## Fast Examples

These scripts are intended to be small enough for routine development checks:

```bash
PYTHONPATH=. python Tests/fmvmm_info_tests.py
PYTHONPATH=. python Tests/inference_tests.py
PYTHONPATH=. python Tests/dmm_inference_example.py
```

- `fmvmm_info_tests.py` fits a two-component FMVMM with soft EM and checks the
  Louis information decomposition and user/internal parameterizations.
- `inference_tests.py` fits FMVMM models and exercises the generic Wald, score,
  and likelihood-ratio interfaces.
- `dmm_inference_example.py` fits a soft Dirichlet mixture and runs the generic
  inference adapter on an identical-mixture model.

## Larger Examples

The remaining scripts are closer to worked examples converted from notebooks.
They may take longer because they fit larger data sets, more components, or more
candidate distribution families:

```bash
PYTHONPATH=. python Tests/dmm_tests.py
PYTHONPATH=. python Tests/fmvmm_tests.py
PYTHONPATH=. python Tests/mixmgh_tests.py
PYTHONPATH=. python Tests/mixsmsn_tests.py
PYTHONPATH=. python Tests/mix_skew_laplace_tests.py
```

