Numerical regression tests
==========================

From the FMVMM repository root, run:

    python -m unittest discover -s Tests/numerics -p "test_*.py"

These checks compare densities with independent high-precision formulas and mixture integrals, test numerical boundaries, and check constrained component fitting. They do not establish global mixture optimization.
