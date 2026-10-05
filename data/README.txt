PUBLIC DATA USED IN THE ARTICLE

This directory contains exactly the three real datasets analysed in the paper:

  Old Faithful: 272 observations, 2 measurements, no external labels.
  Seeds: 210 observations, 7 measurements, wheat-variety labels (1, 2, 3).
  WDBC: 569 observations, 30 measurements, diagnosis labels (B or M).

The CSV files in raw/ contain the original numeric features, before any
standardization or dimension reduction. All source rows are retained in their
original order. Feature values and external labels reproduce the frozen study
data exactly. Numbers are written with 17 significant decimal digits so that
their float64 values survive a CSV round trip without loss.

COLUMNS

The model feature columns are explicitly listed in manifest.json. The columns
row_index, source_row_id and source_id are metadata, never model inputs.
row_index is zero-based. source_row_id is the one-based position in the original
source. The WDBC source_id column preserves the public identifier supplied in
the UCI data file; it is also excluded from fitting. wheat_variety and diagnosis
are external evaluation labels, not features. Labels do not determine the
splits, preprocessing, component count, family selection or fitted parameters.
The two Faithful variables are eruption duration and waiting time, in minutes.

ACTUAL PAPER SPLITS AND TRANSFORMATIONS

Each dataset has one full-data analysis and three random 80/20 holdouts. The
holdout seeds are:

  Old Faithful: 2026098300, 2026098301, 2026098302.
  Seeds:       2026093300, 2026093301, 2026093302.
  WDBC:        2026096300, 2026096301, 2026096302.

NumPy default_rng(seed).permutation(n) generates the permutation. The first
ceil(0.2*n) indices form the test set; the rest form the training set. Each
index vector is sorted before use. Training/test counts are 217/55 for Faithful,
168/42 for Seeds, and 455/114 for WDBC. This is unstratified splitting. Faithful
holdouts describe interpolation of the empirical distribution, not forecasting
of later eruptions.

For each split, training feature means and population standard deviations
(ddof=0) standardize both training and test observations. Seeds and WDBC use the
three leading principal components from the standardized training matrix.
These principal components are not whitened. The sign of each loading vector
is fixed by making its largest absolute loading positive. Faithful retains
both standardized measurements. The full-data transformations use all rows.

model_space/<dataset>/ contains the corresponding train and test CSV files.
The full_test.csv files contain headers only, because the full-data analyses
have no held-out observations. These files are inputs in the modeled space;
the FMVMM fitting code subsequently applies its common training-only whitening
step. They should not be mistaken for the final internal fitting coordinates.

manifest.json records every split index, mean, scale, loading matrix, singular
value, explained-variance fraction, array checksum and CSV checksum. The PCA
predictive densities refer to the three retained coordinates, not to the full
7- or 30-dimensional original feature space. case_manifest.json records all
36 real-data fitting cases: three datasets x four splits x K in {1, 2, 3}.

These are the paper's three-PC, three-holdout settings. An earlier development
data archive used different analysis settings; those settings are not used by
the CSV files here. No Wholesale observations are included.

PROVENANCE AND LICENSES

The frozen sources were obtained on 13 September 2026. CSV format conversion
and the model-space exports were made on 5 October 2026. The original sources
are retained, unmodified, in sources/. Their SHA-256 values are in manifest.json.

Seeds:
  Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., and Lukasik, S.
  (2010). Seeds. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5H30K
  https://archive.ics.uci.edu/dataset/236/seeds
  Download: https://archive.ics.uci.edu/static/public/236/seeds.zip

WDBC:
  Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993).
  Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
  https://doi.org/10.24432/C5DW2B
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
  Download:
  https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip

The official UCI pages identify both datasets as CC BY 4.0 (checked 5 October
2026). Copies, CSV conversions and derived standardized/PCA coordinates retain
this attribution. The license text is in licenses/CC-BY-4.0.txt.

Old Faithful:
  R Core Team and contributors, datasets::faithful, R 4.5.1. Original data
  source credited by R: W. Hardle, Smoothing Techniques with Implementation
  in S (1991).
  https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html
  Frozen source:
  https://svn.r-project.org/R/tags/R-4-5-1/src/library/datasets/data/faithful.R

R's datasets package identifies its license as part of R. The R 4.5.1 copyright
notice states that R is distributed under GPL version 2 or 3. The package
description is retained in sources/R-datasets-DESCRIPTION.in; the release
COPYING and COPYRIGHTS notices are in licenses/. The Faithful format conversion
and standardized exports are supplied with that source and those notices.
No separate CC license is claimed for Faithful.

These independently sourced data retain their own license terms, regardless
of the repository's software license. No license is asserted for third-party
material beyond the terms supplied by the original source.

VALIDATION

From the repository root, run:

  python data/verify_data.py

The verifier uses only the Python standard library and NumPy and accesses no
network. It verifies hashes, reconstructs raw feature and label values from
the original sources, verifies every split, reconstructs training-only
standardization and PCA, and checks all model-space CSVs. validation.json
records the initial export checks. SHA256SUMS covers the delivered files.
