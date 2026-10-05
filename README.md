# FMVMM model-selection reproducibility materials

Code and data for **Posterior-guided family screening for flexible multivariate mixture models**, by Samyajoy Pal and Jochen Einbeck.

The method uses one initial membership matrix to score thirteen distribution families for each mixture component. It fits promising complete mixtures and selects by BIC. This repository contains the numerical code, study protocols, CSV datasets, saved results and analysis scripts used for the paper.

## Reproduce the figures and tables

Use Python **3.12**. From a fresh clone:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

The last command verifies the compressed result files and regenerates the paper's seven figures and numerical tables under `build/figures/` and `build/tables/`. It does not fit models or use the network. Published computational times are preserved as recorded experimental outcomes; looking up a saved fit is never counted as refitting it.

To verify the CSV data and their preprocessing independently:

```bash
python data/verify_data.py
```

The installation step requires package access. The installed environment can subsequently run the checks and analyses offline. See [the analysis guide](docs/ANALYSIS.txt) for the output-to-paper mapping and checks.

## Fit the experiments again

The quick reproduction above uses saved results. The following commands regenerate observations and refit models using the original numerical algorithms and fixed seeds. Refitting requires a POSIX system such as Linux, the pinned dependencies and sufficient computing time.

```bash
python scripts/refit.py check
python scripts/refit.py test
python scripts/refit.py smoke --output runs/smoke
```

The numerical smoke test checks the thirteen-family calculation and complete mixture fits against recorded expected values. Inspect the study cases without fitting:

```bash
python scripts/refit.py plan --scope all
```

Run the simulation and real-data studies separately:

```bash
python scripts/refit.py run --scope simulation --jobs 8 --output runs/simulation
python scripts/refit.py run --scope real_data --jobs 8 --output runs/real_data
```

`--jobs` is the number of parallel, single-threaded worker processes; choose it for the allocated resources. These are substantial computations, not quick demonstrations. To run one case, copy its exact task ID from `plan` and pass `--case TASK_ID`. Use a new output directory. Explicit `--resume` continues unattempted work with the same run identity and retains recorded numerical failures.

New runs construct their own benchmark results. They do not silently mix published fitted outcomes with newly fitted models. See [the refitting guide](docs/REFITTING.txt) for execution stages, outputs, numerical limits and source verification. The original manifest retains its task numbering; the wrapper selects only the 406 paper cases.

## Experiments included

- Seven main simulation settings, with fifty replications each: G3, H3, S3, GH3, S4, H4 and H5.
- Two additional H3 settings, with ten replications each: increased overlap and unequal mixing weights.
- Three real datasets, each analysed in full and with three training/test splits, at component counts one, two and three: 36 fitting cases.
- Five cutoffs: zero, log(n), 2 log(n), 4 log(n) and 8 log(n).
- Top-r families per component; top-M score-ranked combinations; matched-budget random search; beam search; greedy family updating; and constrained fixed-K FAB.

The cutoff study covers all 406 cases. The alternative-method study covers the 200 main three-component replications, twenty additional H3 replications and 36 real-data cases. Full and partial comparison searches remain distinguished. All reported settings include every replication, with failures, iteration limits and large BIC losses retained.

The comparisons and cutoff study use the original datasets. They are paired comparisons, not independent confirmation on new datasets. The primary cutoff remains 2 log(n).

## Which implementations are included?

**All comparison algorithms were implemented for this study.** They are included under `runtime/comparisons/`, together with tests. They are not official implementations distributed by the authors of structural EM or FAB. EMDL is not implemented.

The constrained FAB method follows Fujimaki and Morinaga (2012), uses its own objective and adapts the updates to fixed K, the thirteen FMVMM families and the stated parameter restrictions. The greedy algorithm is the specified BIC family-update procedure. [Implementation and attribution details](docs/IMPLEMENTATIONS.txt) identify every source file and the relevant limitations.

Complete mixture fitting uses the included FMVMM development snapshot, commit `c40dc363f4651e62681bea5420507cebc4bb3b16`, under `runtime/base/native/fmvmm/`. A current PyPI installation is not a substitute for this snapshot. Numerical kernels are preserved; portable orchestration, source checks and path defaults are documented separately.

## CSV datasets

The original feature data are available directly:

- [Old Faithful](data/raw/faithful.csv): 272 rows, two measurements.
- [Seeds](data/raw/seeds.csv): 210 rows, seven measurements and wheat-variety labels.
- [Wisconsin Diagnostic Breast Cancer](data/raw/wdbc.csv): 569 rows, thirty measurements and diagnosis labels.

Row identifiers and labels are **not model inputs**. Seeds and WDBC use three principal components fitted to training observations only. Faithful retains two variables. `data/model_space/` also provides the exact transformed training and test CSVs. The fitting code subsequently performs its common training-based whitening step.

[data/README.txt](data/README.txt) explains the columns, sources, licenses and transformations. `data/manifest.json` contains exact split indices, seeds, means, scales and PCA loadings. `data/verify_data.py` reconstructs the CSV values and preprocessing from the included original public data sources. No Wholesale dataset is included.

## Reproducibility and interpretation

The saved-result path reproduces the reported numerical summaries from their recorded inputs. A new fit can differ slightly across machines and numerical libraries; computational times are machine-dependent, and time limits can affect completion. The exact Python package versions, seeds, source inventory and acceptance criteria are therefore supplied.

The full search fits every family combination under the stated assignment and initialization rule. It does not certify a global likelihood optimum. Larger-component settings contain both full and partial searches. FAB uses a different selection objective; unfinished component-count searches are reported explicitly. These distinctions are preserved in the outputs rather than hidden by a single overall success rate.

This release contains scientific code and public data. It excludes private working notes, manuscript comments, cluster login details, credentials and unnecessary upstream example datasets. It does not contain the unpublished manuscript itself.

## Citation and licenses

Use [CITATION.cff](CITATION.cff) for these reproducibility materials and cite the associated manuscript when describing the screening method. The included FMVMM package retains its MIT license. Dataset sources retain their own terms, including CC BY 4.0 for Seeds and WDBC and the supplied R distribution notices for Faithful. A repository-wide license for the study-specific scripts has not yet been selected; see [the attribution notes](docs/IMPLEMENTATIONS.txt).
