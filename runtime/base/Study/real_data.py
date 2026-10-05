#!/usr/bin/env python3
"""Pinned public data and label-blind preprocessing for the FMVMM study.

Ordinary loading and verification never use the network. Run ``prepare`` to
rebuild numeric caches from the bundled sources; use ``prepare --download``
only on a login/local machine if a source file is missing. No R installation
or reference-FMVMM import is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
DATASET_IDS = ("iris", "wine", "seeds", "vertebral", "glass", "wdbc", "wholesale", "faithful")
SPLIT_SEEDS = {name: [2026091300 + 1000 * i + r for r in range(5)]
               for i, name in enumerate(DATASET_IDS)}
K_GRID = (1, 2, 3, 4)
PCA_MAX_DIM = 5
ACCESS_DATE = "2026-09-13"

_UCI = "https://archive.ics.uci.edu/"
_R = "https://svn.r-project.org/R/tags/R-4-5-1/"
_CELL = ("radius", "texture", "perimeter", "area", "smoothness", "compactness",
         "concavity", "concave_points", "symmetry", "fractal_dimension")
SPECS = {
    "iris": dict(n=150, p=4, uci_id=53, slug="iris", member="iris.data",
                 sha256="d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9",
                 feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                 label_name="species", doi="10.24432/C56C76",
                 citation="Fisher, R. A. (1936). Iris. UCI Machine Learning Repository.",
                 notes="Original iris.data release retained, including documented measurement discrepancies; bezdekIris.data is not substituted."),
    "wine": dict(n=178, p=13, uci_id=109, slug="wine", member="wine.data",
                 sha256="2bae62c4481220623579d4c4fb36b55652b6b75e06e49fa1981b8198362dfdab",
                 feature_names=["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline"],
                 label_name="cultivar", doi="10.24432/C5PC7J",
                 citation="Aeberhard, S., and Forina, M. (1992). Wine. UCI Machine Learning Repository."),
    "seeds": dict(n=210, p=7, uci_id=236, slug="seeds", member="seeds_dataset.txt",
                  sha256="5033bd71b24592fa70d9b3cc5be02332dc6d03d0b27e6655e6d553f39afab323",
                  feature_names=["area", "perimeter", "compactness", "kernel_length", "kernel_width", "asymmetry", "kernel_groove_length"],
                  label_name="wheat_variety", doi="10.24432/C5H30K",
                  citation="Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., and Lukasik, S. (2010). Seeds. UCI Machine Learning Repository."),
    "vertebral": dict(n=310, p=6, uci_id=212, slug="vertebral+column", member="column_3C.dat",
                      sha256="899ed1d137ceb4c36d59fa35f3cbab39ec3b78fd3ca1fdaf2fae155459c1372f",
                      feature_names=["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"],
                      label_name="clinical_category", doi="10.24432/C5K89B",
                      citation="Barreto, G., and Neto, A. (2005). Vertebral Column. UCI Machine Learning Repository.",
                      notes="Only the three-category file is used. The two-category file contains the same observations. PCA also removes the near-redundancy among pelvic incidence, pelvic tilt and sacral slope."),
    "glass": dict(n=214, p=9, uci_id=42, slug="glass+identification", member="glass.data",
                  sha256="499c07a17115bf63100bb19cc1fdefad7f464bf5dc2b168930fa2071fd6b6e65",
                  feature_names=["refractive_index", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"],
                  label_name="glass_type", doi="10.24432/C5WW2P",
                  citation="German, B. (1987). Glass Identification. UCI Machine Learning Repository.",
                  excluded_columns=["id"], notes="Six observed category codes are external labels; they do not determine the K grid."),
    "wdbc": dict(n=569, p=30, uci_id=17, slug="breast+cancer+wisconsin+diagnostic", member="wdbc.data",
                 sha256="bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
                 feature_names=[f"{name}_{summary}" for summary in ("mean", "se", "worst") for name in _CELL],
                 label_name="diagnosis", doi="10.24432/C5DW2B",
                 citation="Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.",
                 excluded_columns=["id"], notes="Thirty continuous nuclear-image features; this is WDBC, not the older Wisconsin breast-cancer data."),
    "wholesale": dict(n=440, p=6, uci_id=292, slug="wholesale+customers", member="Wholesale customers data.csv",
                      sha256="647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8",
                      feature_names=["fresh", "milk", "grocery", "frozen", "detergents_paper", "delicassen"],
                      label_name="channel", doi="10.24432/C5030X",
                      citation="Cardoso, M. (2013). Wholesale Customers. UCI Machine Learning Repository.",
                      transform="log1p", excluded_columns=["channel", "region"],
                      notes="Only six spending measures enter X. Channel is an external descriptive label; Region is preserved as metadata. A fixed log1p transform precedes training-only standardization/PCA."),
    "faithful": dict(n=272, p=2, member=None,
                     sha256="3cdac44bca411deabae2904557fce2272e87c83128087d1927a5384fb8c99e4f",
                     feature_names=["eruption_minutes", "waiting_minutes"], label_name=None,
                     source_url=_R + "src/library/datasets/data/faithful.R",
                     documentation_url="https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html",
                     citation="R Core Team and contributors (R 4.5.1), datasets::faithful; source W. Härdle, Smoothing Techniques with Implementation in S (1991).",
                     notes="Unlabelled Old Faithful eruption/waiting measurements, as bundled in R 4.5.1. Rounded values are retained. Random holdouts assess interpolation of the empirical density, not temporal forecasting or independent future eruptions."),
}

AUXILIARY_SOURCES = [
    dict(path="licenses/CC-BY-4.0.txt", url="https://creativecommons.org/licenses/by/4.0/legalcode.txt", sha256="9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"),
    dict(path="licenses/R-4.5.1-COPYING", url=_R + "COPYING", sha256="e6d6a009505e345fe949e1310334fcb0747f28dae2856759de102ab66b722cb4"),
    dict(path="licenses/R-4.5.1-COPYRIGHTS", url=_R + "doc/COPYRIGHTS", sha256="93a225906ee6514d1482548f8945e41a5cfcbae70610f9577ac7c9a512e060f0"),
    dict(path="raw/R-datasets-DESCRIPTION.in", url=_R + "src/library/datasets/DESCRIPTION.in", sha256="4ce77b76b57ed08343b62e548eb95cbcfcd77d111881e6262b711a56b973f6a4"),
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _raw_path(dataset_id):
    return DATA_DIR / "raw" / ("faithful.R" if dataset_id == "faithful" else dataset_id + ".zip")


def _source_url(dataset_id):
    s = SPECS[dataset_id]
    return s.get("source_url", f"{_UCI}static/public/{s.get('uci_id')}/{s.get('slug')}.zip")


def _ensure_source(path, url, expected, download=False):
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; restore the bundled cache or explicitly run prepare --download on a machine with internet access.")
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "FMVMM-reproducible-study/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Source changed at {url}; refusing to silently replace the frozen dataset.")
        path.write_bytes(content)
    if _sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _parse_raw(dataset_id):
    s = SPECS[dataset_id]
    path = _raw_path(dataset_id)
    _ensure_source(path, _source_url(dataset_id), s["sha256"])
    auxiliary = {}
    if dataset_id == "faithful":
        # Parse two numeric c(...) vectors; never execute downloaded R code.
        source = path.read_text()
        vectors = []
        for name in ("eruptions", "waiting"):
            match = re.search(rf"{name}\s*=\s*c\(([^)]*)\)", source, re.S)
            if match is None or re.search(r"[^\d\s,.eE+\-]", match.group(1)):
                raise ValueError("Unexpected faithful.R format")
            vectors.append(np.fromstring(match.group(1), sep=","))
        X = np.column_stack(vectors)
        y = None
    else:
        with zipfile.ZipFile(path) as archive:
            text = archive.read(s["member"]).decode("utf-8-sig").strip() + "\n"
        if dataset_id in ("seeds", "vertebral"):
            raw = np.loadtxt(io.StringIO(text), dtype=str)
        else:
            raw = np.loadtxt(io.StringIO(text), delimiter=",", dtype=str,
                             skiprows=1 if dataset_id == "wholesale" else 0)
        if dataset_id in ("iris", "seeds", "vertebral"):
            X, y = raw[:, :-1].astype(float), raw[:, -1]
        elif dataset_id == "wine":
            X, y = raw[:, 1:].astype(float), raw[:, 0]
        elif dataset_id == "glass":
            X, y = raw[:, 1:-1].astype(float), raw[:, -1]
            auxiliary["original_id"] = raw[:, 0]
        elif dataset_id == "wdbc":
            X, y = raw[:, 2:].astype(float), raw[:, 1]
            auxiliary["original_id"] = raw[:, 0]
        elif dataset_id == "wholesale":
            X, y = raw[:, 2:].astype(float), raw[:, 0]
            auxiliary["region"] = raw[:, 1]
    if X.shape != (s["n"], s["p"]) or not np.isfinite(X).all():
        raise ValueError(f"Unexpected shape or nonfinite data: {dataset_id} {X.shape}")
    return X, y, auxiliary


def prepare(download=False):
    """Verify pinned sources and rebuild all numeric caches and provenance."""
    for entry in AUXILIARY_SOURCES:
        _ensure_source(DATA_DIR / entry["path"], entry["url"], entry["sha256"], download)
    manifest = dict(schema_version=1, accessed_date=ACCESS_DATE,
                    description="Eight frozen public datasets; labels are external evaluation only.",
                    k_grid=list(K_GRID), holdout_fraction=0.2, repeats=5,
                    split_seeds=SPLIT_SEEDS, pca_max_dim=PCA_MAX_DIM,
                    auxiliary_sources=AUXILIARY_SOURCES, datasets={})
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    for dataset_id in DATASET_IDS:
        s = SPECS[dataset_id]
        _ensure_source(_raw_path(dataset_id), _source_url(dataset_id), s["sha256"], download)
        X, y, auxiliary = _parse_raw(dataset_id)
        arrays = dict(X=X, y=np.asarray([] if y is None else y, dtype=str),
                      feature_names=np.asarray(s["feature_names"], dtype=str),
                      source_row_ids=np.arange(1, X.shape[0] + 1))
        arrays.update({"aux_" + key: value for key, value in auxiliary.items()})
        cache_path = DATA_DIR / "processed" / f"{dataset_id}.npz"
        np.savez_compressed(cache_path, **arrays)
        if y is None:
            label_counts = None
        else:
            values, counts = np.unique(y, return_counts=True)
            label_counts = {str(v): int(c) for v, c in zip(values, counts)}
        manifest["datasets"][dataset_id] = dict(
            dataset_id=dataset_id, n=s["n"], p_raw=s["p"], p_model=min(s["p"], PCA_MAX_DIM),
            feature_names=s["feature_names"], label_name=s["label_name"], label_counts=label_counts,
            labels_used_for="external evaluation only; never for splits, transformations, K, screening, fitting or BIC",
            excluded_columns=s.get("excluded_columns", []),
            fixed_transform=s.get("transform", "identity"), row_policy="all rows, original source order; no imputation, deletion, jitter or subsampling",
            source_url=_source_url(dataset_id),
            documentation_url=s.get("documentation_url", f"{_UCI}dataset/{s.get('uci_id')}/{s.get('slug')}"),
            citation=s["citation"], doi=s.get("doi"),
            license="GPL-2.0-or-GPL-3.0 (R distribution)" if dataset_id == "faithful" else "CC-BY-4.0",
            license_url=_R + "doc/COPYRIGHTS" if dataset_id == "faithful" else "https://creativecommons.org/licenses/by/4.0/",
            raw_file=str(_raw_path(dataset_id).relative_to(DATA_DIR)), raw_sha256=s["sha256"], archive_member=s.get("member"),
            processed_file=str(cache_path.relative_to(DATA_DIR)), processed_sha256=_sha(cache_path),
            X_sha256=hashlib.sha256(np.asarray(X, dtype="<f8").tobytes()).hexdigest(),
            notes=s.get("notes", ""), split_seeds=SPLIT_SEEDS[dataset_id],
        )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return verify()


def load_real_data(dataset_id):
    """Return numeric raw features, optional external labels, and provenance.

    This is offline and checks the frozen source and processed hashes. No
    preprocessing is fitted here; use split_real_data for model inputs.
    """
    if dataset_id not in SPECS:
        raise ValueError(f"Unknown dataset_id {dataset_id!r}; choose from {DATASET_IDS}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    entry = manifest["datasets"][dataset_id]
    _ensure_source(_raw_path(dataset_id), _source_url(dataset_id), SPECS[dataset_id]["sha256"])
    path = DATA_DIR / entry["processed_file"]
    if _sha(path) != entry["processed_sha256"]:
        raise ValueError(f"Processed cache changed: {path}")
    with np.load(path, allow_pickle=False) as cache:
        X = cache["X"].copy()
        y = cache["y"].copy() if entry["label_name"] is not None else None
        result = dict(X=X, y=y, feature_names=cache["feature_names"].tolist(),
                      provenance=entry, source_row_ids=cache["source_row_ids"].copy(),
                      auxiliary={key[4:]: cache[key].copy() for key in cache.files if key.startswith("aux_")})
    if X.shape != (SPECS[dataset_id]["n"], SPECS[dataset_id]["p"]) or not np.isfinite(X).all():
        raise ValueError(f"Invalid numeric cache for {dataset_id}")
    return result


def _preprocess(X, train_index, test_index, fixed_transform="identity"):
    """Fit a fixed-dimensional representation using training features only."""
    X = np.asarray(X, dtype=float)
    if fixed_transform == "log1p":
        if np.any(X < 0):
            raise ValueError("Spending data must be nonnegative")
        transformed = np.log1p(X)
    elif fixed_transform == "identity":
        transformed = X
    else:
        raise ValueError(f"Unknown fixed transformation {fixed_transform}")
    train = transformed[train_index]
    test = transformed[test_index]
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("A training feature has zero variance; no silent feature deletion is allowed")
    Ztrain = (train - mean) / scale
    Ztest = (test - mean) / scale
    p = X.shape[1]
    dimension = min(p, PCA_MAX_DIM)
    components = np.eye(p)
    singular_values = None
    explained_variance_fraction = None
    if p > PCA_MAX_DIM:
        _, singular, vt = np.linalg.svd(Ztrain, full_matrices=False)
        components = vt[:dimension].copy()
        # Fix arbitrary singular-vector signs without labels or test data.
        for row in components:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        singular_values = singular.tolist()
        explained_variance_fraction = float(np.sum(singular[:dimension] ** 2) / np.sum(singular ** 2))
        Ztrain, Ztest = Ztrain @ components.T, Ztest @ components.T
    metadata = dict(fixed_transform=fixed_transform, standardization="training mean and population standard deviation (ddof=0)",
                    mean=mean.tolist(), scale=scale.tolist(), p_raw=p, p_model=dimension,
                    pca_applied=bool(p > PCA_MAX_DIM), pca_components=components.tolist(),
                    pca_dimension_rule="min(number of raw features, 5), fixed for every dataset",
                    pca_whiten=False, singular_values=singular_values,
                    explained_variance_fraction=explained_variance_fraction,
                    modeled_space="PCA projection of standardized features" if p > PCA_MAX_DIM else "standardized features",
                    scaler_log_abs_det_jacobian=float(-np.log(scale).sum()),
                    original_space_density_available=bool(p <= PCA_MAX_DIM),
                    original_space_log_abs_det_jacobian=float(-np.log(scale).sum()) if p <= PCA_MAX_DIM and fixed_transform == "identity" else None,
                    density_note="All predictive scores refer to the fitted feature space. A PCA projection discards coordinates and has no full-original-space density Jacobian.")
    return np.ascontiguousarray(Ztrain), np.ascontiguousarray(Ztest), metadata


def split_real_data(task):
    """Return Xtrain/Xtest, external ytrain/ytest, and fitted preprocessing.

    Required task keys: dataset_id (dataset alias accepted), split='full' or
    'holdout', repeat=0..4. fit_k and seed control later fits, never the split.
    An optional split_seed must equal the frozen seed for this dataset/repeat.
    """
    dataset_id = task.get("dataset_id", task.get("dataset"))
    data = load_real_data(dataset_id)
    split = task.get("split", "full")
    repeat = int(task.get("repeat", 0))
    n = data["X"].shape[0]
    if split == "full":
        if repeat != 0:
            raise ValueError("Full-data tasks use repeat=0")
        train_index, test_index = np.arange(n), np.array([], dtype=int)
        split_seed = None
    elif split == "holdout":
        if not 0 <= repeat < 5:
            raise ValueError("Holdout repeat must be in 0..4")
        split_seed = SPLIT_SEEDS[dataset_id][repeat]
        if "split_seed" in task and int(task["split_seed"]) != split_seed:
            raise ValueError("Task attempts to override the frozen split seed")
        permutation = np.random.default_rng(split_seed).permutation(n)
        ntest = int(np.ceil(0.2 * n))
        test_index = np.sort(permutation[:ntest])
        train_index = np.sort(permutation[ntest:])
    else:
        raise ValueError("split must be 'full' or 'holdout'")
    Xtrain, Xtest, preprocessing = _preprocess(data["X"], train_index, test_index,
                                             SPECS[dataset_id].get("transform", "identity"))
    preprocessing.update(dataset_id=dataset_id, split=split, repeat=repeat, split_seed=split_seed,
                         train_indices=train_index.tolist(), test_indices=test_index.tolist(),
                         source_row_ids_train=data["source_row_ids"][train_index].tolist(),
                         source_row_ids_test=data["source_row_ids"][test_index].tolist(),
                         n_train=len(train_index), n_test=len(test_index),
                         processed_sha256=data["provenance"]["processed_sha256"],
                         fitted_on="full data (descriptive fit)" if split == "full" else "training rows only")
    y = data["y"]
    result = dict(Xtrain=Xtrain, Xtest=Xtest,
                  ytrain=None if y is None else y[train_index],
                  ytest=None if y is None else y[test_index],
                  preprocessing=preprocessing, provenance=data["provenance"],
                  feature_names=data["feature_names"] if data["X"].shape[1] <= PCA_MAX_DIM else [f"PC{i+1}" for i in range(PCA_MAX_DIM)])
    return result


def proposed_real_tasks():
    """Metadata-only grid: 8 datasets × 4 K values × (full + 5 holdouts)."""
    tasks = []
    for index, dataset_id in enumerate(DATASET_IDS):
        for split, repeats in (("full", range(1)), ("holdout", range(5))):
            for repeat in repeats:
                for fit_k in K_GRID:
                    tasks.append(dict(dataset_id=dataset_id, split=split, repeat=repeat, fit_k=fit_k,
                                      seed=2026100000 + 1000 * index + (100 if split == "holdout" else 0) + 10 * repeat + fit_k,
                                      split_seed=None if split == "full" else SPLIT_SEEDS[dataset_id][repeat]))
    return tasks


def verify():
    """Offline audit: pinned sources, numeric equality, and all frozen splits."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    for entry in AUXILIARY_SOURCES:
        _ensure_source(DATA_DIR / entry["path"], entry["url"], entry["sha256"])
    checks = []
    for dataset_id in DATASET_IDS:
        data = load_real_data(dataset_id)
        rawX, rawy, auxiliary = _parse_raw(dataset_id)
        if not np.array_equal(rawX, data["X"]) or ((rawy is None) != (data["y"] is None)):
            raise ValueError(f"Parsed source differs from processed data: {dataset_id}")
        if rawy is not None and not np.array_equal(rawy, data["y"]):
            raise ValueError(f"Labels differ: {dataset_id}")
        if any(not np.array_equal(value, data["auxiliary"][key]) for key, value in auxiliary.items()):
            raise ValueError(f"Auxiliary metadata differ: {dataset_id}")
        entry = manifest["datasets"][dataset_id]
        if hashlib.sha256(np.asarray(rawX, dtype="<f8").tobytes()).hexdigest() != entry["X_sha256"]:
            raise ValueError(f"Numeric content checksum differs: {dataset_id}")
        for repeat in range(5):
            split = split_real_data(dict(dataset_id=dataset_id, split="holdout", repeat=repeat))
            m = split["preprocessing"]
            train, test = set(m["train_indices"]), set(m["test_indices"])
            if train & test or train | test != set(range(entry["n"])):
                raise ValueError(f"Invalid split: {dataset_id}/{repeat}")
            if not np.isfinite(split["Xtrain"]).all() or not np.isfinite(split["Xtest"]).all():
                raise ValueError(f"Nonfinite transformed features: {dataset_id}/{repeat}")
        checks.append(dict(dataset_id=dataset_id, n=entry["n"], p_raw=entry["p_raw"],
                           p_model=entry["p_model"], source_and_cache_match=True,
                           label_counts=entry["label_counts"], verified_holdout_splits=5))
    return dict(passed=True, datasets=checks, base_task_count=len(proposed_real_tasks()),
                network_used=False, reference_fmvmm_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "tasks"), nargs="?", default="verify")
    parser.add_argument("--download", action="store_true", help="Explicitly download missing pinned source files during prepare")
    args = parser.parse_args()
    if args.download and args.command != "prepare":
        parser.error("--download is allowed only with prepare")
    result = prepare(args.download) if args.command == "prepare" else proposed_real_tasks() if args.command == "tasks" else verify()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
