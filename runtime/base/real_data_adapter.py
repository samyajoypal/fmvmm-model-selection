"""Four frozen public datasets for the full-family paper study.

The representation is specified before screening or comparing mixtures.  Every
estimated preprocessing quantity uses training features only.  Labels are
returned solely for external descriptive evaluation; they are never inspected
when splitting or transforming observations.  This module does not modify the
original eight-dataset archive or its original five-dimensional analysis.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np


STUDY_ROOT = Path(os.environ.get("FMVMM_STUDY_ROOT", Path(__file__).resolve().parent.parent))
_spec = importlib.util.spec_from_file_location("_paper_frozen_real_data", STUDY_ROOT / "real_data.py")
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load the frozen real-data archive")
_archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_archive)
_load_real_data = _archive.load_real_data

DATASET_IDS = ("faithful", "seeds", "wdbc", "wholesale")
MODEL_DIMENSIONS = {"faithful": 2, "seeds": 3, "wdbc": 3, "wholesale": 3}
HOLDOUT_REPEATS = 3
HOLDOUT_FRACTION = 0.2
K_GRID = (1, 2, 3)
SPLIT_SEEDS = {name: tuple(_archive.SPLIT_SEEDS[name][:HOLDOUT_REPEATS]) for name in DATASET_IDS}
PROTOCOL_VERSION = "four-dataset-three-PC-v1"


def _array_sha256(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def transform_features(X, train_indices, test_indices, dimension, fixed_transform="identity"):
    """Return training/test arrays and the entire fitted transformation.

    Three leading principal components are used for Seeds, WDBC and Wholesale;
    Faithful retains both original coordinates.  PCA is unwhitened.  The
    downstream FMVMM runner may subsequently whiten the modeled coordinates
    using its common training-only transformation.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Features must be a finite two-dimensional array")
    if not 1 <= dimension <= X.shape[1]:
        raise ValueError("Invalid fixed representation dimension")
    if fixed_transform == "log1p":
        if np.any(X < 0):
            raise ValueError("Wholesale spending must be nonnegative")
        transformed = np.log1p(X)
    elif fixed_transform == "identity":
        transformed = X
    else:
        raise ValueError(f"Unsupported transformation: {fixed_transform}")
    train, test = transformed[train_indices], transformed[test_indices]
    mean, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("A training feature has zero or invalid variance")
    Ztrain, Ztest = (train - mean) / scale, (test - mean) / scale
    pca_applied = dimension < X.shape[1]
    components = np.eye(X.shape[1])
    singular_values, explained = None, None
    if pca_applied:
        _, singular, vt = np.linalg.svd(Ztrain, full_matrices=False)
        if len(singular) < dimension or singular[dimension - 1] <= 0:
            raise ValueError("Training data cannot support the fixed PCA dimension")
        components = vt[:dimension].copy()
        for row in components:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        Ztrain, Ztest = Ztrain @ components.T, Ztest @ components.T
        singular_values = singular.tolist()
        explained = float(np.sum(singular[:dimension] ** 2) / np.sum(singular ** 2))
    preprocessing = dict(
        protocol_version=PROTOCOL_VERSION, fixed_transform=fixed_transform,
        standardization="training mean and population standard deviation (ddof=0)",
        mean=mean.tolist(), scale=scale.tolist(), p_raw=int(X.shape[1]),
        p_model=int(dimension), pca_applied=pca_applied,
        pca_dimension_rule="Faithful: both coordinates; Seeds, WDBC, Wholesale: three leading PCs",
        pca_components=components.tolist(), pca_whiten=False,
        singular_values=singular_values, explained_variance_fraction=explained,
        modeled_space="PCA projection of standardized features" if pca_applied else "standardized features",
        original_space_density_available=not pca_applied,
        original_space_log_abs_det_jacobian=float(-np.log(scale).sum()) if not pca_applied and fixed_transform == "identity" else None,
        density_note="Predictive scores are densities in the fixed modeled feature space. PCA projections do not define a density in the complete original feature space.",
    )
    return np.ascontiguousarray(Ztrain), np.ascontiguousarray(Ztest), preprocessing


def prepare_real_data(task):
    """Load one preregistered split, using the native runner's data schema.

    Accepts dataset_id, split ('full' or 'holdout'), repeat (0, 1 or 2), and
    optionally the matching frozen split_seed. fit_k and seed affect fitting,
    not the split or preprocessing. No network access occurs.
    """
    dataset_id = task.get("dataset_id", task.get("dataset"))
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"Dataset {dataset_id!r} is outside the four-dataset protocol")
    if "p" in task and task["p"] != MODEL_DIMENSIONS[dataset_id]:
        raise ValueError("Task dimension differs from the fixed dataset representation")
    data = _load_real_data(dataset_id)
    n = len(data["X"])
    split, repeat = task.get("split", "full"), int(task.get("repeat", 0))
    if split == "full":
        if repeat != 0 or task.get("split_seed") is not None:
            raise ValueError("Full-data tasks require repeat=0 and no split seed")
        train_indices, test_indices, split_seed = np.arange(n), np.array([], dtype=int), None
    elif split == "holdout":
        if not 0 <= repeat < HOLDOUT_REPEATS:
            raise ValueError("Holdout repeat is outside the frozen 0..2 range")
        split_seed = SPLIT_SEEDS[dataset_id][repeat]
        if "split_seed" in task and task["split_seed"] != split_seed:
            raise ValueError("Task attempts to override the frozen split seed")
        permutation = np.random.default_rng(split_seed).permutation(n)
        ntest = int(np.ceil(HOLDOUT_FRACTION * n))
        test_indices, train_indices = np.sort(permutation[:ntest]), np.sort(permutation[ntest:])
    else:
        raise ValueError("split must be 'full' or 'holdout'")
    dimension = MODEL_DIMENSIONS[dataset_id]
    Xtrain, Xtest, preprocessing = transform_features(
        data["X"], train_indices, test_indices, dimension,
        data["provenance"]["fixed_transform"],
    )
    preprocessing.update(
        dataset_id=dataset_id, split=split, repeat=repeat, split_seed=split_seed,
        train_indices=train_indices.tolist(), test_indices=test_indices.tolist(),
        source_row_ids_train=data["source_row_ids"][train_indices].tolist(),
        source_row_ids_test=data["source_row_ids"][test_indices].tolist(),
        n_train=len(train_indices), n_test=len(test_indices),
        processed_sha256=data["provenance"]["processed_sha256"],
        Xtrain_sha256=_array_sha256(Xtrain), Xtest_sha256=_array_sha256(Xtest),
        fitted_on="full data (descriptive fit)" if split == "full" else "training rows only",
    )
    y = data["y"]
    feature_names = data["feature_names"] if not preprocessing["pca_applied"] else [f"PC{i + 1}" for i in range(dimension)]
    return dict(
        Xtrain=Xtrain, Xtest=Xtest,
        ytrain=None if y is None else y[train_indices],
        ytest=None if y is None else y[test_indices],
        preprocessing=preprocessing, provenance=data["provenance"],
        feature_names=feature_names,
        metadata=dict(dataset_id=dataset_id, preprocessing=preprocessing,
                      provenance=data["provenance"], ground_truth_available=False,
                      labels_used_for="external descriptive evaluation only"),
    )


split_real_data = prepare_real_data
load_task = prepare_real_data


def proposed_real_tasks(k_grid=K_GRID):
    """Return the fixed data splits with caller-declared candidate K values."""
    result = []
    for index, dataset_id in enumerate(DATASET_IDS):
        for split, repeats in (("full", (0,)), ("holdout", range(HOLDOUT_REPEATS))):
            for repeat in repeats:
                for fit_k in k_grid:
                    if not isinstance(fit_k, int) or fit_k < 1:
                        raise ValueError("Candidate K values must be positive integers")
                    result.append(dict(
                        dataset_id=dataset_id, split=split, repeat=repeat,
                        fit_k=fit_k, p=MODEL_DIMENSIONS[dataset_id], library_id="L13",
                        seed=2026170000 + 1000 * index + (100 if split == "holdout" else 0) + 10 * repeat + fit_k,
                        split_seed=None if split == "full" else SPLIT_SEEDS[dataset_id][repeat],
                    ))
    return result
