#!/usr/bin/env python3
"""Verify the paper's exported data offline; no fitting package is required."""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def read_csv(path, features):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = np.array([[float(row[name]) for name in features] for row in rows])
    return rows, values.reshape(len(rows), len(features))


def source_values(name, entry):
    path = HERE / entry["source"]
    if name == "faithful":
        text = path.read_text()
        columns = []
        for key in ("eruptions", "waiting"):
            match = re.search(rf"{key}\s*=\s*c\(([^)]*)\)", text, re.S)
            assert match is not None
            assert re.search(r"[^\d\s,.eE+\-]", match.group(1)) is None
            columns.append(np.fromstring(match.group(1), sep=","))
        return np.column_stack(columns), None, None
    with zipfile.ZipFile(path) as archive:
        text = archive.read(entry["archive_member"]).decode("utf-8-sig")
    raw = np.loadtxt(io.StringIO(text), dtype=str,
                     delimiter="," if name == "wdbc" else None)
    if name == "wdbc":
        return raw[:, 2:].astype(float), raw[:, 1], raw[:, 0]
    return raw[:, :-1].astype(float), raw[:, -1], None


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    cases = json.loads((HERE / "case_manifest.json").read_text())["cases"]
    assert len(cases) == 36
    assert set(manifest["datasets"]) == {"faithful", "seeds", "wdbc"}
    for relpath, entry in manifest["licenses"].items():
        assert sha(HERE / relpath) == entry["sha256"]
    description = manifest["R_datasets_package_description"]
    assert sha(HERE / description["path"]) == description["sha256"]
    checked = []
    for name, entry in manifest["datasets"].items():
        assert sha(HERE / entry["source"]) == entry["source_sha256"]
        assert sha(HERE / entry["csv"]) == entry["csv_sha256"]
        rows, X = read_csv(HERE / entry["csv"], entry["model_feature_columns"])
        expected, labels, source_ids = source_values(name, entry)
        assert X.shape == (entry["n"], entry["p_raw"])
        assert np.array_equal(X, expected)
        assert array_sha(X) == entry["X_float64_sha256"]
        assert [int(row["row_index"]) for row in rows] == list(range(len(X)))
        assert [int(row["source_row_id"]) for row in rows] == list(range(1, len(X)+1))
        if labels is not None:
            assert np.array_equal(labels, [row[entry["label_column"]] for row in rows])
        if source_ids is not None:
            assert np.array_equal(source_ids, [row["source_id"] for row in rows])
        for split_name, split in entry["splits"].items():
            prep = split["preprocessing"]
            if split["split"] == "full":
                train = np.arange(len(X))
                test = np.array([], dtype=int)
            else:
                permutation = np.random.default_rng(prep["split_seed"]).permutation(len(X))
                count = int(np.ceil(.2*len(X)))
                test, train = np.sort(permutation[:count]), np.sort(permutation[count:])
            assert np.array_equal(train, prep["train_indices"])
            assert np.array_equal(test, prep["test_indices"])
            assert not set(train) & set(test)
            assert len(train)+len(test) == len(X)
            mean, scale = X[train].mean(axis=0), X[train].std(axis=0, ddof=0)
            assert np.allclose(mean, prep["mean"], rtol=1e-13, atol=1e-13)
            assert np.allclose(scale, prep["scale"], rtol=1e-13, atol=1e-13)
            if prep["pca_applied"]:
                _, singular, vt = np.linalg.svd((X[train]-mean)/scale, full_matrices=False)
                components = vt[:prep["p_model"]].copy()
                for row in components:
                    if row[np.argmax(np.abs(row))] < 0:
                        row *= -1
                assert np.allclose(components, prep["pca_components"], rtol=1e-10, atol=1e-10)
                assert np.allclose(singular, prep["singular_values"], rtol=1e-10, atol=1e-10)
            for subset, indices in (("train", train), ("test", test)):
                file = split["files"][subset]
                path = HERE / file["path"]
                assert sha(path) == file["sha256"]
                output_rows, values = read_csv(path, file["feature_columns"])
                assert array_sha(values) == file["X_float64_sha256"]
                assert [int(row["row_index"]) for row in output_rows] == indices.tolist()
                if labels is not None:
                    assert np.array_equal(labels[indices], [row[entry["label_column"]] for row in output_rows])
                reconstructed = (X[indices]-np.array(prep["mean"]))/np.array(prep["scale"])
                if prep["pca_applied"]:
                    reconstructed = reconstructed @ np.array(prep["pca_components"]).T
                assert np.allclose(reconstructed, values, rtol=1e-12, atol=1e-12)
                assert values.shape == (len(indices), prep["p_model"])
            checked.append({"dataset": name, "split": split_name,
                            "n_train": len(train), "n_test": len(test)})
    sums = HERE / "SHA256SUMS"
    if sums.exists():
        for line in sums.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            assert sha(HERE / relative) == expected, relative
    print(json.dumps({"passed": True, "network_used": False, "datasets": 3,
                      "distinct_splits": len(checked), "cases": len(cases),
                      "checks": checked}, indent=2))


if __name__ == "__main__":
    main()
