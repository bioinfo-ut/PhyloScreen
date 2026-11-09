"""Tests for saving and loading model bundles, including legacy 1.0.0 bundles."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phyloscreen.features import FEATURE_NAMES
from phyloscreen.model import Model, PhenotypeModel


def _fitted(seed=0) -> PhenotypeModel:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 3))
    y = (X[:, 0] > 0).astype(int)
    return PhenotypeModel(best_C=1.0, seed=seed).fit(X, y)


def test_save_load_roundtrip(tmp_path):
    refs = pd.DataFrame(
        {
            "path": ["a.fa", "b.fa", "c.fa"],
            "amp": pd.array([1, 0, None], dtype="Int64"),
            "cef": pd.array([None, 1, 0], dtype="Int64"),
        }
    )
    bundle = Model(
        models={"amp": _fitted(1), "cef": _fitted(2)},
        references=refs,
        metadata={"phenotypes": ["amp", "cef"], "kmer_size": 21, "sketch_size": 1000},
    )
    bundle.save(tmp_path)

    for name in ("model.joblib", "labels.tsv", "coefficients.json", "metadata.json"):
        assert (tmp_path / name).exists(), name

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["format_version"] == 2
    assert meta["phenotypes"] == ["amp", "cef"]
    assert meta["feature_names"] == FEATURE_NAMES
    assert meta["kmer_size"] == 21

    coefs = json.loads((tmp_path / "coefficients.json").read_text())
    assert set(coefs) == {"amp", "cef"}
    assert set(coefs["amp"]) == set(FEATURE_NAMES)

    # Blank cells for missing labels
    assert (tmp_path / "labels.tsv").read_text().splitlines()[3] == "c.fa\t\t0"

    loaded = Model.load(tmp_path)
    assert loaded.phenotypes == ["amp", "cef"]
    assert set(loaded.models) == {"amp", "cef"}
    assert str(loaded.references["amp"].dtype) == "Int64"
    assert loaded.references["amp"].isna().tolist() == [False, False, True]
    assert loaded.references["cef"].isna().tolist() == [True, False, False]

    X = np.zeros((1, 3))
    for p in ("amp", "cef"):
        np.testing.assert_allclose(loaded.models[p].predict_proba(X), bundle.models[p].predict_proba(X))


def test_legacy_bundle_loads_as_single_phenotype(tmp_path):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
    ])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    pipeline.fit(X, (X[:, 0] > 0).astype(int))
    joblib.dump(pipeline, tmp_path / "model.joblib")
    (tmp_path / "labels.tsv").write_text("path\tphenotype\na.fa\t1\nb.fa\t0\n")

    loaded = Model.load(tmp_path)

    assert loaded.phenotypes == ["phenotype"]
    assert loaded.metadata["format_version"] == 1
    assert list(loaded.references.columns) == ["path", "phenotype"]
    assert loaded.references["phenotype"].tolist() == [1, 0]
    assert loaded.models["phenotype"].classes_.tolist() == [0, 1]
