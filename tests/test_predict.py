"""Prediction tests with mash screen faked (see conftest.py)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phyloscreen import predict as predict_mod
from phyloscreen.predict import OUTPUT_COLUMNS, predict_phenotype
from tests.conftest import make_fake_screen


def test_predict_one_row_per_phenotype(trained_bundle, monkeypatch):
    out, panel, _ = trained_bundle
    monkeypatch.setattr(predict_mod, "screen", make_fake_screen(panel, {"amp": 1, "cef": 1}))

    query = panel.dir / "query.fa"  # novel genome, hidden truth: amp=1, cef=1
    query.write_text(">q\nACGT\n")
    result_path = panel.dir / "pred.tsv"
    results = predict_phenotype(query, out, result_path, threads=1)

    assert [r.phenotype for r in results] == ["amp", "cef"]
    for r in results:
        assert r.prediction in (0, 1)
        assert 0.0 <= r.probability <= 1.0
        assert r.prob_1 + r.prob_0 == pytest.approx(1.0)
        assert r.top_hit_1 is not None and r.top_hit_0 is not None
        assert r.top_hit_1_shared_hashes is not None

    # Signal in the fake similarities is strong enough to recover the truth
    assert results[0].prediction == 1
    assert results[1].prediction == 1

    df = pd.read_csv(result_path, sep="\t")
    assert list(df.columns) == OUTPUT_COLUMNS
    assert df["phenotype"].tolist() == ["amp", "cef"]
    assert (df["query_path"] == str(query)).all()


def test_predict_respects_missing_labels(trained_bundle, monkeypatch):
    """g02 and g03 have no 'cef' label: they must never be a cef top hit."""
    out, panel, _ = trained_bundle
    monkeypatch.setattr(predict_mod, "screen", make_fake_screen(panel))

    # g02 is one of g03's closest neighbours (same amp cluster); both are cef-unlabelled.
    results = predict_phenotype(Path(panel.paths[3]), out, panel.dir / "p.tsv", threads=1)
    cef = next(r for r in results if r.phenotype == "cef")
    unlabelled = {panel.paths[2], panel.paths[3]}
    assert cef.top_hit_1 not in unlabelled
    assert cef.top_hit_0 not in unlabelled


def test_predict_empty_screen_gives_unknown(trained_bundle, monkeypatch):
    out, panel, _ = trained_bundle
    monkeypatch.setattr(predict_mod, "screen", lambda *a, **k: pd.DataFrame())

    results = predict_phenotype(Path(panel.paths[0]), out, panel.dir / "e.tsv", threads=1)

    assert [r.prediction for r in results] == ["unknown", "unknown"]
    df = pd.read_csv(panel.dir / "e.tsv", sep="\t")
    assert len(df) == 2
    assert df["prediction"].tolist() == ["unknown", "unknown"]


def test_predict_legacy_bundle(tmp_path, monkeypatch):
    refs = ["a.fa", "b.fa", "c.fa", "d.fa"]
    labels = [1, 0, 1, 0]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
    ])
    X = np.array([[1, 2.0, 0.9], [0, 0.5, 0.1], [1, 1.5, 0.8], [0, 0.7, 0.2]])
    pipeline.fit(X, labels)
    joblib.dump(pipeline, tmp_path / "model.joblib")
    (tmp_path / "labels.tsv").write_text("path\tphenotype\n" + "".join(f"{r}\t{l}\n" for r, l in zip(refs, labels)))
    (tmp_path / "reference_sketch.msh").write_bytes(b"")

    screen_df = pd.DataFrame(
        {
            "identity": [0.99, 0.9, 0.8, 0.7],
            "shared_hashes": [900, 700, 500, 300],
            "total_hashes": [1000] * 4,
            "median_multiplicity": [1.0] * 4,
            "p_value": [0.0] * 4,
            "query_comment": [""] * 4,
            "reference_path": refs,
        }
    )
    monkeypatch.setattr(predict_mod, "screen", lambda *a, **k: screen_df)

    query = tmp_path / "q.fa"
    query.write_text(">q\nACGT\n")
    results = predict_phenotype(query, tmp_path, tmp_path / "out.tsv")

    assert len(results) == 1
    assert results[0].phenotype == "phenotype"
    assert results[0].prediction == 1
    assert results[0].top_hit_1 == "a.fa" and results[0].top_hit_0 == "b.fa"
    df = pd.read_csv(tmp_path / "out.tsv", sep="\t")
    assert df["phenotype"].tolist() == ["phenotype"]
